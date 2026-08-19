"""MiniMax-H3 (pruned NVFP4) DiT host: load, sample, serve.

Host = ComfyUI's MiniMaxH3Model implementation (the checkpoint's own
host), loaded through ComfyUI's loader so the baseline arm runs exactly
the NVFP4 per-operator path the checkpoint ships with. This module is
shared by both HTTP services; the only difference between the arms is
whether flash_rt.structures binds seams into the module tree.
"""

import math
import os
import time

import torch

CKPT = os.environ.get("MINIMAX_CKPT",
                      "/data/models/minimax_h3_ref2va_pruned_nvfp4.safetensors")

# 480p, 39 frames @ 24 fps == the README's measured configuration
DEFAULT_FRAMES = int(os.environ.get("H3_FRAMES", "39"))
DEFAULT_FPS = 24.0
DEFAULT_STEPS = int(os.environ.get("H3_STEPS", "20"))
DEFAULT_TEXT_LEN = int(os.environ.get("H3_TEXT_LEN", "256"))
# latent dims after the 8x video VAE at 864x480
LATENT_H = 60
LATENT_W = 108
AUDIO_CH = 2
AUDIO_FPS = 40  # audio latent frames per second
VIDEO_C = 24
AUDIO_C = 32
SIGMA_SHIFT_V = 12.0
SIGMA_SHIFT_A = 3.0
AUDIO_SCALE = SIGMA_SHIFT_V / SIGMA_SHIFT_A


def time_shift_sigma(sigma, from_shift, to_shift):
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def build_sigmas(steps: int, shift: float = SIGMA_SHIFT_V,
                 timesteps: int = 1000) -> torch.Tensor:
    """ComfyUI 'simple' scheduler over the flow sigma grid."""
    # flow: sigma(t) = 1 - t, t on [0,1]; shifted by `shift`
    def sigma_of(t):
        return 1.0 - t * shift / (1.0 + (shift - 1.0) * t)

    grid = torch.linspace(0.0, 1.0, timesteps + 1)
    sigs = []
    ss = timesteps / steps
    for x in range(steps):
        idx = timesteps - int(round(x * ss))
        sigs.append(float(sigma_of(grid[idx].item())))
    sigs.append(0.0)
    return torch.FloatTensor(sigs)


def res_multistep(model, x, sigmas, ctx, payload, cb=None,
                  skip_steps=None):
    """ComfyUI res_multistep (eta=0) over the packed [video, audio] latent.

    Mirrors comfy/k_diffusion/sampling.py:res_multistep with cfg_pp off
    and eta=0 (deterministic second-order multistep). ``skip_steps``
    enables zeroth-order TeaCache (mirroring the Motus/Cosmos3/
    MiniMax-Remover mechanism): those steps reuse the cached velocity
    from the last computed step and run a cheap Euler update instead of
    the transformer. The first and last steps are never skipped.
    """
    s_in = x[0].new_ones([1])
    t_fn = lambda t: t.log().neg()
    phi1_fn = lambda t: torch.expm1(t) / t
    phi2_fn = lambda t: (phi1_fn(t) - 1.0) / t
    old_sigma_down = None
    old_denoised = None
    cached_denoised = None
    times = []
    skip_set = set(skip_steps or ())
    skip_set.discard(0)
    if len(sigmas) > 2:
        skip_set.discard(len(sigmas) - 2)
    for i in range(len(sigmas) - 1):
        t0 = time.perf_counter()
        if i in skip_set and cached_denoised is not None:
            denoised = cached_denoised
            computed = False
        else:
            denoised = model(x, sigmas[i] * s_in, ctx,
                             transformer_options={},
                             minimax_payload=payload)
            cached_denoised = denoised
            computed = True
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        sigma_down = sigmas[i + 1]
        if not computed:
            # TeaCache: Euler update from the cached velocity.
            d = (x[0] - denoised[0], x[1] - denoised[1])
            dt = sigma_down - sigmas[i]
            x = [x[0] + d[0] * dt, x[1] + d[1] * dt]
        elif sigma_down == 0 or old_denoised is None:
            d = (x[0] - denoised[0], x[1] - denoised[1])
            dt = sigma_down - sigmas[i]
            x = [x[0] + d[0] * dt, x[1] + d[1] * dt]
        else:
            t, t_old = t_fn(sigmas[i]), t_fn(old_sigma_down)
            t_next = t_fn(sigma_down)
            t_prev = t_fn(sigmas[i - 1])
            h = t_next - t
            c2 = (t_prev - t_old) / h
            phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
            b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)
            sf = torch.exp(-h)
            x0n = sf * x[0] + h * (b1 * denoised[0] + b2 * old_denoised[0])
            x1n = sf * x[1] + h * (b1 * denoised[1] + b2 * old_denoised[1])
            x = [x0n, x1n]
        old_denoised = denoised
        old_sigma_down = sigma_down
        if cb is not None:
            cb(i, times[-1], x)
    return x, times


class MiniMaxHost:
    """One loaded MiniMax-H3 DiT + sampling loop."""

    def __init__(self, frames=DEFAULT_FRAMES, text_len=DEFAULT_TEXT_LEN,
                 steps=DEFAULT_STEPS, fps=DEFAULT_FPS, seed=0,
                 audio_scale=AUDIO_SCALE):
        self.frames = frames
        self.text_len = text_len
        self.steps = steps
        self.fps = fps
        self.seed = seed
        self.audio_scale = audio_scale
        self.audio_t = int(round(frames * AUDIO_FPS / fps))  # 40Hz audio
        self.model = None

    # ---- loading ----------------------------------------------------
    def load(self, ckpt=CKPT, device="cuda"):
        import safetensors.torch
        from comfy.model_detection import model_config_from_unet

        print(f"[host] loading {ckpt}", flush=True)
        t0 = time.perf_counter()
        sd = safetensors.torch.load_file(ckpt, device="cpu")
        print(f"[host] state dict: {len(sd)} tensors in "
              f"{time.perf_counter() - t0:.1f}s", flush=True)

        config = model_config_from_unet(sd, "")
        assert config is not None, "MiniMax-H3 detection failed"
        config.set_inference_dtype(torch.bfloat16, torch.bfloat16,
                                   device=device)
        base = config.get_model(sd, "")
        base.load_model_weights(sd, "")
        self.model = base.diffusion_model.to(device).eval()
        self.base = base
        print(f"[host] model on {device}; params="
              f"{sum(p.numel() for p in self.model.parameters())/1e9:.1f}B",
              flush=True)
        torch.cuda.empty_cache()
        return self

    # ---- inputs -----------------------------------------------------
    def inputs(self, seed=None):
        g = torch.Generator(device="cpu").manual_seed(
            seed if seed is not None else self.seed)
        video = torch.randn(1, VIDEO_C, self.frames, LATENT_H, LATENT_W,
                            generator=g, dtype=torch.bfloat16)
        g2 = torch.Generator(device="cpu").manual_seed(
            (seed if seed is not None else self.seed) + 7)
        audio = torch.randn(1, AUDIO_C, AUDIO_CH, self.audio_t,
                            generator=g2, dtype=torch.bfloat16)
        g3 = torch.Generator(device="cpu").manual_seed(
            (seed if seed is not None else self.seed) + 13)
        ctx = torch.randn(1, self.text_len, 5120, generator=g3,
                          dtype=torch.bfloat16)
        return video, audio, ctx

    def payload(self):
        return {"audio_scale": self.audio_scale, "seed": self.seed}

    def run(self, steps=None, seed=None, warmup=0, skip_steps=None):
        """Full denoise; returns (latents, per-step seconds)."""
        model = self.model
        steps = steps or self.steps
        video, audio, ctx = self.inputs(seed)
        dev = next(model.parameters()).device
        video, audio, ctx = (video.to(dev), audio.to(dev), ctx.to(dev))
        sigmas = build_sigmas(steps).to(dev)
        x = [video, audio]
        payload = self.payload()
        for _ in range(warmup):
            with torch.no_grad():
                model(x, sigmas[0] * torch.ones([1], device=dev), ctx,
                      transformer_options={}, minimax_payload=payload)
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        with torch.no_grad():
            out, times = res_multistep(model, x, sigmas, ctx, payload,
                                       skip_steps=skip_steps)
        return out, times

    def rtf(self, times):
        total = sum(times)
        dur = self.frames / self.fps
        return total, total / self.steps, total / dur
