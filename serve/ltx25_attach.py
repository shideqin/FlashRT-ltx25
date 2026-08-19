"""LTX-2.5 via the structures layer (PR 174): attach, gate, measure.

Host: the unmodified Diffusers LTX2VideoTransformer3DModel loaded from the
official bf16 split checkpoint (single-file). ``forward`` is one real
denoise step on the distilled single-pass schedule inputs (patchified video
+ audio latents, connector-shaped text embeddings, scaled timestep).

Usage::

    python serve/ltx25_attach.py --res 768x512x49f [--scheme nvfp4_balance_sage]
    python serve/ltx25_attach.py --res 1536x1024x121f [--scheme nvfp4_balance_sage]

Deterministic inputs (fixed seed), no_grad throughout. Reports the plan
(report()), the gate verdicts, host-vs-attached paired latency, matched-input
cosine, and detach exactness (max-abs 0.0).
"""

import argparse
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/workspace/data/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch

sys.path.insert(0, "/workspace/data/FlashRT")

CKPT = "/workspace/data/models/LTX-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
CONFIG_DIR = "/workspace/data/models/LTX-2.5/diffusers_transformer"

# latent token counts per resolution (temporal stride 8, spatial 32)
RES = {
    "768x512x49f": dict(frames=7, height=16, width=24, text=16, audio_frames=128),
    "1536x1024x121f": dict(frames=16, height=32, width=48, text=16, audio_frames=128),
}


def load_model():
    from diffusers.models.transformers.transformer_ltx2 import (
        LTX2VideoTransformer3DModel,
    )
    model = LTX2VideoTransformer3DModel.from_single_file(
        CKPT, config=CONFIG_DIR, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False)
    model = model.to("cuda").eval()
    return model


class DenoiseStep:
    """One distilled single-pass denoise step on fixed, seeded inputs."""

    def __init__(self, model, res_name, seed=0):
        spec = RES[res_name]
        torch.manual_seed(seed)
        dev = next(model.parameters()).device
        dt = torch.bfloat16
        frames, h, w, text, af = (spec["frames"], spec["height"],
                                  spec["width"], spec["text"],
                                  spec["audio_frames"])
        S = frames * h * w
        self.video = torch.randn(1, S, 128, device=dev, dtype=dt)
        self.audio = torch.randn(1, af, 128, device=dev, dtype=dt)
        self.enc = torch.randn(1, text, 4096, device=dev, dtype=dt)
        self.aenc = torch.randn(1, text, 2048, device=dev, dtype=dt)
        self.t = torch.tensor([500.0], device=dev, dtype=dt)
        self.kw = dict(
            hidden_states=self.video, audio_hidden_states=self.audio,
            encoder_hidden_states=self.enc,
            audio_encoder_hidden_states=self.aenc,
            timestep=self.t, sigma=self.t, num_frames=frames, height=h,
            width=w, fps=24.0, audio_num_frames=af, return_dict=False)
        self.model = model

    def __call__(self):
        with torch.no_grad():
            return self.model(**self.kw)


def timed(fn, iters=5, warmup=2):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="768x512x49f")
    ap.add_argument("--scheme", default="nvfp4_balance_sage")
    ap.add_argument("--attention_forms", default=None)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--skip-attach", action="store_true")
    args = ap.parse_args()

    from flash_rt import structures

    print(f"[ltx25] loading {CKPT} ...", flush=True)
    model = load_model()
    print(f"[ltx25] params {sum(p.numel() for p in model.parameters())/1e9:.1f}B "
          f"mem {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    step = DenoiseStep(model, args.res)
    host_out = step()
    torch.cuda.synchronize()
    host_ms = timed(step, iters=3)
    print(f"[ltx25] host forward: {host_ms:.1f} ms", flush=True)

    if args.skip_attach:
        return

    kw = dict(scheme=args.scheme, rounds=args.rounds, iters=args.iters)
    if args.attention_forms:
        kw["attention_forms"] = tuple(args.attention_forms.split(","))
    print(f"[ltx25] structures.attach({args.res}, scheme={args.scheme}, "
          f"{kw})", flush=True)
    t0 = time.time()
    plan = structures.attach(model, step, verbose=True, **kw)
    print(f"[ltx25] attach done in {time.time()-t0:.1f}s", flush=True)

    print("\n=== PLAN REPORT ===", flush=True)
    print(plan.report(), flush=True)

    print("\n=== VERIFY ===", flush=True)
    attached_out = step()
    torch.cuda.synchronize()
    attached_ms = timed(step, iters=args.iters)
    cos = torch.nn.functional.cosine_similarity(
        host_out[0].float().reshape(-1), attached_out[0].float().reshape(-1),
        dim=0)
    print(f"host {host_ms:.1f} ms | attached {attached_ms:.1f} ms | "
          f"speedup {host_ms/attached_ms:.2f}x | cosine {float(cos):.6f}",
          flush=True)
    print(f"peak mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB",
          flush=True)

    print("\n=== DETACH ===", flush=True)
    plan.detach()
    torch.cuda.empty_cache()
    out = step()
    torch.cuda.synchronize()
    maxabs = float((out[0].float() - host_out[0].float()).abs().max())
    cos_restored = torch.nn.functional.cosine_similarity(
        host_out[0].float().reshape(-1), out[0].float().reshape(-1), dim=0)
    print(f"detach max-abs vs original host: {maxabs:.6f} "
          f"cosine {float(cos_restored):.8f} "
          f"(0.0 / 1.0 = bit-exact restore)", flush=True)


if __name__ == "__main__":
    main()
