"""LTX-2.5 diffusers-host denoise flow: baseline vs structures attach.

Same host (Diffusers LTX2VideoTransformer3DModel, official bf16 checkpoint).
Stage-1 latent shape (half res) and stage-2 latent shape (full res) are both
real; the 8-stage1 + 3-stage2 denoise is timed per shape, baseline vs
attached (scheme="nvfp4_balance_sage"). RTF = denoise wall / video duration.

Usage: python serve/ltx25_diffusers_flow.py --res 768x512x49f
"""

import argparse
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/workspace/data/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch  # noqa: E402

sys.path.insert(0, "/workspace/data/FlashRT")

CKPT = "/workspace/data/models/LTX-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
CONFIG_DIR = "/workspace/data/models/LTX-2.5/diffusers_transformer"

# per-resolution: (stage1 latent, stage2 latent) token geometry
# stage1 = half-res spatial, same temporal stride 8; stage2 = full-res
RES = {
    "768x512x49f": dict(
        s1=dict(frames=7, height=8, width=12, text=16, audio_frames=128),
        s2=dict(frames=7, height=16, width=24, text=16, audio_frames=128)),
    "1536x1024x121f": dict(
        s1=dict(frames=16, height=16, width=24, text=16, audio_frames=128),
        s2=dict(frames=16, height=32, width=48, text=16, audio_frames=128)),
}


def load_model():
    from diffusers.models.transformers.transformer_ltx2 import (
        LTX2VideoTransformer3DModel,
    )
    model = LTX2VideoTransformer3DModel.from_single_file(
        CKPT, config=CONFIG_DIR, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False)
    return model.to("cuda").eval()


def make_step(model, spec, seed=0):
    torch.manual_seed(seed)
    dev = next(model.parameters()).device
    dt = torch.bfloat16
    frames, h, w, text, af = (spec["frames"], spec["height"], spec["width"],
                              spec["text"], spec["audio_frames"])
    S = frames * h * w
    video = torch.randn(1, S, 128, device=dev, dtype=dt)
    audio = torch.randn(1, af, 128, device=dev, dtype=dt)
    enc = torch.randn(1, text, 4096, device=dev, dtype=dt)
    aenc = torch.randn(1, text, 2048, device=dev, dtype=dt)
    t = torch.tensor([500.0], device=dev, dtype=dt)
    kw = dict(
        hidden_states=video, audio_hidden_states=audio,
        encoder_hidden_states=enc, audio_encoder_hidden_states=aenc,
        timestep=t, sigma=t, num_frames=frames, height=h, width=w,
        fps=24.0, audio_num_frames=af, return_dict=False)
    return lambda: model(**kw)


def time_steps(fn, n):
    with torch.no_grad():
        fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(n):
            fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / 1000 / n  # s per step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="768x512x49f")
    ap.add_argument("--scheme", default="nvfp4_balance_sage")
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()
    h, w, f = (int(x) for x in args.res.replace("f", "").split("x"))
    video_s = f / 24.0
    specs = RES[args.res]

    print(f"[df] loading {CKPT} ...", flush=True)
    model = load_model()
    print(f"[df] params {sum(p.numel() for p in model.parameters())/1e9:.1f}B",
          flush=True)

    # baseline: 8 x stage1 + 3 x stage2
    t1 = time_steps(make_step(model, specs["s1"], 0), args.iters)
    t2 = time_steps(make_step(model, specs["s2"], 1), args.iters)
    base = t1 * 8 + t2 * 3
    print(f"[df] baseline: s1 {t1*1000:.1f}ms x8 + s2 {t2*1000:.1f}ms x3 = "
          f"{base:.2f}s | RTF {base/video_s:.1f}x", flush=True)

    # attach (calibrate on BOTH stage shapes; family binds each it serves)
    from flash_rt import structures
    step1 = make_step(model, specs["s1"], 0)
    step2 = make_step(model, specs["s2"], 0)
    print(f"[df] structures.attach(scheme={args.scheme}, 2 shapes) ...",
          flush=True)
    t0 = time.time()
    plan = structures.attach(model, [step1, step2], verbose=True,
                             scheme=args.scheme, rounds=1, iters=args.iters)
    print(f"[df] attach done {time.time()-t0:.0f}s", flush=True)
    print(plan.report(), flush=True)

    t1a = time_steps(make_step(model, specs["s1"], 0), args.iters)
    t2a = time_steps(make_step(model, specs["s2"], 1), args.iters)
    att = t1a * 8 + t2a * 3
    print(f"[df] attached: s1 {t1a*1000:.1f}ms x8 + s2 {t2a*1000:.1f}ms x3 = "
          f"{att:.2f}s | RTF {att/video_s:.1f}x | speedup {base/att:.2f}x",
          flush=True)

    # parity on stage-2 shape, then detach restore
    torch.manual_seed(7)
    kw = {}
    spec = specs["s2"]
    dev = next(model.parameters()).device
    dt = torch.bfloat16
    S = spec["frames"] * spec["height"] * spec["width"]
    kw = dict(
        hidden_states=torch.randn(1, S, 128, device=dev, dtype=dt),
        audio_hidden_states=torch.randn(1, spec["audio_frames"], 128,
                                        device=dev, dtype=dt),
        encoder_hidden_states=torch.randn(1, spec["text"], 4096,
                                          device=dev, dtype=dt),
        audio_encoder_hidden_states=torch.randn(1, spec["text"], 2048,
                                                device=dev, dtype=dt),
        timestep=torch.tensor([500.0], device=dev, dtype=dt),
        sigma=torch.tensor([500.0], device=dev, dtype=dt),
        num_frames=spec["frames"], height=spec["height"], width=spec["width"],
        fps=24.0, audio_num_frames=spec["audio_frames"], return_dict=False)
    with torch.no_grad():
        a = model(**kw)[0]
        torch.cuda.synchronize()
    plan.detach()
    torch.cuda.empty_cache()
    with torch.no_grad():
        b = model(**kw)[0]
        torch.cuda.synchronize()
    cos = torch.nn.functional.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0)
    md = float((a.float() - b.float()).abs().max())
    print(f"[df] parity attached-vs-host: cosine {float(cos):.6f} "
          f"max|d| {md:.3f}", flush=True)


if __name__ == "__main__":
    main()
