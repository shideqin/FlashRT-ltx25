"""LTX-2.5 bit-exact RTF harness: eager + resident transformer + prompt cache.

Numerically-exact accelerations only:
1. prompt embedding cache (same prompt -> same tensor, skip Gemma-4 forward)
2. resident transformer (same weights, same forward, skip per-infer rebuild)
No kernel swap, no torch.compile, no CUDA graph capture. Output must be
bit-identical to the unmodified baseline.

Usage::

    python serve/ltx25_rtf_exact.py --res 768x512x49f
"""

import argparse
import os
import sys

os.environ.setdefault("FLASH_RT_LTX2_ROOT", "/workspace/data/LTX-2")
os.environ.setdefault("HF_HOME", "/workspace/data/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, "/workspace/data/FlashRT")

import flash_rt  # noqa: E402
from flash_rt.models.ltx25._resident_graph import (  # noqa: E402
    CachingPromptEncoder, ResidentSwapBuilder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="768x512x49f")
    ap.add_argument("--prompt", default="A golden retriever running through "
                                        "a sunny meadow")
    ap.add_argument("--tag", default="exact")
    ap.add_argument("--no-cache", dest="cache", action="store_false",
                    default=True)
    ap.add_argument("--no-resident", dest="resident", action="store_false",
                    default=True)
    args = ap.parse_args()
    h, w, f = (int(x) for x in args.res.replace("f", "").split("x"))
    video_s = f / 24.0

    print(f"[exact] arm: sdpa eager, prompt_cache={args.cache} "
          f"resident={args.resident}", flush=True)
    pipe = flash_rt.load_model(
        checkpoint="/workspace/data/models/LTX-2.5",
        config="ltx25",
        attention="sdpa",
        fuse=False,
        compile_mode=None,
    )
    front = pipe.pipeline
    front.set_prompt(args.prompt)  # triggers _load_pipe()
    inner = getattr(front, "_pipe")

    if args.resident:
        from flash_rt.models.ltx25._nvfp4_ffn_swap import (
            SwapInstallingBuilder)
        builder = inner.stage._transformer_builder
        inner.stage = inner.stage.with_builder(ResidentSwapBuilder(
            builder, []))
        print("[exact] transformer builder wrapped resident (no swaps)",
              flush=True)
    if args.cache:
        enc = inner.prompt_encoder
        inner.prompt_encoder = CachingPromptEncoder(enc)
        print("[exact] prompt encoder wrapped with cache", flush=True)

    for i in range(2):
        out = f"/tmp/ltx25_{args.res}_{args.tag}_{i}.mp4"
        stats = pipe.infer(prompt=args.prompt, seed=42, height=h, width=w,
                           num_frames=f, frame_rate=24,
                           output_path=out)
        rtf = stats["total_s"] / video_s
        print(f"[exact] run {i}: denoise {stats['denoise_and_prep_s']}s | "
              f"decode {stats['vae_decode_encode_s']}s | total "
              f"{stats['total_s']}s | RTF {rtf:.1f}x | peak "
              f"{stats['peak_mem_gb']}GB -> {out}", flush=True)
    pipe.close()


if __name__ == "__main__":
    main()
