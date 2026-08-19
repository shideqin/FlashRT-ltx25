"""PR 172 native runtime: LTX-2.5 distilled pipeline with FlashRT swaps.

Drives the merged ltx25 torch frontend (sage2-fvk attention, W4A4 NVFP4 FFN
chain, optional whole-loop capture) on the official LTX-2 monorepo packages.
The bf16 distilled transformer is used as the host (the ModelScope nvfp4
file is corrupt and set aside); the FFN chain repacks weights to NVFP4 at
adopt, so the W4A4 path is exercised regardless.

Usage::

    FLASH_RT_LTX2_ROOT=/workspace/data/LTX-2 python serve/ltx25_runtime.py \
        --res 768x512x49f [--attention sage2-fvk] [--fuse] [--compile capture|default]
"""

import argparse
import os
import sys

os.environ.setdefault("FLASH_RT_LTX2_ROOT", "/workspace/data/LTX-2")
os.environ.setdefault("HF_HOME", "/workspace/data/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, "/workspace/data/FlashRT")

import torch  # noqa: E402
import flash_rt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="768x512x49f")
    ap.add_argument("--attention", default="sage2-fvk")
    ap.add_argument("--fuse", action="store_true", default=True)
    ap.add_argument("--no-fuse", dest="fuse", action="store_false")
    ap.add_argument("--compile", default=None)
    ap.add_argument("--prompt", default="A golden retriever running through "
                                        "a sunny meadow")
    args = ap.parse_args()
    h, w, f = (int(x) for x in args.res.replace("f", "").split("x"))

    print(f"[ltx25] loading frontend (attention={args.attention}, "
          f"fuse={args.fuse}, compile={args.compile})", flush=True)
    pipe = flash_rt.load_model(
        checkpoint="/workspace/data/models/LTX-2.5",
        config="ltx25",
        attention=args.attention,
        fuse=args.fuse,
        compile_mode=args.compile,
    )
    print(f"[ltx25] load ok", flush=True)
    stats = pipe.infer(prompt=args.prompt, seed=42, height=h, width=w,
                       num_frames=f, frame_rate=24,
                       output_path=f"/tmp/ltx25_{args.res}.mp4")
    for k, v in stats.items():
        print(f"  {k}: {v}", flush=True)
    pipe.close()


if __name__ == "__main__":
    main()
