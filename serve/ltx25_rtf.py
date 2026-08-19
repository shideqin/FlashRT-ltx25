"""LTX-2.5 E2E RTF harness: warmup + timed runs in one process, per arm.

Runs the official DistilledPipeline twice per process (warmup, then timed
steady-state) so one-time costs (weight reads, torch.compile, CUDA graph
capture) are excluded from the RTF number. Reports per-run stats and RTF
against the video duration (num_frames / frame_rate).

Usage::

    python serve/ltx25_rtf.py --res 768x512x49f --attention sdpa --no-fuse
    python serve/ltx25_rtf.py --res 768x512x49f --attention sage2-fvk --fuse
    python serve/ltx25_rtf.py --res 768x512x49f --attention sage2-fvk --fuse --compile capture
"""

import argparse
import os
import sys

os.environ.setdefault("FLASH_RT_LTX2_ROOT", "/workspace/data/LTX-2")
os.environ.setdefault("HF_HOME", "/workspace/data/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, "/workspace/data/FlashRT")

import flash_rt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="768x512x49f")
    ap.add_argument("--attention", default="sdpa")
    ap.add_argument("--fuse", action="store_true", default=False)
    ap.add_argument("--compile", default=None)
    ap.add_argument("--prompt", default="A golden retriever running through "
                                        "a sunny meadow")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    h, w, f = (int(x) for x in args.res.replace("f", "").split("x"))
    video_s = f / 24.0

    print(f"[rtf] arm: attention={args.attention} fuse={args.fuse} "
          f"compile={args.compile} tag={args.tag}", flush=True)
    pipe = flash_rt.load_model(
        checkpoint="/workspace/data/models/LTX-2.5",
        config="ltx25",
        attention=args.attention,
        fuse=args.fuse,
        compile_mode=args.compile,
    )
    for i in range(2):
        out = f"/tmp/ltx25_{args.res}_{args.tag}_{i}.mp4"
        stats = pipe.infer(prompt=args.prompt, seed=42, height=h, width=w,
                           num_frames=f, frame_rate=24,
                           output_path=out)
        rtf = stats["total_s"] / video_s
        print(f"[rtf] run {i}: denoise {stats['denoise_and_prep_s']}s | "
              f"decode {stats['vae_decode_encode_s']}s | total "
              f"{stats['total_s']}s | RTF {rtf:.1f}x | peak "
              f"{stats['peak_mem_gb']}GB -> {out}", flush=True)
    pipe.close()


if __name__ == "__main__":
    main()
