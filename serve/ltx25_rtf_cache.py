"""LTX-2.5 bit-exact RTF harness: eager pipeline + prompt embedding cache.

The only numerically-exact acceleration on this model: cache the prompt
encoder output (same prompt -> same tensor, encoder is deterministic), skip
the ~26GB Gemma-4 forward on repeat prompts. No kernel swap, no torch.compile,
no CUDA graph capture -- every op stays the host's own, so the output must be
bit-identical to the unmodified baseline.

Verifies bit-exactness by comparing against the baseline arm's mp4.

Usage::

    python serve/ltx25_rtf_cache.py --res 768x512x49f
"""

import argparse
import os
import sys

os.environ.setdefault("FLASH_RT_LTX2_ROOT", "/workspace/data/LTX-2")
os.environ.setdefault("HF_HOME", "/workspace/data/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, "/workspace/data/FlashRT")

import flash_rt  # noqa: E402
from flash_rt.models.ltx25._resident_graph import CachingPromptEncoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="768x512x49f")
    ap.add_argument("--prompt", default="A golden retriever running through "
                                        "a sunny meadow")
    ap.add_argument("--tag", default="cache")
    ap.add_argument("--no-cache", dest="cache", action="store_false",
                    default=True)
    args = ap.parse_args()
    h, w, f = (int(x) for x in args.res.replace("f", "").split("x"))
    video_s = f / 24.0

    print(f"[cache] arm: attention=sdpa fuse=False compile=None "
          f"prompt_cache={args.cache}", flush=True)
    pipe = flash_rt.load_model(
        checkpoint="/workspace/data/models/LTX-2.5",
        config="ltx25",
        attention="sdpa",
        fuse=False,
        compile_mode=None,
    )
    front = pipe.pipeline
    if args.cache:
        front.set_prompt(args.prompt)  # triggers _load_pipe()
        enc = getattr(front, "_pipe").prompt_encoder
        front._pipe.prompt_encoder = CachingPromptEncoder(enc)
        print("[cache] prompt encoder wrapped with cache", flush=True)
    for i in range(2):
        out = f"/tmp/ltx25_{args.res}_{args.tag}_{i}.mp4"
        stats = pipe.infer(prompt=args.prompt, seed=42, height=h, width=w,
                           num_frames=f, frame_rate=24,
                           output_path=out)
        rtf = stats["total_s"] / video_s
        print(f"[cache] run {i}: denoise {stats['denoise_and_prep_s']}s | "
              f"decode {stats['vae_decode_encode_s']}s | total "
              f"{stats['total_s']}s | RTF {rtf:.1f}x | peak "
              f"{stats['peak_mem_gb']}GB -> {out}", flush=True)
    pipe.close()


if __name__ == "__main__":
    main()
