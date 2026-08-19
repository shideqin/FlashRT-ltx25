"""GPU-busy vs wall time for the bit-exact arm's denoise (steady-state).

Answers: is the remaining time GPU compute (floor reached) or launch/Python
overhead (bit-exact headroom left)?

Usage: python serve/ltx25_gpu_share.py --res 768x512x49f
"""

import argparse
import os
import sys
import time

os.environ.setdefault("FLASH_RT_LTX2_ROOT", "/workspace/data/LTX-2")
os.environ.setdefault("HF_HOME", "/workspace/data/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, "/workspace/data/FlashRT")

import torch  # noqa: E402

import flash_rt  # noqa: E402
from flash_rt.models.ltx25._resident_graph import (  # noqa: E402
    CachingPromptEncoder, ResidentSwapBuilder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="768x512x49f")
    ap.add_argument("--prompt", default="A golden retriever running through "
                                        "a sunny meadow")
    args = ap.parse_args()
    h, w, f = (int(x) for x in args.res.replace("f", "").split("x"))

    pipe = flash_rt.load_model(
        checkpoint="/workspace/data/models/LTX-2.5",
        config="ltx25", attention="sdpa", fuse=False, compile_mode=None)
    front = pipe.pipeline
    front.set_prompt(args.prompt)
    inner = getattr(front, "_pipe")
    inner.stage = inner.stage.with_builder(ResidentSwapBuilder(
        inner.stage._transformer_builder, []))
    inner.prompt_encoder = CachingPromptEncoder(inner.prompt_encoder)

    # warmup + steady state
    pipe.infer(prompt=args.prompt, seed=42, height=h, width=w,
               num_frames=f, frame_rate=24, output_path="/tmp/gpu_share_w.mp4")
    torch.cuda.synchronize()

    s_wall = time.perf_counter()
    s_ev, e_ev = torch.cuda.Event(True), torch.cuda.Event(True)
    s_ev.record()
    stats = pipe.infer(prompt=args.prompt, seed=42, height=h, width=w,
                       num_frames=f, frame_rate=24,
                       output_path="/tmp/gpu_share_t.mp4")
    e_ev.record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - s_wall
    gpu = s_ev.elapsed_time(e_ev) / 1000
    print(f"[gpu_share] wall {wall:.3f}s | gpu-busy {gpu:.3f}s | "
          f"non-gpu {wall-gpu:.3f}s ({(wall-gpu)/wall*100:.2f}%)")
    print(f"[gpu_share] denoise_and_prep {stats['denoise_and_prep_s']}s | "
          f"decode {stats['vae_decode_encode_s']}s")
    pipe.close()


if __name__ == "__main__":
    main()
