"""Per-phase GPU memory for baseline vs exact arms.

Reports allocated/reserved peak, and the memory held during denoise vs
decode, plus the resident transformer's footprint for the exact arm.

Usage: python serve/ltx25_mem.py --res 768x512x49f
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
from flash_rt.models.ltx25._resident_graph import (  # noqa: E402
    CachingPromptEncoder, ResidentSwapBuilder)

PROMPT = "A golden retriever running through a sunny meadow"


def mem_gb():
    return (torch.cuda.memory_allocated() / 2**30,
            torch.cuda.memory_reserved() / 2**30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="768x512x49f")
    args = ap.parse_args()
    h, w, f = (int(x) for x in args.res.replace("f", "").split("x"))

    pipe = flash_rt.load_model(
        checkpoint="/workspace/data/models/LTX-2.5",
        config="ltx25", attention="sdpa", fuse=False, compile_mode=None)
    front = pipe.pipeline
    front.set_prompt(PROMPT)
    inner = getattr(front, "_pipe")

    def snap(label):
        a, r = mem_gb()
        print(f"[mem] {label}: allocated {a:.2f}GB reserved {r:.2f}GB",
              flush=True)

    # ---- baseline: instrument peak during phases ----
    inner.video_decoder._orig_call = inner.video_decoder.__call__
    def dec(video, *a, **k):
        snap("baseline vae_decode start")
        return inner.video_decoder._orig_call(video, *a, **k)
    inner.video_decoder.__call__ = dec
    pipe.infer(prompt=PROMPT, seed=42, height=h, width=w, num_frames=f,
               frame_rate=24, output_path="/tmp/mem_base_1.mp4")
    torch.cuda.synchronize()
    a, r = mem_gb()
    print(f"[mem] baseline after infer: alloc {a:.2f}GB reserved {r:.2f}GB "
          f"peak_alloc {torch.cuda.max_memory_allocated()/2**30:.2f}GB",
          flush=True)
    inner.video_decoder.__call__ = inner.video_decoder._orig_call

    # ---- exact arm: resident + cache ----
    inner.stage = inner.stage.with_builder(ResidentSwapBuilder(
        inner.stage._transformer_builder, []))
    inner.prompt_encoder = CachingPromptEncoder(inner.prompt_encoder)
    pipe.infer(prompt=PROMPT, seed=42, height=h, width=w, num_frames=f,
               frame_rate=24, output_path="/tmp/mem_exact_0.mp4")  # warmup
    torch.cuda.synchronize()
    snap("exact steady-state, before infer (resident held)")
    inner.video_decoder._orig_call = inner.video_decoder.__call__
    def dec2(video, *a, **k):
        snap("exact vae_decode start")
        return inner.video_decoder._orig_call(video, *a, **k)
    inner.video_decoder.__call__ = dec2
    pipe.infer(prompt=PROMPT, seed=42, height=h, width=w, num_frames=f,
               frame_rate=24, output_path="/tmp/mem_exact_1.mp4")
    torch.cuda.synchronize()
    a, r = mem_gb()
    print(f"[mem] exact after infer: alloc {a:.2f}GB reserved {r:.2f}GB "
          f"peak_alloc {torch.cuda.max_memory_allocated()/2**30:.2f}GB",
          flush=True)
    pipe.close()


if __name__ == "__main__":
    main()
