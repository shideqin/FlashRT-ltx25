"""Full LTX-2.5 production flow, segment-by-segment: baseline vs exact.

Segments (one complete generation): prompt encode -> stage-1 denoise (8 steps)
-> stage-2 denoise (3 steps, after latent upsample) -> VAE decode -> mp4
encode (audio muxed). Baseline and exact (resident+cache) arms run in one
process, same seed 42, same prompt; each arm runs twice (run1 = steady state).

Usage: python serve/ltx25_flow.py --res 768x512x49f
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

PROMPT = "A golden retriever running through a sunny meadow"


class FlowTimer:
    def __init__(self) -> None:
        self.times = {}

    def note(self, key, seconds):
        self.times[key] = self.times.get(key, 0.0) + seconds

    class Proxy:
        def __init__(self, timer, inner, key):
            self._timer = timer
            self._inner = inner
            self._key = key

        def __call__(self, *a, **k):
            t0 = time.perf_counter()
            try:
                return self._inner(*a, **k)
            finally:
                self._timer.note(self._key, time.perf_counter() - t0)

        def __getattr__(self, item):
            return getattr(self._inner, item)

    def wrap(self, obj, key):
        return self.Proxy(self, obj, key)

    def wrap_stage(self, obj, full_width):
        return self.StageProxy(self, obj, full_width)

    def reset(self):
        self.times = {}

    class StageProxy(Proxy):
        def __call__(self, *a, **k):
            kw = k.get("width")
            self._key = ("stage1_denoise" if kw == self._full_width // 2
                         else "stage2_denoise")
            return super().__call__(*a, **k)

        def __init__(self, timer, inner, full_width):
            self._full_width = full_width
            super().__init__(timer, inner, None)

    def report(self, label):
        order = ["prompt_encode", "stage1_denoise", "stage2_denoise",
                 "vae_decode", "mp4_encode"]
        parts = " | ".join(
            f"{k} {self.times.get(k, 0.0):.2f}s" for k in order
            if self.times.get(k, 0.0) > 0)
        total = sum(v for k, v in self.times.items()
                    if k in order and v > 0)
        print(f"[flow] {label}: {parts} | total {total:.2f}s", flush=True)
        return total


def run_arm(pipe, front, inner, timer, tag, res, h, w, f):
    out = f"/tmp/ltx25_flow_{res}_{tag}.mp4"
    stats = pipe.infer(prompt=PROMPT, seed=42, height=h, width=w,
                       num_frames=f, frame_rate=24, output_path=out)
    video_s = f / 24.0
    print(f"[flow] {tag}: denoise_and_prep {stats['denoise_and_prep_s']}s | "
          f"decode+encode {stats['vae_decode_encode_s']}s | total "
          f"{stats['total_s']}s | RTF {stats['total_s']/video_s:.1f}x",
          flush=True)
    return out


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

    # --- baseline arm: instrument, run twice, keep steady-state ---
    t_base = FlowTimer()
    inner.prompt_encoder = t_base.wrap(inner.prompt_encoder, "prompt_encode")
    inner.stage = t_base.wrap_stage(inner.stage, w)
    inner.video_decoder = t_base.wrap(inner.video_decoder, "vae_decode")
    import ltx_pipelines.utils.media_io as mio
    orig_encode = mio.encode_video

    def enc(video, **kw):
        t0 = time.perf_counter()
        try:
            return orig_encode(video, **kw)
        finally:
            t_base.note("mp4_encode", time.perf_counter() - t0)
    mio.encode_video = enc

    run_arm(pipe, front, inner, t_base, "base", args.res, h, w, f)  # warmup
    t_base.reset()
    base_out = run_arm(pipe, front, inner, t_base, "base", args.res, h, w, f)
    t_base.report("baseline steady-state (run 1)")

    # --- exact arm: resident + cache, instrument, steady-state ---
    inner.stage = inner.stage.with_builder(ResidentSwapBuilder(
        inner.stage._transformer_builder, []))
    inner.prompt_encoder = CachingPromptEncoder(inner.prompt_encoder)
    t_exact = FlowTimer()
    inner.prompt_encoder = t_exact.wrap(inner.prompt_encoder, "prompt_encode")
    inner.stage = t_exact.wrap_stage(inner.stage, w)
    inner.video_decoder = t_exact.wrap(inner.video_decoder, "vae_decode")

    def enc2(video, **kw):
        t0 = time.perf_counter()
        try:
            return orig_encode(video, **kw)
        finally:
            t_exact.note("mp4_encode", time.perf_counter() - t0)
    mio.encode_video = enc2

    run_arm(pipe, front, inner, t_exact, "exact", args.res, h, w, f)  # warmup
    t_exact.reset()
    exact_out = run_arm(pipe, front, inner, t_exact, "exact", args.res, h, w, f)
    t_exact.report("exact steady-state (run 1)")

    # --- final frame parity (decoded mp4) ---
    from serve_parity import compare_mp4
    cos_v, md_v = compare_mp4(base_out, exact_out)
    print(f"[flow] base vs exact mp4: cosine {cos_v:.9f} max|d| {md_v:.6f}",
          flush=True)
    pipe.close()


if __name__ == "__main__":
    main()
