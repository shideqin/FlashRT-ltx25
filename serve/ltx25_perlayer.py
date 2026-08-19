"""Per-layer bit-exactness proof: baseline vs exact arm, block by block.

Hooks every transformer block's video/audio input activation during one
baseline infer and one exact (resident+cache) infer in the same process,
same seed, same prompt. Reports per-block cosine and max|d| — must be
1.000000 / 0.0 everywhere if the exact arm changes nothing numerically.

The exact arm reuses the same prompt embedding tensor and the same weights;
the denoise loop is the host's own code. This verifies it at every block.

Usage: python serve/ltx25_perlayer.py --res 768x512x49f
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


class BlockTracer:
    def __init__(self, model) -> None:
        vm = getattr(model, "velocity_model", model)
        self.blocks = vm.transformer_blocks
        self.video_ins = []
        self.handles = []
        for i, blk in enumerate(self.blocks):
            h = blk.register_forward_hook(self._make(i), with_kwargs=True)
            self.handles.append(h)

    def _make(self, i):
        def hook(module, args, kwargs, out):
            video = kwargs.get("video")
            if video is None:
                video = args[0] if args else None
            x = video.x if video is not None else None
            if x is None:
                return
            self.video_ins.append((i, x.detach().clone()))
        return hook

    def clear(self):
        self.video_ins = []

    def remove(self):
        for h in self.handles:
            h.remove()


def load_frontend():
    pipe = flash_rt.load_model(
        checkpoint="/workspace/data/models/LTX-2.5",
        config="ltx25", attention="sdpa", fuse=False, compile_mode=None)
    front = pipe.pipeline
    front.set_prompt(PROMPT)
    return pipe, getattr(front, "_pipe")


PROMPT = "A golden retriever running through a sunny meadow"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="768x512x49f")
    args = ap.parse_args()
    h, w, f = (int(x) for x in args.res.replace("f", "").split("x"))

    pipe, inner = load_frontend()

    # ---- baseline pass (no wrappers) ----
    model = inner.stage._build_transformer()
    tracer = BlockTracer(model)
    print(f"[perlayer] baseline pass over {len(tracer.blocks)} blocks",
          flush=True)
    pipe.infer(prompt=PROMPT, seed=42, height=h, width=w, num_frames=f,
               frame_rate=24, output_path="/tmp/perlayer_base.mp4")
    torch.cuda.synchronize()
    base_ins = tracer.video_ins
    tracer.remove()
    print(f"[perlayer] baseline captured {len(base_ins)} block activations",
          flush=True)
    del model
    torch.cuda.empty_cache()

    # ---- exact pass (resident + cache) ----
    inner.stage = inner.stage.with_builder(ResidentSwapBuilder(
        inner.stage._transformer_builder, []))
    inner.prompt_encoder = CachingPromptEncoder(inner.prompt_encoder)
    exact_model = inner.stage._build_transformer()
    tracer2 = BlockTracer(exact_model)
    print("[perlayer] exact pass (resident+cache)", flush=True)
    pipe.infer(prompt=PROMPT, seed=42, height=h, width=w, num_frames=f,
               frame_rate=24, output_path="/tmp/perlayer_exact.mp4")
    torch.cuda.synchronize()
    exact_ins = tracer2.video_ins
    tracer2.remove()

    assert len(base_ins) == len(exact_ins), (
        f"activation count mismatch {len(base_ins)} vs {len(exact_ins)}")
    worst = (1.0, 0.0, None)
    for (bi, b), (ei, e) in zip(base_ins, exact_ins):
        assert bi == ei
        bf, ef = b.double().reshape(-1), e.double().reshape(-1)
        cos = float(torch.nn.functional.cosine_similarity(
            bf, ef, dim=0))
        md = float((b.float() - e.float()).abs().max())
        if cos < worst[0] or md > worst[1]:
            worst = (cos, md, bi)
    print(f"[perlayer] blocks compared: {len(base_ins)} "
          f"(48 blocks x 11 steps = 528 expected)", flush=True)
    print(f"[perlayer] worst block {worst[2]}: cosine {worst[0]:.9f} "
          f"max|d| {worst[1]:.6f}", flush=True)

    from serve_parity import compare_mp4
    cos_v, md_v = compare_mp4("/tmp/perlayer_base.mp4",
                              "/tmp/perlayer_exact.mp4")
    print(f"[perlayer] final mp4: cosine {cos_v:.9f} max|d| {md_v:.6f}",
          flush=True)
    pipe.close()


if __name__ == "__main__":
    main()
