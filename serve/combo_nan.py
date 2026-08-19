"""Locate first NaN in (step, block) for a black combo run."""
import os
import sys

os.environ.setdefault("FLASH_RT_LTX2_ROOT", "/workspace/data/LTX-2")
os.environ.setdefault("HF_HOME", "/workspace/data/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/workspace/data/FlashRT")
sys.path.insert(0, "/workspace/data/LTX-2/packages/ltx-core/src")
sys.path.insert(0, "/workspace/data/LTX-2/packages/ltx-pipelines/src")
sys.path.insert(0, "/workspace/data/LTX-2/packages/ltx-kernels/src")

import torch  # noqa: E402


def main():
    import flash_rt  # noqa: PLC0415
    from ltx_core.model.transformer.transformer import (
        BasicAVTransformerBlock, FeedForward)

    nan_events = []

    orig_block = BasicAVTransformerBlock.forward

    def traced_block(self, *a, **k):
        out = orig_block(self, *a, **k)
        v = out[0]
        x = v.x if hasattr(v, "x") else None
        if x is not None and not torch.isfinite(x).all():
            nan_events.append(("block", len(nan_events),
                               float(x.float().abs().max())))
        return out

    BasicAVTransformerBlock.forward = traced_block

    orig_ff = FeedForward.forward

    def traced_ff(self, *a, **k):
        o = orig_ff(self, *a, **k)
        if not torch.isfinite(o).all():
            nan_events.append(("ff", len(nan_events),
                               float(o.float().abs().max())))
        return o

    FeedForward.forward = traced_ff

    pipe = flash_rt.load_model(
        checkpoint="/workspace/data/models/LTX-2.5", config="ltx25",
        attention="sage2-fvk", fuse=True, compile_mode=None)
    stats = pipe.infer(
        prompt="A golden retriever running through a sunny meadow",
        seed=42, height=512, width=768, num_frames=49, frame_rate=24,
        output_path="/tmp/combo_nan.mp4")
    torch.cuda.synchronize()
    import os as _os
    print(f"size={_os.path.getsize('/tmp/combo_nan.mp4')}", flush=True)
    print(f"nan events: {len(nan_events)}", flush=True)
    if nan_events:
        print(f"first: {nan_events[0]}", flush=True)
        # events around the first: show the 6 before and after
        for e in nan_events[:8]:
            print(f"  {e}", flush=True)
    pipe.close()


if __name__ == "__main__":
    main()
