"""Debug combo: hook block OUTPUT (not input) — find the polluted block."""
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

    captured = []
    ff_captured = []

    orig_block = BasicAVTransformerBlock.forward

    def traced_block(self, *a, **k):
        out = orig_block(self, *a, **k)
        # out is a tuple (video, audio) of latents
        v = out[0]
        x = v.x if hasattr(v, "x") else None
        if x is not None:
            captured.append((
                len(captured), float(x.float().std()),
                bool(torch.isfinite(x).all()),
                float(x.float().abs().max()),
                float(x.float().mean())))
        return out

    BasicAVTransformerBlock.forward = traced_block

    orig_ff = FeedForward.forward

    def traced_ff(self, *a, **k):
        out = orig_ff(self, *a, **k)
        ff_captured.append((
            len(ff_captured),
            float(out.float().std()),
            bool(torch.isfinite(out).all()),
            float(out.float().abs().max())))
        return out

    FeedForward.forward = traced_ff

    pipe = flash_rt.load_model(
        checkpoint="/workspace/data/models/LTX-2.5", config="ltx25",
        attention="sage2-fvk", fuse=True, compile_mode=None)
    stats = pipe.infer(
        prompt="A golden retriever running through a sunny meadow",
        seed=42, height=512, width=768, num_frames=49, frame_rate=24,
        output_path="/tmp/combo_out.mp4")
    torch.cuda.synchronize()
    import os as _os
    print(f"out size: {_os.path.getsize('/tmp/combo_out.mp4')}", flush=True)
    print(f"block outputs captured: {len(captured)}", flush=True)
    print(f"ff outputs captured: {len(ff_captured)}", flush=True)

    if captured:
        bad = [c for c in captured if (not c[2]) or c[1] > 100 or c[1] < 1e-3]
        print(f"bad block outputs: {len(bad)}", flush=True)
        if bad:
            print(f"first bad: {bad[0]}", flush=True)
        # print last 6
        for c in captured[-6:]:
            print(f"  block#{c[0]}: std={c[1]:.4f} finite={c[2]} "
                  f"max={c[3]:.2f} mean={c[4]:.4f}", flush=True)
    if ff_captured:
        badff = [c for c in ff_captured if (not c[2]) or c[1] > 100 or c[1] < 1e-4]
        print(f"bad ff outputs: {len(badff)}", flush=True)
        if badff:
            print(f"first bad ff: {badff[0]}", flush=True)
        for c in ff_captured[:4]:
            print(f"  ff#{c[0]}: std={c[1]:.4f} finite={c[2]} max={c[3]:.2f}",
                  flush=True)
        for c in ff_captured[-4:]:
            print(f"  ff#{c[0]}: std={c[1]:.4f} finite={c[2]} max={c[3]:.2f}",
                  flush=True)
    pipe.close()


if __name__ == "__main__":
    main()
