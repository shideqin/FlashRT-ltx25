"""Check if FFN input (norm output) is already NaN before the swap chain."""
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
    from ltx_core.model.transformer.transformer import FeedForward

    # hook the swapped FFN: input x before chain, output after
    in_stats = []
    out_stats = []

    orig_ff = FeedForward.forward

    def traced_ff(self, *a, **k):
        x = a[0] if a else k.get("x")
        if x is not None:
            in_stats.append((float(x.float().std()),
                             bool(torch.isfinite(x).all()),
                             float(x.float().abs().max())))
        o = orig_ff(self, *a, **k)
        out_stats.append((float(o.float().std()),
                          bool(torch.isfinite(o).all()),
                          float(o.float().abs().max())))
        return o

    FeedForward.forward = traced_ff

    pipe = flash_rt.load_model(
        checkpoint="/workspace/data/models/LTX-2.5", config="ltx25",
        attention="sage2-fvk", fuse=True, compile_mode=None)
    stats = pipe.infer(
        prompt="A golden retriever running through a sunny meadow",
        seed=42, height=512, width=768, num_frames=49, frame_rate=24,
        output_path="/tmp/combo_in.mp4")
    torch.cuda.synchronize()
    import os as _os
    print(f"size={_os.path.getsize('/tmp/combo_in.mp4')}", flush=True)
    print(f"ff calls: {len(in_stats)}", flush=True)
    # first 4 inputs and outputs
    for i in range(min(6, len(in_stats))):
        s = in_stats[i]
        o = out_stats[i]
        print(f"  ff#{i} in: std={s[0]:.4f} finite={s[1]} max={s[2]:.2f} | "
              f"out: std={o[0]:.4f} finite={o[1]} max={o[2]:.2f}", flush=True)
    pipe.close()


if __name__ == "__main__":
    main()
