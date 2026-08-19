"""Compare FF outputs: FFN-only vs combo, find whether FFN chain is the culprit."""
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


def run(attention, out):
    import flash_rt  # noqa: PLC0415
    from ltx_core.model.transformer.transformer import FeedForward

    ff_captured = []
    orig_ff = FeedForward.forward

    def traced_ff(self, *a, **k):
        o = orig_ff(self, *a, **k)
        ff_captured.append((float(o.float().std()),
                            bool(torch.isfinite(o).all()),
                            float(o.float().abs().max())))
        return o

    FeedForward.forward = traced_ff
    pipe = flash_rt.load_model(
        checkpoint="/workspace/data/models/LTX-2.5", config="ltx25",
        attention=attention, fuse=True, compile_mode=None)
    pipe.infer(prompt="A golden retriever running through a sunny meadow",
               seed=42, height=512, width=768, num_frames=49, frame_rate=24,
               output_path=out)
    torch.cuda.synchronize()
    import os as _os
    print(f"[{attention}] size={_os.path.getsize(out)} ff_calls={len(ff_captured)}",
          flush=True)
    # find first bad (nan or std>200)
    first_bad = next((i for i, c in enumerate(ff_captured)
                      if (not c[1]) or c[0] > 200), None)
    print(f"  first_bad={first_bad} ({'none' if first_bad is None else ''})",
          flush=True)
    if first_bad is not None:
        lo = max(0, first_bad - 3)
        for i in range(lo, min(first_bad + 2, len(ff_captured))):
            c = ff_captured[i]
            print(f"  ff#{i}: std={c[0]:.2f} finite={c[1]} max={c[2]:.2f}",
                  flush=True)
    pipe.close()


if __name__ == "__main__":
    run("sdpa", "/tmp/ffn_only.mp4")
    run("sage2-fvk", "/tmp/combo_ff.mp4")
