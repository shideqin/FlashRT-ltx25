"""Full trace: per-step noise_pred stats + all 946 FFN finite states."""
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

    ff_bad = []

    orig_ff = FeedForward.forward

    def traced_ff(self, *a, **k):
        o = orig_ff(self, *a, **k)
        if not torch.isfinite(o).all():
            ff_bad.append(len(ff_bad))
        return o

    FeedForward.forward = traced_ff

    # patch transformer call to capture per-step noise_pred
    step_stats = []
    from ltx_core.model.transformer.model import LTXModel
    orig_tc = LTXModel.forward

    def traced_tc(self, *a, **k):
        out = orig_tc(self, *a, **k)
        try:
            nv = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(nv, (tuple, list)):
                nv = nv[0]
            step_stats.append((float(nv.float().std()),
                               bool(torch.isfinite(nv).all())))
        except Exception:
            step_stats.append((-1.0, True))
        return out

    LTXModel.forward = traced_tc

    pipe = flash_rt.load_model(
        checkpoint="/workspace/data/models/LTX-2.5", config="ltx25",
        attention="sage2-fvk", fuse=True, compile_mode=None)
    stats = pipe.infer(
        prompt="A golden retriever running through a sunny meadow",
        seed=42, height=512, width=768, num_frames=49, frame_rate=24,
        output_path="/tmp/combo_full.mp4")
    torch.cuda.synchronize()
    import os as _os
    print(f"size={_os.path.getsize('/tmp/combo_full.mp4')}", flush=True)
    print(f"steps={len(step_stats)} ff_bad={len(ff_bad)}", flush=True)
    for i, (s, f) in enumerate(step_stats):
        print(f"  step{i}: std={s:.4f} finite={f}", flush=True)
    if ff_bad:
        print(f"first_bad_ff={ff_bad[0]} total_bad={len(ff_bad)}", flush=True)
    pipe.close()


if __name__ == "__main__":
    main()
