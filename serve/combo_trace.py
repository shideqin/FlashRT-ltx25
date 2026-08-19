"""Debug combo explosion: patch block forward globally, trace collapse."""

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
    from ltx_core.model.transformer.transformer import BasicAVTransformerBlock

    captured = []

    orig_forward = BasicAVTransformerBlock.forward

    def traced(self, *a, **k):
        out = orig_forward(self, *a, **k)
        v = k.get("video")
        x = v.x if v is not None else None
        if x is not None:
            captured.append((
                len(captured), float(x.float().std()),
                bool(torch.isfinite(x).all()),
                float(x.float().abs().max())))
        return out

    BasicAVTransformerBlock.forward = traced

    pipe = flash_rt.load_model(
        checkpoint="/workspace/data/models/LTX-2.5", config="ltx25",
        attention="sage2-fvk", fuse=True, compile_mode=None)
    stats = pipe.infer(prompt="A golden retriever running through a sunny meadow",
                       seed=42, height=512, width=768, num_frames=49,
                       frame_rate=24, output_path="/tmp/combo_trace.mp4")
    torch.cuda.synchronize()

    print(f"captured {len(captured)} block activations", flush=True)
    if not captured:
        print("NO activations captured — patch not hit", flush=True)
        pipe.close()
        return
    # first 3 and worst
    for c in captured[:3]:
        print(f"  #{c[0]}: std={c[1]:.6f} finite={c[2]} max={c[3]:.4f}",
              flush=True)
    worst = min(captured, key=lambda c: c[1])
    print(f"WORST #{worst[0]}: std={worst[1]:.6f} finite={worst[2]} "
          f"max={worst[3]:.4f}", flush=True)
    collapsed = [c for c in captured if c[1] < 0.01]
    print(f"collapsed (std<0.01): {len(collapsed)}/{len(captured)}", flush=True)
    if collapsed:
        print(f"first collapse at #{collapsed[0][0]}", flush=True)
        # std trend around collapse
        ci = collapsed[0][0]
        lo = max(0, ci - 3)
        for c in captured[lo:ci + 2]:
            print(f"  #{c[0]}: std={c[1]:.6f} max={c[3]:.4f}", flush=True)
    pipe.close()


if __name__ == "__main__":
    main()
