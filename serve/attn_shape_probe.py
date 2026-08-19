"""Record the (lq, lk, heads) keys the sage2 attention actually sees in EAGER
combo mode at 768x512x49f."""
import os
import sys

os.environ.setdefault("FLASH_RT_LTX2_ROOT", "/workspace/data/LTX-2")
os.environ.setdefault("HF_HOME", "/workspace/data/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/workspace/data/FlashRT")

import torch  # noqa: E402


def main():
    import flash_rt  # noqa: PLC0415
    from flash_rt.models.ltx25 import _attn_swap

    seen = {}
    orig = _attn_swap.FvkSage2Attention.__call__

    def traced(self, q, k, v, heads):
        key = (q.shape[1], k.shape[1], heads)
        seen[key] = seen.get(key, 0) + 1
        return orig(self, q, k, v, heads)

    _attn_swap.FvkSage2Attention.__call__ = traced

    pipe = flash_rt.load_model(
        checkpoint="/workspace/data/models/LTX-2.5", config="ltx25",
        attention="sage2-fvk", fuse=True, compile_mode=None)
    stats = pipe.infer(
        prompt="A golden retriever running through a sunny meadow",
        seed=42, height=512, width=768, num_frames=49, frame_rate=24,
        output_path="/tmp/attn_shape_probe.mp4")
    torch.cuda.synchronize()
    import os as _os
    print(f"size={_os.path.getsize('/tmp/attn_shape_probe.mp4')}", flush=True)
    for key, n in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"  (lq={key[0]}, lk={key[1]}, heads={key[2]}): {n} calls",
              flush=True)
    pipe.close()


if __name__ == "__main__":
    main()
