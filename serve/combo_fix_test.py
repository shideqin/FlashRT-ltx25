"""Repeat combo (sage2+FFN) 4 times; report output sizes."""
import os
import sys

os.environ.setdefault("FLASH_RT_LTX2_ROOT", "/workspace/data/LTX-2")
os.environ.setdefault("HF_HOME", "/workspace/data/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/workspace/data/FlashRT")

import torch  # noqa: E402
import flash_rt  # noqa: E402


def main():
    sizes = []
    for trial in range(4):
        pipe = flash_rt.load_model(
            checkpoint="/workspace/data/models/LTX-2.5", config="ltx25",
            attention="sage2-fvk", fuse=True, compile_mode=None)
        out = f"/tmp/combo_fix_{trial}.mp4"
        stats = pipe.infer(
            prompt="A golden retriever running through a sunny meadow",
            seed=42, height=512, width=768, num_frames=49, frame_rate=24,
            output_path=out)
        sz = os.path.getsize(out)
        sizes.append(sz)
        print(f"trial {trial}: total={stats['total_s']:.1f}s size={sz}",
              flush=True)
        pipe.close()
        torch.cuda.empty_cache()
    print(f"sizes: {sizes} black: {sum(1 for s in sizes if s < 10000)}",
          flush=True)


if __name__ == "__main__":
    main()
