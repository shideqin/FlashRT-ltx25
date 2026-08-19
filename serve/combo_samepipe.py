"""Same-pipe two infers (previously both black) + one 768 RTF-style timed run."""
import os
import sys

os.environ.setdefault("FLASH_RT_LTX2_ROOT", "/workspace/data/LTX-2")
os.environ.setdefault("HF_HOME", "/workspace/data/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/workspace/data/FlashRT")

import torch  # noqa: E402
import flash_rt  # noqa: E402

PROMPT = "A golden retriever running through a sunny meadow"


def main():
    pipe = flash_rt.load_model(
        checkpoint="/workspace/data/models/LTX-2.5", config="ltx25",
        attention="sage2-fvk", fuse=True, compile_mode=None)
    stats1 = pipe.infer(prompt=PROMPT, seed=42, height=512, width=768,
                        num_frames=49, frame_rate=24, output_path="/tmp/samepipe_0.mp4")
    stats2 = pipe.infer(prompt=PROMPT, seed=42, height=512, width=768,
                        num_frames=49, frame_rate=24, output_path="/tmp/samepipe_1.mp4")
    s0 = os.path.getsize("/tmp/samepipe_0.mp4")
    s1 = os.path.getsize("/tmp/samepipe_1.mp4")
    print(f"same-pipe run0: {stats1['total_s']:.1f}s size={s0}", flush=True)
    print(f"same-pipe run1: {stats2['total_s']:.1f}s size={s1}", flush=True)
    pipe.close()


if __name__ == "__main__":
    main()
