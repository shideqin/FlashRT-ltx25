"""Capture final latent via RecordingDiffusionStage, compare 3 configs."""
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

from ltx_pipelines.utils.blocks import RecordingDiffusionStage  # noqa: E402

PROMPT = "A golden retriever running through a sunny meadow"


def run(attention, fuse, out):
    import flash_rt  # noqa: PLC0415
    pipe = flash_rt.load_model(
        checkpoint="/workspace/data/models/LTX-2.5", config="ltx25",
        attention=attention, fuse=fuse, compile_mode=None)
    front = pipe.pipeline
    front.set_prompt(PROMPT)
    inner = getattr(front, "_pipe")
    rec = RecordingDiffusionStage(inner.stage)
    inner.stage = rec
    stats = pipe.infer(prompt=PROMPT, seed=42, height=512, width=768,
                       num_frames=49, frame_rate=24, output_path=out)
    torch.cuda.synchronize()
    states = rec.video_states
    if states:
        lat = states[-1].latent
        print(f"[{attention}/fuse={fuse}] final latent: "
              f"shape={tuple(lat.shape)} std={lat.float().std():.4f} "
              f"min={lat.min():.4f} max={lat.max():.4f} "
              f"mean={lat.mean():.4f} finite={torch.isfinite(lat).all().item()}",
              flush=True)
    else:
        print(f"[{attention}/fuse={fuse}] no states captured", flush=True)
    print(f"  total_s={stats['total_s']:.1f} out_size={os.path.getsize(out)}",
          flush=True)
    pipe.close()


if __name__ == "__main__":
    run("sdpa", False, "/tmp/lat_base.mp4")
    run("sdpa", True, "/tmp/lat_ffn.mp4")
    run("sage2-fvk", True, "/tmp/lat_combo.mp4")
