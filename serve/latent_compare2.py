"""Capture final latent via patching DistilledPipeline.__call__ output."""
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

from ltx_pipelines.distilled import DistilledPipeline  # noqa: E402

captured = {}

orig_call = DistilledPipeline.__call__


def traced(self, *a, **k):
    out = orig_call(self, *a, **k)
    # out = (video_chunks_iter, audio, num_frames, tiling)
    captured["latent"] = None  # latent not directly in output; video chunks are
    return out


DistilledPipeline.__call__ = traced


def run(attention, fuse, out):
    import flash_rt  # noqa: PLC0415
    pipe = flash_rt.load_model(
        checkpoint="/workspace/data/models/LTX-2.5", config="ltx25",
        attention=attention, fuse=fuse, compile_mode=None)
    stats = pipe.infer(
        prompt="A golden retriever running through a sunny meadow",
        seed=42, height=512, width=768, num_frames=49, frame_rate=24,
        output_path=out)
    torch.cuda.synchronize()
    print(f"[{attention}/fuse={fuse}] total_s={stats['total_s']:.1f} "
          f"out_size={os.path.getsize(out)}", flush=True)
    pipe.close()


if __name__ == "__main__":
    run("sdpa", False, "/tmp/lat_base.mp4")
    run("sdpa", True, "/tmp/lat_ffn.mp4")
    run("sage2-fvk", True, "/tmp/lat_combo.mp4")
