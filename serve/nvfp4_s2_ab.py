"""NVFP4 attach A/B at S=2688 (the calibrated shape): host vs attached."""
import os
import sys

os.environ.setdefault("HF_HOME", "/workspace/data/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/workspace/data/FlashRT")

import torch  # noqa: E402
from diffusers.models.transformers.transformer_ltx2 import (  # noqa: E402
    LTX2VideoTransformer3DModel)

CKPT = "/workspace/data/models/LTX-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
CFG = "/workspace/data/models/LTX-2.5/diffusers_transformer"


def mk(spec, seed):
    torch.manual_seed(seed)
    dt = torch.bfloat16
    f, h, w, text, af = (spec["frames"], spec["height"], spec["width"],
                         spec["text"], spec["audio_frames"])
    S = f * h * w
    return dict(
        hidden_states=torch.randn(1, S, 128, device="cuda", dtype=dt),
        audio_hidden_states=torch.randn(1, af, 128, device="cuda", dtype=dt),
        encoder_hidden_states=torch.randn(1, text, 4096, device="cuda",
                                          dtype=dt),
        audio_encoder_hidden_states=torch.randn(1, text, 2048, device="cuda",
                                                dtype=dt),
        timestep=torch.tensor([500.0], device="cuda", dtype=dt),
        sigma=torch.tensor([500.0], device="cuda", dtype=dt),
        num_frames=f, height=h, width=w, fps=24.0,
        audio_num_frames=af, return_dict=False)


def timeit(fn, n=5):
    with torch.no_grad():
        fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(n):
            fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / n


def main():
    print("loading ...", flush=True)
    m = LTX2VideoTransformer3DModel.from_single_file(
        CKPT, config=CFG, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False).to("cuda").eval()
    S2 = dict(frames=7, height=16, width=24, text=16, audio_frames=128)

    torch.cuda.reset_peak_memory_stats()
    h = timeit(lambda: m(**mk(S2, 3)))
    peak0 = torch.cuda.max_memory_allocated() / 2**30
    print(f"HOST S=2688: {h:.1f}ms peak {peak0:.1f}GB", flush=True)

    from flash_rt import structures
    print("attaching ...", flush=True)
    plan = structures.attach(m, lambda: m(**mk(S2, 0)), verbose=True,
                             scheme="nvfp4_balance_sage", rounds=2, iters=5)
    torch.cuda.reset_peak_memory_stats()
    a = timeit(lambda: m(**mk(S2, 3)))
    peak1 = torch.cuda.max_memory_allocated() / 2**30
    print(f"ATTACHED S=2688: {a:.1f}ms peak {peak1:.1f}GB = {h/a:.2f}x",
          flush=True)
    video_s = 49 / 24
    print(f"11-step: base {h*11/1000:.2f}s RTF {h*11/1000/video_s:.1f}x -> "
          f"att {a*11/1000:.2f}s RTF {a*11/1000/video_s:.1f}x = {h/a:.2f}x",
          flush=True)


if __name__ == "__main__":
    main()
