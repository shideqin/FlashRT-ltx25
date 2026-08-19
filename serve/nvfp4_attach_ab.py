"""NVFP4 attach vs host: clean A/B on the same process, 768.

host timing BEFORE attach, attached timing AFTER attach, same inputs.
Reports S=2688 (stage2), S=576 (stage1), and 11-step mixed RTF.
"""

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
    S1 = dict(frames=7, height=8, width=12, text=16, audio_frames=128)
    S2 = dict(frames=7, height=16, width=24, text=16, audio_frames=128)

    # host BEFORE attach
    h_s1 = timeit(lambda: m(**mk(S1, 3)))
    h_s2 = timeit(lambda: m(**mk(S2, 3)))
    print(f"HOST: S=576 {h_s1:.1f}ms | S=2688 {h_s2:.1f}ms", flush=True)

    from flash_rt import structures
    print("attaching ...", flush=True)
    plan = structures.attach(m, lambda: m(**mk(S2, 0)), verbose=True,
                             scheme="nvfp4_balance_sage", rounds=2, iters=5)

    a_s1 = timeit(lambda: m(**mk(S1, 3)))
    a_s2 = timeit(lambda: m(**mk(S2, 3)))
    print(f"ATTACHED: S=576 {a_s1:.1f}ms ({h_s1/a_s1:.2f}x) | "
          f"S=2688 {a_s2:.1f}ms ({h_s2/a_s2:.2f}x)", flush=True)

    base11 = (h_s1 * 8 + h_s2 * 3) / 1000
    att11 = (a_s1 * 8 + a_s2 * 3) / 1000
    video_s = 49 / 24
    print(f"11-STEP: base {base11:.2f}s RTF {base11/video_s:.1f}x -> "
          f"att {att11:.2f}s RTF {att11/video_s:.1f}x = "
          f"{base11/att11:.2f}x", flush=True)
    print(f"PEAK: host {torch.cuda.max_memory_allocated()/2**30:.1f}GB",
          flush=True)


if __name__ == "__main__":
    main()
