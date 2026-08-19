"""Debug: sage2 + FFN swap combo black-output on NVFP4 768.

Single transformer forward, check intermediate activation stats.
"""
import os
import sys

os.environ.setdefault("FLASH_RT_LTX2_ROOT", "/workspace/data/LTX-2")
os.environ.setdefault("HF_HOME", "/workspace/data/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/workspace/data/FlashRT")

import torch  # noqa: E402


def mk(seed):
    torch.manual_seed(seed)
    dt = torch.bfloat16
    return dict(
        hidden_states=torch.randn(1, 2688, 128, device="cuda", dtype=dt),
        audio_hidden_states=torch.randn(1, 128, 128, device="cuda", dtype=dt),
        encoder_hidden_states=torch.randn(1, 16, 4096, device="cuda", dtype=dt),
        audio_encoder_hidden_states=torch.randn(1, 16, 2048, device="cuda",
                                                dtype=dt),
        timestep=torch.tensor([500.0], device="cuda", dtype=dt),
        sigma=torch.tensor([500.0], device="cuda", dtype=dt),
        num_frames=7, height=16, width=24, fps=24.0,
        audio_num_frames=128, return_dict=False)


def main():
    import flash_rt  # noqa: PLC0415
    from flash_rt.models.ltx25._attn_swap import make_ltx25_attention
    from flash_rt.models.ltx25._nvfp4_ffn_swap import (
        SwapInstallingBuilder, install_nvfp4_ffn)

    pipe = flash_rt.load_model(
        checkpoint="/workspace/data/models/LTX-2.5", config="ltx25",
        attention="sdpa", fuse=False, compile_mode=None)
    front = pipe.pipeline
    front.set_prompt("A golden retriever running through a sunny meadow")
    inner = getattr(front, "_pipe")

    # baseline: sdpa, no swap
    with torch.no_grad():
        base = inner.stage._build_transformer()
        out0 = base(**mk(0))
        print(f"host out: video {out0[0].shape} "
              f"finite={torch.isfinite(out0[0]).all().item()} "
              f"std={out0[0].float().std().item():.4f}", flush=True)

    # FFN swap only
    inner.stage = inner.stage.with_builder(SwapInstallingBuilder(
        inner.stage._transformer_builder, [install_nvfp4_ffn]))
    with torch.no_grad():
        m1 = inner.stage._build_transformer()
        out1 = m1(**mk(0))
        print(f"FFN-swap out: std={out1[0].float().std().item():.4f} "
              f"finite={torch.isfinite(out1[0]).all().item()}", flush=True)
    print(f"FFN-only cos: {torch.nn.functional.cosine_similarity(out0[0].float().reshape(-1), out1[0].float().reshape(-1), dim=0):.6f}", flush=True)

    # sage2 attention only (on the ORIGINAL builder path via with_attention)
    attn = make_ltx25_attention("sage2-fvk")
    inner.stage = inner.stage.with_attention(attn)
    with torch.no_grad():
        m2 = inner.stage._build_transformer()
        out2 = m2(**mk(0))
        print(f"sage2-only out: std={out2[0].float().std().item():.4f} "
              f"finite={torch.isfinite(out2[0]).all().item()}", flush=True)
    print(f"sage2-only cos: {torch.nn.functional.cosine_similarity(out0[0].float().reshape(-1), out2[0].float().reshape(-1), dim=0):.6f}", flush=True)

    # both
    inner.stage = inner.stage.with_builder(SwapInstallingBuilder(
        inner.stage._transformer_builder, [install_nvfp4_ffn]))
    with torch.no_grad():
        m3 = inner.stage._build_transformer()
        out3 = m3(**mk(0))
        print(f"combo out: std={out3[0].float().std().item():.4f} "
              f"finite={torch.isfinite(out3[0]).all().item()}", flush=True)
    print(f"combo cos: {torch.nn.functional.cosine_similarity(out0[0].float().reshape(-1), out3[0].float().reshape(-1), dim=0):.6f}", flush=True)
    pipe.close()


if __name__ == "__main__":
    main()
