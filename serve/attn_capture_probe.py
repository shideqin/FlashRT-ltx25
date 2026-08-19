"""Isolated CUDA-graph capture probe for FvkSage2Attention at 768 stage-1 shape.

Checks whether graph capture + replay reproduces the eager output, both for a
single call and for two back-to-back calls sharing the internal buffers (the
48-block reuse pattern).
"""
import os
import sys

os.environ.setdefault("FLASH_RT_LTX2_ROOT", "/workspace/data/LTX-2")
os.environ.setdefault("HF_HOME", "/workspace/data/hf_cache")
sys.path.insert(0, "/workspace/data/FlashRT")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def cos(a, b):
    return float(F.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0))


def main():
    from flash_rt.models.ltx25._attn_swap import make_ltx25_attention
    attn = make_ltx25_attention("sage2-fvk")
    torch.manual_seed(0)
    b, lq, lk, h, d = 1, 672, 672, 32, 128

    q = torch.randn(b, lq, h * d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(b, lk, h * d, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(b, lk, h * d, dtype=torch.bfloat16, device="cuda")

    o_eager = attn(q, k, v, h)
    torch.cuda.synchronize()

    sq, sk, sv = q.clone(), k.clone(), v.clone()
    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1):
        o_cap = attn(sq, sk, sv, h)
    g1.replay()
    torch.cuda.synchronize()
    print(f"single: eager_finite={bool(torch.isfinite(o_eager).all())} "
          f"cap_finite={bool(torch.isfinite(o_cap).all())} "
          f"cos={cos(o_eager, o_cap):.6f}")

    q2 = torch.randn(b, lq, h * d, dtype=torch.bfloat16, device="cuda")
    k2 = torch.randn(b, lk, h * d, dtype=torch.bfloat16, device="cuda")
    v2 = torch.randn(b, lk, h * d, dtype=torch.bfloat16, device="cuda")

    r1e = attn(q, k, v, h) * 2.0
    r2e = attn(q2, k2, v2, h) * 2.0
    torch.cuda.synchronize()

    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        c1 = attn(q, k, v, h)
        r1c = c1 * 2.0
        c2 = attn(q2, k2, v2, h)
        r2c = c2 * 2.0
    g2.replay()
    torch.cuda.synchronize()
    print(f"two-call: r1 cos={cos(r1e, r1c):.6f} "
          f"r2 cos={cos(r2e, r2c):.6f} "
          f"r1_finite={bool(torch.isfinite(r1c).all())} "
          f"r2_finite={bool(torch.isfinite(r2c).all())}")


if __name__ == "__main__":
    main()
