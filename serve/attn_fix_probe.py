"""Sanity: the fixed FvkSage2Attention at lq=672/51 now matches SDPA, eager
and captured."""
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


def sdpa_ref(q, k, v, heads):
    b, lq, hd = q.shape
    d = hd // heads
    q4, k4, v4 = (t.view(b, -1, heads, d).transpose(1, 2) for t in (q, k, v))
    o = F.scaled_dot_product_attention(q4, k4, v4)
    return o.transpose(1, 2).reshape(b, -1, hd)


def main():
    from flash_rt.models.ltx25._attn_swap import make_ltx25_attention
    attn = make_ltx25_attention("sage2-fvk")
    torch.manual_seed(0)
    for lq in (51, 672, 2688):
        b, lk, h, d = 1, 672, 32, 128
        q = torch.randn(b, lq, h * d, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(b, lk, h * d, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(b, lk, h * d, dtype=torch.bfloat16, device="cuda")
        ref = sdpa_ref(q, k, v, h)
        torch.cuda.synchronize()
        o_e = attn(q, k, v, h)
        torch.cuda.synchronize()
        # captured replay
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            o_c = attn(q, k, v, h)
        g.replay()
        torch.cuda.synchronize()
        print(f"lq={lq}: eager cos={cos(ref, o_e):.6f} "
              f"captured cos={cos(ref, o_c):.6f} "
              f"finite={bool(torch.isfinite(o_c).all())}")


if __name__ == "__main__":
    main()
