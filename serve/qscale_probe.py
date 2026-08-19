"""Confirm the q-scale stride mismatch at lq=672 vs a padded-scale run.

Both use the fvk raw path; the 'padded' variant feeds a 128-padded Q input so
the quantize writes scales with stride ceil(768/32)=24, matching the attention
kernel's num_warp_block_q. Reference is torch SDPA on the same q/k/v.
"""
import os
import sys

os.environ.setdefault("FLASH_RT_LTX2_ROOT", "/workspace/data/LTX-2")
os.environ.setdefault("HF_HOME", "/workspace/data/hf_cache")
sys.path.insert(0, "/workspace/data/FlashRT")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import flash_rt.flash_rt_kernels as fvk  # noqa: E402


def cos(a, b):
    return float(F.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0))


def sdpa_ref(q, k, v, heads):
    b, lq, hd = q.shape
    d = hd // heads
    q4, k4, v4 = (t.view(b, -1, heads, d).transpose(1, 2) for t in (q, k, v))
    o = F.scaled_dot_product_attention(q4, k4, v4)
    return o.transpose(1, 2).reshape(b, -1, hd)


def run_fvk(q, k, v, heads, padded_q):
    b, lq, hd = q.shape
    d = hd // heads
    lk = k.shape[1]
    if padded_q:
        lq_run = ((lq + 127) // 128) * 128
        qp = torch.zeros(b, lq_run, hd, dtype=torch.bfloat16, device=q.device)
        qp[:, :lq] = q
        qn = qp
    else:
        lq_run = lq
        qn = q
    qn = qn.view(b, lq_run, heads, d).contiguous()
    kn = k.view(b, lk, heads, d).contiguous()
    vn = v.view(b, lk, heads, d).contiguous()
    pad_lk = ((lk + 63) // 64) * 64
    q8 = torch.zeros(b, lq_run, heads, 128, dtype=torch.int8, device=q.device)
    k8 = torch.zeros(b, pad_lk, heads, 128, dtype=torch.int8, device=q.device)
    vt = torch.zeros(b, 128, heads, pad_lk, dtype=torch.bfloat16, device=q.device)
    v8 = torch.zeros(b, 128, heads, pad_lk, dtype=torch.float8_e4m3fn, device=q.device)
    qs = torch.zeros(b, heads, (lq_run + 31) // 32, dtype=torch.float32, device=q.device)
    ks = torch.zeros(b, heads, (lk + 63) // 64, dtype=torch.float32, device=q.device)
    vs = torch.zeros(b, heads, 128, dtype=torch.float32, device=q.device)
    out = torch.zeros(b, lq_run, heads, 128, dtype=torch.bfloat16, device=q.device)
    stream = torch.cuda.current_stream().cuda_stream
    fvk.quant_per_warp_int8_bf16_d128(
        int(qn.data_ptr()), int(q8.data_ptr()), int(qs.data_ptr()),
        b, lq_run, heads, stream)
    fvk.quant_per_block_int8_bf16_d128(
        int(kn.data_ptr()), int(k8.data_ptr()), int(ks.data_ptr()),
        b, lk, heads, stream)
    fvk.concat3_v_transpose_pad_permute_bf16_d128(
        int(vn.data_ptr()), int(vn.data_ptr()), int(vn.data_ptr()),
        int(vt.data_ptr()),
        b, lk, 0, 0, heads,
        int(vn.stride(0)), int(vn.stride(1)),
        int(vn.stride(0)), int(vn.stride(1)),
        int(vn.stride(0)), int(vn.stride(1)), stream)
    fvk.v_tpp_bf16_quant_fp8_d128(
        int(vt.data_ptr()), int(v8.data_ptr()), int(vs.data_ptr()),
        b, lk, heads, stream)
    rc = fvk.sage2_qk_int8_sv_f8_bf16_nhd_d128(
        int(q8.data_ptr()), int(k8.data_ptr()), int(v8.data_ptr()),
        int(out.data_ptr()), int(qs.data_ptr()), int(ks.data_ptr()),
        int(vs.data_ptr()), b, lq_run, lk, heads, float(d ** -0.5), stream)
    if rc != 0:
        raise RuntimeError(f"rc={rc}")
    return out[:, :lq].view(b, lq, hd).clone()


def main():
    torch.manual_seed(0)
    b, lq, lk, h, d = 1, 672, 672, 32, 128
    q = torch.randn(b, lq, h * d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(b, lk, h * d, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(b, lk, h * d, dtype=torch.bfloat16, device="cuda")

    ref = sdpa_ref(q, k, v, h)
    torch.cuda.synchronize()
    o_unpad = run_fvk(q, k, v, h, padded_q=False)
    torch.cuda.synchronize()
    o_pad = run_fvk(q, k, v, h, padded_q=True)
    torch.cuda.synchronize()
    print(f"lq=672 vs SDPA:  unpadded cos={cos(ref, o_unpad):.6f}")
    print(f"lq=672 vs SDPA:  padded   cos={cos(ref, o_pad):.6f}")
    print(f"padded vs unpadded cos={cos(o_pad, o_unpad):.6f}")

    torch.manual_seed(1)
    q2 = torch.randn(b, lq, h * d, dtype=torch.bfloat16, device="cuda")
    k2 = torch.randn(b, lk, h * d, dtype=torch.bfloat16, device="cuda")
    v2 = torch.randn(b, lk, h * d, dtype=torch.bfloat16, device="cuda")
    ref2 = sdpa_ref(q2, k2, v2, h)
    torch.cuda.synchronize()
    print(f"lq=672 seed1 vs SDPA: unpadded cos={cos(ref2, run_fvk(q2,k2,v2,h,False)):.6f}")
    print(f"lq=672 seed1 vs SDPA: padded   cos={cos(ref2, run_fvk(q2,k2,v2,h,True)):.6f}")


if __name__ == "__main__":
    main()
