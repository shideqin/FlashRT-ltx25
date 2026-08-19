"""SageAttention3 (flashrt sageattention3-blackwell) adapter for the
ComfyUI MiniMax-H3 host.

Replaces the host SDPA body (host layout [B,H,S,D]) with the Sage3
FP4+BF16 fused prefill. One workspace per distinct shape is shared by
all same-shape sites (the fused workspace is shape-keyed, not
site-keyed). Speed vs host SDPA at S=63,566: ~2.0x; single-layer
cosine vs bf16 SDPA ~0.996.
"""
import torch
from flash_rt.structures.impls import hub_kernel

_s3 = None


def _kernel():
    global _s3
    if _s3 is None:
        _s3 = hub_kernel("flashrt/sageattention3-blackwell", ">=1")
    return _s3


class Sage3Core(torch.nn.Module):
    """Stateless-per-shape Sage3 attention; shares one workspace per
    shape across all sites."""

    _shared_ws = {}

    def __init__(self, q_shape):
        b, heads, seq, head_dim = q_shape
        self.q_shape = tuple(q_shape)
        self.kv_shape = tuple(q_shape)
        if head_dim not in _kernel().SUPPORTED_HEAD_DIMS:
            raise ValueError(f"sage3: head_dim {head_dim} unsupported")
        key = (b, seq, heads, head_dim)
        if key not in Sage3Core._shared_ws:
            q = torch.empty(b, seq, heads, head_dim, dtype=torch.bfloat16,
                            device="cuda")
            Sage3Core._shared_ws[key] = _kernel().allocate_fused_workspace(
                q, q, q)
        self.ws = Sage3Core._shared_ws[key]

    def __call__(self, query, key, value):
        if tuple(query.shape) != self.q_shape:
            raise ValueError(f"sage3: shape moved from {self.q_shape} "
                             f"to {tuple(query.shape)}")
        s3 = _kernel()
        # host [B,H,S,D] -> NHD [B,S,H,D]
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        out = s3.sage3_prefill_fp4_bf16(q, k, v, workspace=self.ws)
        return out.transpose(1, 2)
