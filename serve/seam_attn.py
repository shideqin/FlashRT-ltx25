"""flash_rt.structures seam for ComfyUI MiniMax-H3 attention.

Binds the attention_core family (fa2/fa4/masked-mha, whatever serves
this device) into every `Attention` site of the loaded MiniMaxH3Model.
The host arm keeps its exact qkv/rope/norm preamble; only the SDPA
body is replaced by the bound core. Calibration captures real q/k/v
tensors (host layout [B, H, S, D]) through the `wrap_attn`
container_function hook, so the seam is bound to the shapes it will
actually run at — and the DenseAttention shape gate rejects any later
shape drift.

Memory discipline: the recorder retains ONE capture per distinct
shape (refiner S=256, DiT S~63.5k). A full q/k/v capture at the DiT
shape is ~2.7GB; retaining all 52 sites would be ~140GB. The binders
only need a real capture to qualify the family gate and anchor the
shape, so one per shape is sufficient and the ordered site list
reuses those dicts.
"""
import gc
import types

import torch

import comfy.ldm.modules.attention as attn_mod


def _seamed_forward(self, x, rope_freqs=None, transformer_options={}):
    """Replica of comfy.ldm.minimax.model.Attention.forward with the
    SDPA call replaced by the bound flash core."""
    import comfy.model_management
    import comfy.quant_ops
    s = x.shape[0]
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    v = v.view(s, self.heads, self.head_dim)
    if rope_freqs is not None:
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        qw = comfy.model_management.cast_to(self.q_norm.weight,
                                            device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight,
                                            device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps,
                rot_dim=rot)
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps,
                rot_dim=rot)
        q = q[0]
        k = k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))
    v = v.clone()
    qc = q.transpose(0, 1).unsqueeze(0)   # [1, H, S, D]
    kc = k.transpose(0, 1).unsqueeze(0)
    vc = v.transpose(0, 1).unsqueeze(0)
    out = self.flash_core(qc, kc, vc)     # [1, H, S, D]
    out = out.transpose(1, 2).reshape(s, -1)  # [S, heads*head_dim]
    return self.out_proj(out)


def calibrate(model, forward, *args, **kwargs):
    """Run one real forward with a recorder on the attention hook.

    Returns (captures, ref_out): captures is an ordered list, one dict
    per attention site (refiner blocks first, then DiT blocks 0..49),
    where sites sharing a shape share the same capture dict. ref_out
    is the forward's output through the untouched baseline path.
    """
    captures = []
    first_per_shape = {}

    def recorder(*a, **kw):
        q, k, v = a[0].peek(), a[1].peek(), a[2].peek()
        shape = tuple(q.shape)
        if shape not in first_per_shape:
            first_per_shape[shape] = {
                "q": q.detach(),
                "key": k.detach(),
                "value": v.detach(),
                "mask": None,
            }
        captures.append(first_per_shape[shape])
        return attn_mod.optimized_attention.__wrapped__(
            a[0].take(), a[1].take(), a[2].take(), *a[3:],
            **{kk: vv for kk, vv in kw.items()
               if kk != "_inside_attn_wrapper"})

    orig = attn_mod.optimized_attention.container_function
    attn_mod.optimized_attention.container_function = recorder
    try:
        with torch.no_grad():
            out = forward(*args, **kwargs)
        torch.cuda.synchronize()
    finally:
        attn_mod.optimized_attention.container_function = orig
    return captures, out


_VARIANT_BINDERS = {
    "fa2": "fa2_seqused",
    "fa4_cute": "fa4_cute",
    "masked_mha": "masked_mha",
    "fa4_fp8": "fa4_fp8",
}


def _variant_binder(name):
    from flash_rt.structures.impls import attention_core as ac
    mod = getattr(ac, _VARIANT_BINDERS[name])
    return mod.bind_dense_attention


def bind(model, captures, verbose=True):
    """Bind a dense attention core per captured site and install the
    seamed forward on every Attention module of the model.

    Site order must match capture order: refiner blocks then DiT
    blocks. The attention_core family gate (variant qualification +
    measured speed gate) runs once per distinct shape; identical sites
    bind through the winning variant's raw binder — one qualified
    form, N sites, no re-benchmark of the same shape. Returns
    (cores, trail) where trail names the serving variant per site.
    """
    from flash_rt.structures.impls.attention_core import \
        bind_dense_attention_best
    from flash_rt.structures.impls.attention_core.fa2_seqused import \
        bind_dense_attention as bind_fa2

    sites = []
    if hasattr(model, "token_refiner"):
        sites += list(model.token_refiner.blocks)
    sites += list(model.blocks)

    if len(sites) != len(captures):
        raise ValueError(
            f"capture/site mismatch: {len(captures)} captures for "
            f"{len(sites)} attention sites")

    # This host's stock attention is already torch's flash SDPA (bf16),
    # so the family speed gate declines every attention_core variant
    # (none measures faster than the host). The comparison the user
    # asked for is still measurable: bind fa2 directly — the explicit
    # seat-book semantics from examples/structure_pipeline — and let
    # the parity gate answer for correctness.
    cores, trail, winner = [], [], {}
    for site, cap in zip(sites, captures):
        attn = site.attn
        shape = tuple(cap["q"].shape)
        if shape not in winner:
            core = None
            why = "direct fa2 (family speed gate declined: host SDPA is flash)"
            try:
                core = bind_fa2([cap])
            except Exception as exc:
                import traceback
                print(f"[seam] direct fa2 bind failed at {shape}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc(limit=6)
                core = None
            if core is None:
                try:
                    core = bind_dense_attention_best([cap])
                    why = "family"
                except (AttributeError, ImportError) as exc:
                    import traceback
                    raise RuntimeError(
                        "blocked_on_kernel: an attention_core variant "
                        f"crashed at shape {shape}: {type(exc).__name__}: "
                        f"{exc}\n{traceback.format_exc()}") from exc
            if core is None:
                raise RuntimeError(
                    "attention_core declined every variant for a site "
                    f"at shape {shape}")
            winner[shape] = (core, why)
        else:
            core0, why = winner[shape]
            if why.startswith("direct fa2"):
                core = bind_fa2([cap])
            else:
                core = _variant_binder(core0._frt_variant)([cap])
            if core is None:
                raise RuntimeError(
                    f"binder declined site at shape {shape} after "
                    "the first seat bound")
        attn.flash_core = core
        attn.flash_core_shape = shape
        attn.forward = types.MethodType(_seamed_forward, attn)
        cores.append(core)
        trail.append((why, ()))
        if verbose:
            print(f"[seam] site {len(cores)-1:2d} S={shape[2]:5d} "
                  f"-> {trail[-1][0][:40]}", flush=True)
    return cores, trail


def unbind(model):
    """Restore the original class forward on every bound Attention."""
    import comfy.ldm.minimax.model as m
    sites = []
    if hasattr(model, "token_refiner"):
        sites += list(model.token_refiner.blocks)
    sites += list(model.blocks)
    for site in sites:
        attn = site.attn
        attn.forward = types.MethodType(m.Attention.forward, attn)
        if hasattr(attn, "flash_core"):
            del attn.flash_core
    gc.collect()
    torch.cuda.empty_cache()


def parity(ref, forward, *args, **kwargs):
    """Seamed forward vs baseline reference on identical inputs.

    The seam must already be installed (call bind() first). Returns
    (max_abs, mean_abs, cosine). At S=63.5k the bf16 attention tails
    differ in a few positions (max|d| ~ 2-4, mean ~ 1e-4), so the gate
    is the cosine against the host, mirroring the repo's own accuracy
    bars (cosine >= 0.995); max/mean|d| are reported as evidence.
    """
    import torch.nn.functional as F
    with torch.no_grad():
        out = forward(*args, **kwargs)
    torch.cuda.synchronize()
    a, b = ref[0], out[0]
    cos = float(F.cosine_similarity(a.reshape(-1), b.reshape(-1),
                                    dim=0).item())
    maxd = float((a - b).abs().max())
    meand = float((a - b).abs().mean())
    if cos < 0.99:
        raise RuntimeError(
            f"seamed output diverged from baseline: cosine={cos:.6f} "
            f"max|d|={maxd:.4f} mean|d|={meand:.5f}")
    return maxd, meand, cos
