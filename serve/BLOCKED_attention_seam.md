# MiniMax-H3 NVFP4 — flash_rt.seams attention arm: BLOCKED_ON_KERNEL

Host: ComfyUI MiniMaxH3Model (`comfy/ldm/minimax/model.py`), torch 2.12.1+cu130,
NVIDIA RTX 6000D (sm_120a), CUDA 13.0, kernel hub offline (cache only).

Seam: `flash_rt.structures.impls.attention_core` dense family, bound into
every `Attention` site of the 50-block DiT + 2-block token refiner
(`serve/seam_attn.py`). Boundary: the host's SDPA body
(`comfy.ldm.modules.attention.attention_pytorch`, `skip_reshape=True`) at
captured shape `[1, 56, 63566, 128]` bf16 (480p, 39 frames, text 256).

## Status per variant (family order in `bind_dense_attention_best`)

| Variant | Artifact | Refusal / blocker |
|---|---|---|
| fa2 | `flashrt/fa2-seqused-runtime` >=1 | **absent from local cache**; hub unreachable (`Errno 101`). `KernelUnavailable` (clean refusal). |
| fa4_cute | `kernels-community/flash-attn4` @ed382de (v0) | loads (noarch `torch-cuda` variant) but its **sm_120 forward path is defective at this commit**: `FlashAttentionForwardSm120` inherits the SM80 kernel (`flash_fwd_sm120.py` keeps `arch = 80` for the CpAsync path) yet `FlashAttentionForwardBase.__init__` (`flash_fwd.py:110`) overwrites `self.arch` with the DSL device arch (`sm_120a`) → `self.use_tma_O = True` (`flash_fwd.py:652`) → the SM80 `__call__` never constructs `tma_atom_O` → epilogue passes `None` → `AttributeError: 'NoneType' object has no attribute '_trait'` at `cpasync/helpers.py:209` in `tma_partition`. Fires at **any** sequence length (reproduced at `[1,4,256,128]`). Also cutlass-dsl vintage mismatch: built against `cute.core.ThrMma`-era DSL; 4.7.0 needs 6 aliases shimmed; 4.4.x imports clean but hits the same sm_120 epilogue bug. |
| masked_mha | `flashrt/masked-mha-runtime` v1, archs `['11.0a','12.0']` | **torch-2.11-gated build** (`torch211-cxx11-cu130-…` variant) — refused by the kernel client under torch 2.12 (`Torch version (2.11) does not match … (2.12)`). If it loaded, the workspace is impossible at this shape anyway: persistent `_logits` buffer `[56, 63566, 63568]` bf16 ≈ **452 GB** (`masked_mha.py:72-74`). |
| fa4_fp8 | `flashrt/fp8-cross-attention-blackwell` v1, archs `['10.0a','11.0a']` | **no sm_120 build variant** — refused by the arch check. |
| (sageattention3) | `flashrt/sageattention3-blackwell` v1, archs `['12.0a']` | sm_120a ✓ but **torch-2.11-gated build** — refused under torch 2.12. No `attention_core` impl consumes it. |

## Required to unblock

A kernel artifact usable at (torch 2.12, sm_120a) for dense bf16 attention,
head_dim 128, seq ≈ 63.5k, batch 1, no mask:
- flash-attn4 with a working SM120 forward (fix the arch-overwrite / TMA-O
  atom construction), or
- `fa2-seqused-runtime` published for sm_120a / torch 2.12, or
- masked-mha with a tiled/paged workspace instead of the full `[H,S,S]`
  logits buffer, built for torch 2.12, or
- fa4_fp8 rebuilt with an sm_120a variant, or
- sageattention3-blackwell built for torch 2.12.

Deliver as a published Hub artifact + correctness/compile evidence per the
kernels-side acceptance gates; the harness (`serve/seam_attn.py`,
`serve/bench_seamed.py`) binds whatever the family serves, no harness change
needed.

## Verified run (2026-08-17, `python serve/bench_seamed.py`)

```
[host] model on cuda; params=20.1B
[base] attn   852.9ms/block  mlp   63.2ms/block
[bench] captured 52 attention sites (refiner 2 + dit 50)
RuntimeError: blocked_on_kernel: an attention_core variant crashed at
  shape (1, 56, 256, 128): AttributeError: module 'cutlass.cute.core'
  has no attribute 'ThrMma'
```

Load, baseline per-block timing, and calibration all complete; the bind
stops at the first variant the family reaches after fa2's clean refusal
(fa4_cute import crash — cutlass-dsl vintage mismatch under 4.7.0; under
4.4.x it imports but dies at the sm_120 TMA-O epilogue bug). The blocker
is raised by `seam_attn.bind` with the full traceback, so a broken
artifact is evidence, not a bare crash.

## How to run the comparison once an artifact lands

```
cd /workspace/data/serve && python bench_seamed.py
```

No harness change needed: `seam_attn.bind` binds whatever the
attention_core family serves first. Expected output: per-block
attn/mlp for both arms, parity max|d|, seamed 20-step e2e, and the
comparison table against the baseline in this file. Baseline e2e
reference: 918.8 s total / 45.9 s/step / RTF 565× (see below).

## What is verified working (harness side)

- Calibration via the `wrap_attn` container hook captures all 52 sites at
  real shapes (`[1,56,256,128]` refiner ×2, `[1,56,63566,128]` DiT ×50).
- `bind_dense_attention_best` refusal ladder executes: fa2 unavailable →
  fa4_cute loads → masked_mha → fa4_fp8, each refusal recorded.
- Baseline arm fully measured (see below) with identical inputs/noise and
  the same warmup for both arms.

## Baseline (reference, unmodified host)

| Metric | Value |
|---|---|
| per-block attn (S=63,566, σ=0.5) | 851.7 ms |
| per-block mlp | 63.0 ms |
| projected 50-block attn / mlp | 42.6 s / 3.2 s per step |
| e2e 20 steps @480p/39f | 918.8 s total, 45.9 s/step, RTF 565× |

Attention is 93% of the block cost; the seam target is exactly that body.
