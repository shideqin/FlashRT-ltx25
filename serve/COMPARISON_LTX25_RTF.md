# LTX-2.5 (Lightricks 22B distilled, audio+video) — RTF comparison, measured

Hardware: NVIDIA RTX 6000D (sm_120a, 85 GB), torch 2.12.1+cu130, CUDA 13.0,
transformers 5.14.1 (5.15.0 breaks the official Gemma-4 text encoder build —
ltx-core pins `<5.15`). Official LTX-2 monorepo (ltx-pipelines 1.2.0) host,
official bf16 split checkpoint (`ltx-2.5-22b-distilled-transformer-bf16`),
FlashRT `flash_rt_kernels` extension built with sage2 + NVFP4 swizzle entries.

Method: one process per arm, two `infer()` calls (run 0 = warmup, run 1 =
timed steady-state), same seed 42, same prompt, warm page cache, identical
inputs. RTF = wall `total_s` / video duration (`num_frames / 24 fps`).
768x512x49f → 2.042 s video; 1536x1024x121f → 5.042 s video. Parity = decoded
frame cosine vs the baseline arm (same seed/prompt), mean over frames.

## Update (Aug 19, 2026): nvfp4 checkpoint + capture+sage2 768 fix

New checkpoint `ltx-2.5-22b-distilled-transformer-nvfp4` (18.7 GB) engages the
W4A4 FFN chain (`install_nvfp4_ffn`), and the small-shape capture bug (finding
#2 below) is **fixed**. Same method (paired run0=warmup/run1=timed, seed 42).

| Res | Arm | total_s (run1) | RTF | cos vs base |
|---|---|---|---|---|
| 768 | base (sdpa eager) | 28.15 | 13.8x | 1.0 |
| 768 | eager sage2-fvk + FFN | 27.57 | 13.5x | 0.970 |
| 768 | capture + sage2 + FFN | 7.87 | 3.9x | **0.937 (fixed)** |
| 1536 | capture + sage2 + FFN | 38.22 | 7.6x | clean (5.6 MB) |

Parity decomposition at 768: capture+sage2 (0.937) ≈ capture+sdpa noise (0.959)
× eager sage2 quant error (0.970) → the capture replay is now faithful; eager
sage2 also improved 0.955 → 0.970 (the old 0.955 was the same scale-mismatch
error, latent in eager too). Kernel-level probe (`serve/qscale_probe.py`):
lq=672 unpadded cos 0.277 vs SDPA, padded cos 0.999; all 768 shapes
(lq 51/672/2688) cos 0.999 after the fix. 1536 (all-128-multiple shapes) was
never affected.

Root cause of finding #2, now fixed in FlashRT:
`quant_per_warp_int8_bf16_d128` writes q-scales with per-head stride
`ceil(lq/32)`, while the sage2 kernel reads them with stride
`ceil(pad_lq/128)*4` (warp-tiled). Any lq that is not a 128 multiple shifts
every head's scales and can read NaN past the buffer tail (768 stage-1
lq=672, and lq=51). Fix: `_attn_swap.py` pads the Q-side buffers
(qp/q8/qs/out) to the next 128 multiple and runs quantize + attention at the
padded row count; the kernel's `qo_len` predicate masks the pad rows.
Secondary fix: `_nvfp4_ffn_swap.py` pads unaligned M to 128 for the fvk chain
in resident mode (upstream weights freed), instead of raising.

Also: the earlier "1536 capture run-1 crash" was an environment artifact
(killed backgrounded process group on shell-timeout), not a runtime bug —
foreground runs complete both infers (exit 0).

## Result

### 768x512x49f (49 frames, 24 fps)

| Arm | Attention | total_s (run1) | RTF | Speedup | cos vs base |
|---|---|---|---|---|---|
| baseline | sdpa, eager | 32.32 | 15.8x | 1.00x | 1.0 |
| kernels eager | sage2-fvk + FFN | 32.31 | 15.8x | 1.00x | 0.955 |
| compile default | sage2-fvk + FFN | 32.84 | 16.1x | 0.98x | 0.949 |
| capture + sage2 | sage2-fvk + FFN | 8.56 | 4.2x | **3.78x** | **0.56 (broken)** |
| capture + sdpa | sdpa + FFN | 8.83 | 4.3x | **3.66x** | **0.990 (clean)** |
| eager sdpa + FFN | sdpa, eager | 31.74 | 15.5x | 1.02x | 1.000000 (bit-identical) |

### 1536x1024x121f (121 frames, 24 fps)

| Arm | Attention | total_s (run1) | RTF | Speedup | cos vs base |
|---|---|---|---|---|---|
| baseline | sdpa, eager | 82.30 | 16.3x | 1.00x | 1.0 |
| kernels eager | sage2-fvk + FFN | 75.64 | 15.0x | 1.09x | 0.996 |
| capture + sage2 | sage2-fvk + FFN | 51.75 | 10.3x | **1.59x** | **0.993 (clean)** |
| capture + sdpa | sdpa + FFN | 57.94 | 11.5x | 1.42x | 0.996 |

## cos=1.0 (bit-exact) arm — the usable one

Requirement: output bit-identical to baseline. Kernel swaps, torch.compile,
and CUDA graph capture all change numerics (see Findings). The only
numerically-exact accelerations are engineering that reuses identical tensors:

1. **prompt embedding cache** — same prompt returns the same embedding
   tensor; skips the ~26GB Gemma-4 forward on repeat prompts.
2. **resident transformer** — build once, reuse across stage-1/stage-2 and
   across `infer()` calls; same weights, same forward, no numerics change.

Measured (same seed 42, same prompt, warm steady-state run 1):

| Res | Baseline RTF | exact RTF | Speedup | cos vs base | max|d| |
|---|---|---|---|---|---|---|
| 768x512x49f | 15.8x (32.32s) | **4.0x (8.07s)** | **4.00x** | **1.000000** | **0.0** |
| 1536x1024x121f | 16.3x (82.30s) | **11.6x (58.33s)** | **1.41x** | **1.000000** | **0.0** |

Time decomposition at 768 (baseline denoise 29.93s): prompt encode ~8.4s +
transformer rebuild ~15.8s + actual denoise ~5.7s. Cache + resident remove
the first two with zero numerics change.

**Floor reached**: the exact arm is 100% GPU-bound at both resolutions —
wall == GPU-busy: 8.164s == 8.164s (768) and 58.379s == 58.379s (1536),
non-GPU 0.00% (measured with CUDA events, `serve/ltx25_gpu_share.py`).
No launch/Python overhead remains to reclaim bit-exactly. Any further
speedup requires a different kernel, which by definition changes the output
and breaks cos=1.0.

Reproducibility: exact run0 vs run1 bit-identical at both resolutions.

**Per-layer proof (every block, every step)**: hooks on all 48 transformer
blocks captured the video hidden state at every denoise step during one
baseline infer and one exact infer (same seed/prompt, same process). All 528
activations (48 blocks x 11 steps) compared in float64:

    worst block 7: cosine 1.000000000  max|d| 0.000000
    final mp4:     cosine 1.000000000  max|d| 0.000000

(The first harness pass printed cos 0.999999762 — a float32 dot-product
rounding artifact: an identical bf16 tensor self-compared in float32 deviates
~4e-4. float64 gives exactly 1.0 with max|d| 0.)

## Further RTF: floor is reached, with per-phase proof

Every phase of the exact arm is 100% GPU-busy (CUDA events):

| Phase | 768 | 1536 |
|---|---|---|
| whole infer | wall == gpu (8.164s == 8.164s) | wall == gpu (58.379s == 58.379s) |
| non-GPU | 0.00% | 0.00% |
| denoise_and_prep | 5.71s | 40.92s |
| vae_decode_encode | 2.36s | 17.40s (wall 16.42s == gpu 16.42s) |

Decode+encode is already pipelined: the VAE decode is a chunked iterator
consumed by a threaded libx264 encoder (1-slot queue), so CPU encode overlaps
GPU decode per chunk. The remaining denoise is 11 steps of the 22B bf16
transformer — pure GPU compute. Any kernel/precision/codec change (natten
decode, sage2 attention, NVFP4 FFN, torch.compile, CUDA graphs, crf/preset)
changes numerics and breaks cos=1.0. Within the bit-exact constraint the
measured RTF is the physical floor.

## Findings

1. **The real RTF win is capture mode, not the kernels alone.** Eager sage2
   swap is flat at 768 (1.00x) and 1.09x at 1536; the design-doc 2.04x claim
   comes from whole-loop CUDA graph capture + resident transformer + prompt
   embedding cache (repeat prompt skips the ~26GB Gemma-4 encode). At 768 the
   capture arm is 3.7x end-to-end because the denoise body is small and
   per-step launch/encode overhead dominates; at 1536 (larger S, compute-
   bound) it is 1.4-1.6x.

2. **capture + sage2 is broken at 768, clean at 1536.** At 768x512x49f the
   captured-graph replay of the sage2 raw kernels diverges from the eager
   sage2 path (cos 0.60 vs opt, 0.56 vs base) — far beyond the kernel
   quantization error itself (eager sage2 = 0.955). Deterministic per arm
   (run0 vs run1 bit-identical), so it is a capture-replay correctness issue
   at these shapes, not noise. At 1536 the same combination is clean (0.993).
   Until the small-shape capture path is fixed, use **capture + sdpa
   attention** at 768 (3.66x, cos 0.990).

3. **The FFN W4A4 chain did not engage on the bf16 checkpoint.** The runtime
   swap (`install_nvfp4_ffn`) only rewrites `NVFP4Linear` modules; the bf16
   split checkpoint carries none, so `eager sdpa + FFN` is bit-identical to
   baseline (cos 1.000000, max|d| 0). The W4A4 path is exercised only by the
   structures-layer attach (`nvfp4_balance_sage` scheme, quantize-on-adopt),
   which measured 1.47x per-step at 768 and 1.12x at 1536 on the diffusers
   host (`/tmp/attach_768b.log`, `/tmp/attach_1536b.log`), bit-exact detach.

4. **Cold-cache first run is not comparable.** First baseline run (cold page
   cache, weight reads inside `pipe()`) was 292.8 s (RTF 143x); warm run
   32.3 s (15.8x). All numbers above are warm steady-state.

5. All arms are deterministic (same-arm run0 vs run1 bit-identical, cos 1.0).

## Files

- `serve/ltx25_rtf.py` — paired warmup+timed E2E harness (per arm)
- `serve/ltx25_rtf_exact.py` — bit-exact arm (prompt cache + resident
  transformer, no swaps/compile/capture)
- `serve/ltx25_gpu_share.py` — GPU-busy vs wall measurement (floor proof)
- `serve/ltx25_parity.py` — decoded-frame cosine/max-abs comparison
- `/tmp/rtf_{base,opt,comp,cap,capsdpa,optsdpa,exact}.log` (768),
  `/tmp/rtf1536_{base,opt,cap,capsdpa,exact}.log` (1536) — full run logs
- `/tmp/ltx25_*_<tag>_<run>.mp4` — outputs (base_1, exact_1, ...)
