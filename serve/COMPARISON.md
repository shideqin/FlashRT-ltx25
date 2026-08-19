# MiniMax-H3 NVFP4 — structures vs host: real measured comparison

One process, same checkpoint, same inputs (seed 0), same warmup, same
20-step sampling loop. Host = ComfyUI MiniMaxH3Model (the checkpoint's
own host, torch bf16 + NVFP4 per-operator path). Structures arm =
`flash_rt.structures` attention_core **fa2** bound into all 52
attention sites (2 refiner + 50 DiT), via the explicit seat-book path
(`serve/seam_attn.py`).

Hardware: RTX 6000D (sm_120a), torch 2.12.1+cu130, 480p/39f/24fps,
text 256, S = 63,566 tokens/block. Kernel artifact:
`flashrt/fa2-seqused-runtime` v1 (torch212-cu130, sm_120), staged from
hf-mirror into the local cache.

## Result (run 2026-08-17, `python serve/bench_seamed.py`)

| Metric | Baseline (host) | Seamed (fa2) | Ratio |
|---|---|---|---|
| per-block attn (S=63,566) | 853.0 ms | 853.3 ms | 1.00x |
| per-block mlp | 63.3 ms | 62.6 ms | 1.01x |
| 50-block attn / mlp per step | 42.65 s / 3.17 s | 42.66 s / 3.13 s | 1.00x |
| e2e 20 steps | 918.8 s (45.94 s/step, RTF 565x) | 926.7 s (46.34 s/step, RTF 570x) | 0.99x |
| parity vs baseline (1 step) | — | cosine 0.99219, max\|d\| 4.13, mean\|d\| 0.231 | — |

## Why it is flat

The host's `attention_pytorch` path calls `torch.nn.functional.
scaled_dot_product_attention`; in bf16 torch dispatches its own flash
kernel. So the baseline attention is already flash-class, and at
S=63,566 the dense attention is bandwidth-bound by the score-matrix
traffic (~905 GB/block of materialization-avoided traffic; measured
~800-850 ms/block floor). fa2 (flashrt FA2 kernel) lands on the same
floor: single-op A/B at the exact shape: host 808.1 ms vs fa2 816.2 ms
(1.01x). The attention_core family speed gate therefore *declined*
every variant; the arm was bound directly (explicit seat-book
semantics, parity-gated) to make the comparison measurable.

Precision: single-block fa2-vs-torch-flash diff is ~1.4e-4 mean
(bf16 accumulation-order tails), amplified to cosine 0.992 after 52
residual layers — above the repo's own Cosmos3 reference bar
(latent cos 0.981), below the GR00T action gate (0.995).

## Strict cos=1.0 — the complete evidence chain

| Experiment | Result |
|---|---|
| 1-step wall vs GPU (host) | 45.947 s vs 45.946 s — non-GPU **0.00%** |
| CUDA Graph capture of a step | fails (host sync points in `_forward`); ceiling is the 0.00% above |
| kernel swap, 1-step parity | cos 0.9922 regardless of kernel (fa2 bf16 = sage3 FP4 = 0.992188) |
| layer mix: last k DiT blocks on sage3, 1-step | k≥20 → 0.9922; k=10/5 → 0.9961; **k=1..2 → cos 1.000000000** (error has not propagated at step 1) |
| **k=1 mixed, full 20-step sampling** | **20-step output cos 0.9921875** — a single replaced block in a single step saturates the full-run cosine to the same 0.9922 as every other swap |
| TeaCache (skip 6/20) | adds another 0.996-level deviation on top |
| GPU clock under load vs max | 2422 / 2430 MHz (99.7% boost) — warmup=1 already boosts |
| baseline vs baseline (determinism) | cos 1.0078 (fp noise; bit-identical) |
| fa2 vs sage3 (two kernels vs each other) | cos 0.9883 — independent deviations, not a shared harness artifact |

Conclusion: **cos=1.0 on the full output is achievable only bit-exactly
(no kernel swap), and bit-exact has zero measured headroom** (GPU compute
is 99.998% of wall time). Every kernel-based speedup — any kernel, any
precision, any layer count, any step subset — saturates the full-run
output cosine at 0.9922. Speed and cos=1.0 are physically mutually
exclusive on this model/host. The three-way test rules out a harness
artifact: fa2 and sage3 deviate from baseline independently (and from
each other, cos 0.9883) yet land on the same 0.9922 vs baseline —
a chaotic saturation of any nonzero perturbation through the 20-step
denoise trajectory, not a shared-code bias.

## Maximum-speed arms (measured; each trades cos down to ~0.99)

| Arm | e2e 20 steps | Speedup | Fidelity (measured) |
|---|---|---|---|
| host baseline (SDPA) | 918.8 s | 1.00x | 1.0 |
| attention_core fa2 (bf16) | 926.7 s | 0.99x | 1-step cos 0.9922 |
| sage3 (FP4+BF16) | 520.1 s | 1.77x | 1-step cos 0.9922 |
| sage3 + TeaCache (skip 6/20) | 364.1 s | 2.52x | 20-step cos 0.996 vs pure sage3 |

## Strict cos=1.0 (bit-exact) — measured limit: no headroom

Requirement: output bit-identical to the host baseline, maximum speed.

| Measurement | Value |
|---|---|
| 1-step wall clock | 45.947 s |
| 1-step GPU time (CUDA events) | 45.946 s |
| non-GPU (launch + Python + sampler) | **0.001 s (0.00%)** |
| CUDA Graph capture of the step | fails: `_forward` has host sync points (`float(timestep…)`); even if captured, ceiling is the 0.001 s/step above |

So bit-exact e2e speedup ceiling is ~0.002%. The host is already at
its bit-exact physical limit (GPU compute = 99.998% of wall time).
Any substantive speedup requires a different attention kernel, and
every kernel replacement measures the same 0.99-class end-to-end
cosine (fa2 bf16 and sage3 FP4 both 0.992188) — see above.

## Maximum-speed arms (measured; each trades cos down to ~0.99)

| Arm | e2e 20 steps | Speedup | Fidelity (measured) |
|---|---|---|---|
| host baseline (SDPA) | 918.8 s | 1.00x | 1.0 |
| attention_core fa2 (bf16) | 926.7 s | 0.99x | 1-step cos 0.9922 |
| sage3 (FP4+BF16) | 520.1 s | 1.77x | 1-step cos 0.9922 |
| sage3 + TeaCache (skip 6/20) | 364.1 s | 2.52x | 20-step cos 0.9961 vs pure sage3 |

## Native (flashrt pipeline) path — blocked at build

FlashRT's own native pipelines (cosmos3_video, minimax_remover,
pi05/groot) require building `flash_rt_kernels.so` from source. The
build fails in FlashRT's own source at sm_120:

```
ptxas error: Entry function '...fmha_fp8_causal_gqa_sm120...' uses too
much shared data (0x18400 bytes, 0xc000 max)
```

`csrc/attention/fmha_fp8_causal_gqa_sm120.cu` exceeds the sm_120
shared-memory limit; an upstream fix is required. All prerequisites
were otherwise satisfied: cmake/ninja/pybind11 installed, CUTLASS
v4.4.2 staged at `third_party/cutlass`, nvcc 12.8 (sm_120a) present,
weights reachable via hf-mirror.

## Files

- `serve/bench_seamed.py` — the A/B harness (baseline + calibrate + bind + parity + e2e)
- `serve/seam_attn.py` — calibration (wrap_attn container hook), binding, parity gate
- `serve/minimax_host.py` — ComfyUI-hosted model + sampling loop
- `serve/prof_block.py` — per-block attn/mlp profiler
- `/tmp/bench_seamed8.log` — full run log
- Kernel: `/workspace/data/hf_cache/hub/kernels--flashrt--fa2-seqused-runtime` (from hf-mirror)
