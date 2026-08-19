# serve/ — LTX-2.5 RTF benchmark & diagnostic harness (FlashRT)

Harness for measuring and comparing LTX-2.5 22B-distilled (audio+video)
inference on the FlashRT `sage2-fvk` attention + NVFP4 W4A4 FFN runtime.

## Expected environment (this harness was run on)

- NVIDIA RTX 6000D (sm_120a, 85 GB), CUDA 13, torch 2.12.1+cu130
- `transformers` pinned `<5.15` (5.15 breaks the Gemma-4 text encoder build)
- Checkpoint tree at `/workspace/data/models/LTX-2.5` with
  `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors`
  (18.7 GB, quantize-on-load W4A4) or `-bf16.safetensors` (42 GB)
- Env vars: `FLASH_RT_LTX2_ROOT=<ltx2 repo>` (defaults `/workspace/data/LTX-2`),
  `HF_HOME=<cache>`, `HF_ENDPOINT=https://hf-mirror.com`
- `flash_rt_kernels` built from the FlashRT repo with sage2 + NVFP4 entries
- `natten` recommended (`uv sync --package ltx-core --extra natten`); without
  it the VAE decode falls back to a slower Triton `na3d` path

## Key scripts

- `ltx25_rtf.py` — paired warmup+timed E2E harness.
  Flags: `--res 768x512x49f|1536x1024x121f`, `--attention sage2-fvk|sdpa`,
  `--fuse`, `--compile capture` (omit `--compile` for eager), `--tag NAME`.
  Prints `[rtf] run N: ... RTF ...` lines and writes `<tag>_<run>.mp4`.
- `ltx25_rtf_exact.py` — bit-exact arm (prompt embedding cache + resident
  transformer; no swaps/compile/capture), cos = 1.000000.
- `ltx25_parity.py` — decoded-frame cosine/max-abs between two mp4s.
- `ltx25_gpu_share.py` — GPU-busy vs wall time (floor proof).
- Diagnostic probes: `attn_shape_probe.py` (real eager shapes),
  `qscale_probe.py` (q-scale layout mismatch proof),
  `attn_fix_probe.py` (post-fix padded-lq parity vs SDPA).

## Known issues (fixed / resolved)

- **capture + sage2 black at 768** — FIXED. q-scale layout mismatch for
  non-128-multiple lq (see COMPARISON_LTX25_RTF.md). `_attn_swap.py` pads the
  Q-side buffers to a 128 multiple; `_nvfp4_ffn_swap.py` pads unaligned M for
  the resident FFN chain.
- **1536 capture run-1 "crash"** — environment artifact (backgrounded process
  group killed on shell timeout); foreground runs complete both infers.

## Results

See `COMPARISON_LTX25_RTF.md` (bf16-era table + Aug-19 nvfp4 update).
768 capture+sage2+FFN: RTF 3.9x, cos 0.937. 1536 capture+sage2+FFN: RTF 7.6x.
