"""MiniMax-H3 NVFP4: baseline vs flash_rt.seams, one process.

Measures, on identical model/inputs/warmup:
  - per-block attn/mlp cost, baseline SDPA vs bound attention_core
  - end-to-end 20-step RTF, seamed arm (baseline e2e comes from the
    parallel background run at the same seed/config)
  - seamed-vs-baseline output parity on identical inputs
"""
import json
import os
import sys
import time

import torch

# Kernel artifacts served from the local cache. fa2-seqused-runtime v1
# (torch212-cu130, sm_120) was staged from hf-mirror; the LOCAL_KERNELS
# override resolves it without any hub round-trip.
os.environ["LOCAL_KERNELS"] = (
    "flashrt/fa2-seqused-runtime="
    "/workspace/data/hf_cache/hub/kernels--flashrt--fa2-seqused-runtime")
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, '/data/ComfyUI')
sys.path.insert(0, '/workspace/data/serve')

import gc

from minimax_host import MiniMaxHost, build_sigmas
from seam_attn import calibrate, bind, parity, unbind
from comfy.ldm.minimax.model import (PackedLayout, rope_rotation_table,
                                     patchify_video, pack_audio)

FRAMES = 39
TEXT_LEN = 256
STEPS = 20
SIGMA_V = 0.5  # mid-schedule profile step for the per-block timing
NUM_TIMED = 10  # blocks 0..9 timed per arm

h = MiniMaxHost(frames=FRAMES, text_len=TEXT_LEN, steps=STEPS)
h.load()
model = h.model
dev = next(model.parameters()).device
v, a, c = h.inputs()
v, a, c = v.to(dev), a.to(dev), c.to(dev)
payload = h.payload()
layout = PackedLayout(TEXT_LEN, v.shape[2], v.shape[3], v.shape[4],
                      a.shape[-1])


def denoise_step(x, sigma):
    """One model call the way res_multistep makes it."""
    with torch.no_grad():
        out = model(x, sigma * torch.ones([1], device=dev), c,
                    transformer_options={}, minimax_payload=payload)
    torch.cuda.synchronize()
    return out


def build_block_inputs():
    """Mirror of _forward up to the block loop, at SIGMA_V."""
    latent_t, lat_h, lat_w = v.shape[2], v.shape[3], v.shape[4]
    audio_t = a.shape[-1]
    t_v = float(1.0 - SIGMA_V)
    t_a = float(1.0 - 0.5)
    seg_t = {"text": t_v, "video": t_v, "audio": t_a,
             "cond": t_v, "ref_img": t_v, "cond_audio": t_a,
             "ref_audio": t_a}
    unique_t = sorted({t_v, t_a})
    t_row = {t: i for i, t in enumerate(unique_t)}
    seg_tag = {"text": 1, "video": 0, "audio": 2}
    mod_segments = []
    for a0, b0, kind in layout.segments:
        mod_segments.append((a0, b0,
                             t_row[seg_t[kind]] * 3 + seg_tag[kind]))

    video_rows = patchify_video(v.to(torch.float32), model.patch_size)
    audio_rows = pack_audio(a.to(torch.float32))
    video_embed = model.video_patch_proj(video_rows).to(torch.bfloat16)
    audio_embed = model.audio_patch_proj(audio_rows).to(torch.bfloat16)
    text_states = model.token_refiner(
        model.condition_proj(c)[0], transformer_options={})
    h_t = torch.empty(layout.seq_len, 5376, dtype=torch.bfloat16,
                      device=dev)
    voff = aoff = 0
    for a0, b0, kind in layout.segments:
        n = b0 - a0
        if kind == "text":
            h_t[a0:b0] = text_states
        elif kind in ("cond", "ref_img", "video"):
            h_t[a0:b0] = video_embed[voff:voff + n]
            voff += n
        else:
            h_t[a0:b0] = audio_embed[aoff:aoff + n]
            aoff += n

    t_vals = torch.tensor(unique_t, dtype=torch.float32, device=dev)
    table = model.adaln_t_table.to(dev)
    pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
    t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
    rope_freqs = rope_rotation_table(
        model.rope_freqs(layout.position_ids, dev), torch.bfloat16)
    return h_t, t_emb, mod_segments, rope_freqs


def time_blocks(tag):
    """Time attn and mlp bodies separately for blocks 0..NUM_TIMED."""
    h_t, t_emb, mod_segments, rope_freqs = build_block_inputs()
    attn_ms, mlp_ms = [], []
    for i in range(NUM_TIMED):
        blk = model.blocks[i]
        xb = h_t.clone()
        st = torch.cuda.Event(True)
        en = torch.cuda.Event(True)
        with torch.no_grad():
            torch.cuda.synchronize()
            (shift_msa, scale_msa, gate_msa,
             shift_mlp, scale_mlp, gate_mlp) = blk.adaln_proj(t_emb)
            hh = blk.norm1(xb)
            for a0, b0, row in mod_segments:
                hh[a0:b0].mul_(1.0 + scale_msa[row].to(hh.dtype)) \
                          .add_(shift_msa[row].to(hh.dtype))
            st.record()
            att = blk.attn(hh, rope_freqs=rope_freqs)
            en.record()
            torch.cuda.synchronize()
            attn_ms.append(st.elapsed_time(en))
            for a0, b0, row in mod_segments:
                xb[a0:b0].addcmul_(att[a0:b0],
                                   gate_msa[row].to(xb.dtype))
            hh = blk.norm2(xb)
            for a0, b0, row in mod_segments:
                hh[a0:b0].mul_(1.0 + scale_mlp[row].to(hh.dtype)) \
                          .add_(shift_mlp[row].to(hh.dtype))
            st.record()
            y = blk.mlp(hh)
            en.record()
            torch.cuda.synchronize()
            mlp_ms.append(st.elapsed_time(en))
            for a0, b0, row in mod_segments:
                xb[a0:b0].addcmul_(y[a0:b0], gate_mlp[row].to(xb.dtype))
    mean = lambda xs: sum(xs) / len(xs)
    print(f"[{tag}] attn {mean(attn_ms):7.1f}ms/block  "
          f"mlp {mean(mlp_ms):6.1f}ms/block", flush=True)
    return mean(attn_ms), mean(mlp_ms)


sig0 = float(build_sigmas(STEPS)[0])

# ---- baseline per-block ----
attn0, mlp0 = time_blocks("base")

# ---- calibration: one real denoise step with recorder ----
print("[bench] calibrating on one denoise step", flush=True)
captures, ref_out = calibrate(model,
                              lambda: denoise_step([v, a], sig0))
print(f"[bench] captured {len(captures)} attention sites "
      f"(refiner 2 + dit 50)", flush=True)

# ---- bind ----
print("[bench] binding attention_core per site "
      f"({time.strftime('%H:%M:%S')})", flush=True)
cores, trail = bind(model, captures)
variants = {}
for name, _ in trail:
    variants[name] = variants.get(name, 0) + 1
print(f"[bench] bound: {variants} "
      f"({time.strftime('%H:%M:%S')})", flush=True)

# ---- free calibration captures (2.7GB) before the timed phases ----
del captures
gc.collect()
torch.cuda.empty_cache()

# ---- parity (seam installed) ----
print("[bench] parity check", flush=True)
maxd, meand, cos = parity(ref_out, lambda: denoise_step([v, a], sig0))
print(f"[bench] parity vs baseline: cosine={cos:.6f} "
      f"max|d|={maxd:.4f} mean|d|={meand:.5f}", flush=True)

# ---- seamed per-block ----
attn1, mlp1 = time_blocks("seam")

# ---- seamed e2e ----
print("[bench] seamed e2e 20 steps", flush=True)
out, times = h.run(warmup=1)
unbind(model)
total, per_step, rtf = h.rtf(times)
print(f"[bench] seamed e2e: total={total:.1f}s per_step="
      f"{per_step*1000:.0f}ms rtf={rtf:.3f}x", flush=True)

# ---- baseline e2e from the parallel background run ----
base_e2e = None
try:
    with open("/tmp/e2e_baseline.json") as fh:
        base_e2e = json.load(fh)
except FileNotFoundError:
    pass

print("\n===== MiniMax-H3 NVFP4: baseline vs flash_rt.seams =====")
print(f"per-block (blocks 0..{NUM_TIMED-1}, S={layout.seq_len}, "
      f"sigma={SIGMA_V}):")
print(f"  baseline attn {attn0:.1f}ms  mlp {mlp0:.1f}ms   "
      f"50-blk attn {attn0*50/1000:.2f}s mlp {mlp0*50/1000:.2f}s")
print(f"  seamed   attn {attn1:.1f}ms  mlp {mlp1:.1f}ms   "
      f"50-blk attn {attn1*50/1000:.2f}s mlp {mlp1*50/1000:.2f}s")
if attn1:
    print(f"  attn speedup {attn0/attn1:.2f}x   "
          f"block speedup {(attn0+mlp0)/(attn1+mlp1):.2f}x")
if base_e2e is not None:
    print(f"e2e 20 steps @480p39f: baseline {base_e2e['total']:.1f}s "
          f"({base_e2e['per_step']*1000:.0f}ms/step, "
          f"rtf {base_e2e['rtf']:.3f}x)  seamed {total:.1f}s "
          f"({per_step*1000:.0f}ms/step, rtf {rtf:.3f}x)")
    print(f"  e2e speedup {base_e2e['total']/total:.2f}x")
else:
    print(f"e2e 20 steps @480p39f: seamed {total:.1f}s "
          f"({per_step*1000:.0f}ms/step, rtf {rtf:.3f}x) "
          f"[baseline e2e not available]")
print(f"parity: cosine={cos:.6f} max|d|={maxd:.4f} mean|d|={meand:.5f}")
print(f"variants: {variants}")
