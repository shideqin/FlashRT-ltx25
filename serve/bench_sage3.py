"""SageAttention3 arm: 52 sites bound to sage3, end-to-end 1-step parity
+ per-block timing + e2e 20 steps, vs the host baseline."""
import os
import sys
import types
import time

os.environ["LOCAL_KERNELS"] = open("/tmp/lk.env").read().strip()
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, '/data/ComfyUI')
sys.path.insert(0, '/workspace/data/serve')

import torch
import torch.nn.functional as F

from minimax_host import MiniMaxHost, build_sigmas
from seam_attn import calibrate, _seamed_forward
from sage3_attn import Sage3Core
from comfy.ldm.minimax.model import (PackedLayout, rope_rotation_table,
                                     patchify_video, pack_audio)

FRAMES, TEXT_LEN, STEPS = 39, 256, 20

h = MiniMaxHost(frames=FRAMES, text_len=TEXT_LEN, steps=STEPS)
h.load()
model = h.model
dev = next(model.parameters()).device
v, a, c = h.inputs()
v, a, c = v.to(dev), a.to(dev), c.to(dev)
payload = h.payload()
sig0 = float(build_sigmas(STEPS)[0])

def denoise_step(x, sigma):
    with torch.no_grad():
        out = model(x, sigma * torch.ones([1], device=dev), c,
                    transformer_options={}, minimax_payload=payload)
    torch.cuda.synchronize()
    return out

# ---- baseline reference + captures ----
captures, ref_out = calibrate(model, lambda: denoise_step([v, a], sig0))
print(f"[sage3] captured {len(captures)} sites", flush=True)

# ---- bind sage3 to all sites ----
sites = list(model.token_refiner.blocks) + list(model.blocks)
cores = []
for site, cap in zip(sites, captures):
    core = Sage3Core(tuple(cap["q"].shape))
    attn = site.attn
    attn.flash_core = core
    attn.forward = types.MethodType(_seamed_forward, attn)
    cores.append(core)
print(f"[sage3] bound {len(cores)} sites", flush=True)

# ---- 1-step parity ----
out = denoise_step([v, a], sig0)
ra, rb = ref_out[0], out[0]
cos = float(F.cosine_similarity(ra.reshape(-1), rb.reshape(-1), dim=0))
print(f"[sage3] 1-step parity vs baseline: cosine={cos:.6f} "
      f"max|d|={float((ra-rb).abs().max()):.4f} "
      f"mean|d|={float((ra-rb).abs().mean()):.5f}", flush=True)

# ---- per-block timing ----
layout = PackedLayout(TEXT_LEN, v.shape[2], v.shape[3], v.shape[4], a.shape[-1])
sigma_v = 0.5
t_v = float(1.0 - sigma_v); t_a = float(1.0 - 0.5)
seg_t = {"text": t_v, "video": t_v, "audio": t_a, "cond": t_v, "ref_img": t_v,
         "cond_audio": t_a, "ref_audio": t_a}
unique_t = sorted({t_v, t_a}); t_row = {t: i for i, t in enumerate(unique_t)}
seg_tag = {"text": 1, "video": 0, "audio": 2}
mod_segments = [(x0, y0, t_row[seg_t[k]] * 3 + seg_tag[k])
                for x0, y0, k in layout.segments]
video_rows = patchify_video(v.to(torch.float32), model.patch_size)
audio_rows = pack_audio(a.to(torch.float32))
video_embed = model.video_patch_proj(video_rows).to(torch.bfloat16)
audio_embed = model.audio_patch_proj(audio_rows).to(torch.bfloat16)
text_states = model.token_refiner(model.condition_proj(c)[0],
                                  transformer_options={})
h_t = torch.empty(layout.seq_len, 5376, dtype=torch.bfloat16, device=dev)
voff = aoff = 0
for x0, y0, kind in layout.segments:
    n = y0 - x0
    if kind == "text": h_t[x0:y0] = text_states
    elif kind in ("cond", "ref_img", "video"):
        h_t[x0:y0] = video_embed[voff:voff+n]; voff += n
    else:
        h_t[x0:y0] = audio_embed[aoff:aoff+n]; aoff += n
t_vals = torch.tensor(unique_t, dtype=torch.float32, device=dev)
table = model.adaln_t_table.to(dev)
pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
t_emb = torch.lerp(table[i0], table[i0+1], (pos - i0).unsqueeze(1))
rope_freqs = rope_rotation_table(model.rope_freqs(layout.position_ids, dev),
                                 torch.bfloat16)
attn_ms = []
for i in range(10):
    blk = model.blocks[i]
    xb = h_t.clone()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    with torch.no_grad():
        sm, sc, gm, _, _, _ = blk.adaln_proj(t_emb)
        hh = blk.norm1(xb)
        for x0, y0, row in mod_segments:
            hh[x0:y0].mul_(1.0 + sc[row].to(hh.dtype)).add_(sm[row].to(hh.dtype))
        torch.cuda.synchronize()
        st.record()
        att = blk.attn(hh, rope_freqs=rope_freqs)
        en.record(); torch.cuda.synchronize()
        attn_ms.append(st.elapsed_time(en))
print(f"[sage3] attn {sum(attn_ms)/10:.1f}ms/block  "
      f"(host 853.0ms)", flush=True)

# ---- e2e: pure sage3 then sage3+TeaCache, same seed ----
print(f"[sage3] e2e 20 steps, pure ({time.strftime('%H:%M:%S')})", flush=True)
out0, times0 = h.run(warmup=1)
tot0, per0, rtf0 = h.rtf(times0)
print(f"[sage3] e2e pure: total={tot0:.1f}s per_step={per0*1000:.0f}ms "
      f"rtf={rtf0:.3f}x (host 918.8s / 45942ms / 565x)", flush=True)

SKIP = [3, 5, 7, 9, 11, 13]  # 6 of 20 steps cached (TeaCache)
print(f"[sage3] e2e 20 steps +TeaCache skip={SKIP} "
      f"({time.strftime('%H:%M:%S')})", flush=True)
out1, times1 = h.run(warmup=1, skip_steps=SKIP)
tot1, per1, rtf1 = h.rtf(times1)
n_computed = sum(1 for t in times1 if t > 0.05)
print(f"[sage3] e2e +TeaCache: total={tot1:.1f}s per_step={per1*1000:.0f}ms "
      f"rtf={rtf1:.3f}x (computed {n_computed}/20 steps)", flush=True)

# TeaCache effect on output vs pure sage3 (same seed)
dv = (out1[0] - out0[0]).abs()
cos_tc = float(F.cosine_similarity(out1[0].reshape(-1), out0[0].reshape(-1),
                                   dim=0))
print(f"[sage3] TeaCache output vs pure sage3: cosine={cos_tc:.6f} "
      f"max|d|={float(dv.max()):.4f} mean|d|={float(dv.mean()):.5f}",
      flush=True)
import json
json.dump({"pure": {"total": tot0, "per_step": per0, "rtf": rtf0},
           "teacache": {"total": tot1, "per_step": per1, "rtf": rtf1,
                        "skip": SKIP},
           "parity_cos": cos, "attn_ms": sum(attn_ms)/10,
           "teacache_vs_pure_cos": cos_tc},
          open("/tmp/sage3_e2e.json", "w"))
