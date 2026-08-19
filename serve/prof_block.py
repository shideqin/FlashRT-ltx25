"""Profile per-block attn/mlp cost on the loaded host."""
import sys, time
import torch

sys.path.insert(0, '/data/ComfyUI')
from minimax_host import MiniMaxHost
from comfy.ldm.minimax.model import (PackedLayout, rope_rotation_table,
                                     patchify_video, pack_audio,
                                     _frame_grid)

h = MiniMaxHost(frames=22, text_len=256, steps=2)
h.load()
model = h.model
dev = next(model.parameters()).device
v, a, c = h.inputs()
x = [v.to(dev), a.to(dev)]

# ---- preamble (mirror of _forward up to the block loop) ----
latent_t, lat_h, lat_w = v.shape[2], v.shape[3], v.shape[4]
audio_t = a.shape[-1]
text_len = c.shape[1]
layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t)
payload = h.payload()
sigma_v = 0.5  # mid-schedule t ~ 0.5
t_v = float(1.0 - sigma_v)
t_a = float(1.0 - 0.5)  # simplified: audio t same here
seg_t = {"text": t_v, "video": t_v, "audio": t_a,
         "cond": t_v, "ref_img": t_v, "cond_audio": t_a, "ref_audio": t_a}
unique_t = sorted({t_v, t_a})
t_row = {t: i for i, t in enumerate(unique_t)}
seg_tag = {"text": 1, "video": 0, "audio": 2}
mod_segments = []
for a0, b0, kind in layout.segments:
    mod_segments.append((a0, b0, t_row[seg_t[kind]] * 3 + seg_tag[kind]))

video_rows = patchify_video(v.to(torch.float32), model.patch_size)
audio_rows = pack_audio(a.to(torch.float32))
video_embed = model.video_patch_proj(video_rows).to(torch.bfloat16)
audio_embed = model.audio_patch_proj(audio_rows).to(torch.bfloat16)
text_states = model.token_refiner(model.condition_proj(c.to(dev))[0], transformer_options={})
h_t = torch.empty(layout.seq_len, 5376, dtype=torch.bfloat16, device=dev)
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
rope_freqs = rope_rotation_table(model.rope_freqs(layout.position_ids, dev), torch.bfloat16)

blk = model.blocks[0]
with torch.no_grad():
    for _ in range(2):
        out = blk(h_t, t_emb, mod_segments, rope_freqs)
torch.cuda.synchronize()

# time attn and mlp separately across blocks 0..9
attn_ms, mlp_ms, other_ms = [], [], []
for i in range(10):
    blk = model.blocks[i]
    xb = h_t.clone()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    with torch.no_grad():
        torch.cuda.synchronize()
        st.record()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = blk.adaln_proj(t_emb)
        hh = blk.norm1(xb)
        for a0, b0, row in mod_segments:
            hh[a0:b0].mul_(1.0 + scale_msa[row].to(hh.dtype)).add_(shift_msa[row].to(hh.dtype))
        att = blk.attn(hh, rope_freqs=rope_freqs)
        for a0, b0, row in mod_segments:
            xb[a0:b0].addcmul_(att[a0:b0], gate_msa[row].to(xb.dtype))
        en.record(); torch.cuda.synchronize()
        attn_ms.append(st.elapsed_time(en))
        st.record()
        hh = blk.norm2(xb)
        for a0, b0, row in mod_segments:
            hh[a0:b0].mul_(1.0 + scale_mlp[row].to(hh.dtype)).add_(shift_mlp[row].to(hh.dtype))
        y = blk.mlp(hh)
        for a0, b0, row in mod_segments:
            xb[a0:b0].addcmul_(y[a0:b0], gate_mlp[row].to(xb.dtype))
        en.record(); torch.cuda.synchronize()
        mlp_ms.append(st.elapsed_time(en))

print(f"attn: {sum(attn_ms)/10:.1f}ms/block  mlp: {sum(mlp_ms)/10:.1f}ms/block")
print(f"projected 50 blocks: attn {sum(attn_ms)/10*50/1000:.2f}s  mlp {sum(mlp_ms)/10*50/1000:.2f}s")

# what does optimized_attention use?
import comfy.ldm.modules.attention as attn_mod
print("optimized_attention:", attn_mod.optimized_attention.__module__, attn_mod.optimized_attention.__name__)
