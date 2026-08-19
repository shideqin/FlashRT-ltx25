"""Shared mp4 compare helper used by serve/* harnesses."""

import av
import numpy as np


def load_frames(path):
    c = av.open(path)
    v = c.streams.video[0]
    frames = []
    for frame in c.decode(v):
        frames.append(frame.to_ndarray(format="rgb24").astype(np.float32))
    c.close()
    return np.stack(frames)


def compare_mp4(a_path, b_path):
    a = load_frames(a_path)
    b = load_frames(b_path)
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    fa = a.astype(np.float64).reshape(a.shape[0], -1)
    fb = b.astype(np.float64).reshape(b.shape[0], -1)
    cos = float(np.sum(fa * fb) / (
        np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-12))
    md = float(np.abs(a - b).max())
    return cos, md


if __name__ == "__main__":
    import sys
    c, m = compare_mp4(sys.argv[1], sys.argv[2])
    print(f"cosine {c:.9f} max|d| {m:.6f}")
