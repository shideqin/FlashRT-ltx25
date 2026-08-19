"""Frame-level parity between two LTX-2.5 output videos (decoded from mp4).

Usage: python serve/ltx25_parity.py A.mp4 B.mp4
"""

import sys

import av
import numpy as np


def load_frames(path):
    c = av.open(path)
    v = c.streams.video[0]
    frames = []
    for frame in c.decode(v):
        arr = frame.to_ndarray(format="rgb24").astype(np.float32)
        frames.append(arr)
    c.close()
    return np.stack(frames)


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    a = load_frames(a_path)
    b = load_frames(b_path)
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    n = a.shape[0]
    per = []
    for i in range(n):
        fa, fb = a[i], b[i]
        cos = float(np.dot(fa.ravel(), fb.ravel()) /
                    (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-12))
        maxd = float(np.abs(fa - fb).max())
        meand = float(np.abs(fa - fb).mean())
        per.append((cos, maxd, meand))
    cos_all = [p[0] for p in per]
    print(f"frames: {n}")
    print(f"per-frame cosine: min {min(cos_all):.6f} "
          f"max {max(cos_all):.6f} mean {np.mean(cos_all):.6f}")
    print(f"per-frame max|d|: min {min(p[1] for p in per):.3f} "
          f"max {max(p[1] for p in per):.3f}")
    print(f"per-frame mean|d|: min {min(p[2] for p in per):.4f} "
          f"max {max(p[2] for p in per):.4f}")
    print("first frame diff:", f"cos {per[0][0]:.6f} "
          f"max|d| {per[0][1]:.3f} mean|d| {per[0][2]:.4f}")
    print("last frame diff:", f"cos {per[-1][0]:.6f} "
          f"max|d| {per[-1][1]:.3f} mean|d| {per[-1][2]:.4f}")


if __name__ == "__main__":
    main()
