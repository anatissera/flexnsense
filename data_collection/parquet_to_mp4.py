#!/usr/bin/env python
"""Render an episode .parquet (NormaCore schema) to an .mp4 with all cameras side by side.

Usage: python parquet_to_mp4.py episode_0001.parquet [output.mp4]
"""
import io
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".mp4"
t = pq.read_table(src, columns=["timestamp_ns_since_episode_start", "images"])
ts = t.column("timestamp_ns_since_episode_start").to_pylist()
fps = max(1.0, (len(ts) - 1) / (ts[-1] / 1e9)) if len(ts) > 1 and ts[-1] else 10.0

ff = None
for row in t.column("images").to_pylist():
    frame = np.hstack([np.asarray(Image.open(io.BytesIO(im["jpeg"])).convert("RGB")) for im in row])
    if ff is None:
        h, w = frame.shape[:2]
        ff = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-",
             "-vf", "scale=iw*2:ih*2:flags=neighbor", "-pix_fmt", "yuv420p", dst],
            stdin=subprocess.PIPE,
        )
    ff.stdin.write(frame.tobytes())
ff.stdin.close()
ff.wait()
print(f"{dst}: {len(ts)} frames, {fps:.1f} fps, {ts[-1] / 1e9:.1f} s")
