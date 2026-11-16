#!/usr/bin/env python
"""Render an episode .parquet (NormaCore schema) to an .mp4 with all cameras side by side.

If the episode has hand keypoints (`hand_*` columns), they are drawn on the overhead camera:
fingertips in orange, the other joints in cyan, and in red when that frame had no detection
(the values are then copied from the nearest frame that had one).

Usage: python parquet_to_mp4.py episode_0001.parquet [output.mp4]
"""
import io
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

FINGERTIPS = (4, 8, 12, 16, 20)

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".mp4"
t = pq.read_table(src)
hands = "hand_keypoints_2d" in t.column_names
if hands:
    kps = t.column("hand_keypoints_2d").to_pylist()
    cams = t.column("hand_camera").to_pylist()
    detected = t.column("hand_detected").to_pylist()
ts = t.column("timestamp_ns_since_episode_start").to_pylist()
fps = max(1.0, (len(ts) - 1) / (ts[-1] / 1e9)) if len(ts) > 1 and ts[-1] else 10.0

ff = None
for i, row in enumerate(t.column("images").to_pylist()):
    images = [Image.open(io.BytesIO(im["jpeg"])).convert("RGB") for im in row]
    if hands:
        draw = ImageDraw.Draw(images[cams[i]])
        w, h = images[cams[i]].size
        for k in range(21):
            x, y = kps[i][2 * k] * w, kps[i][2 * k + 1] * h
            tip = k in FINGERTIPS
            color = (255, 0, 0) if not detected[i] else (255, 170, 0) if tip else (0, 230, 255)
            r = 2 if tip else 1
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    frame = np.hstack([np.asarray(im) for im in images])
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
print(f"{dst}: {len(ts)} frames, {fps:.1f} fps, {ts[-1] / 1e9:.1f} s"
      + (f", hand detected in {sum(detected)}/{len(ts)} frames" if hands else ""))
