"""Turn the recorded physical-contact episodes (film_phys.py --record) into Parquet with NormaCore's episode columns,
WITH the two camera pictures of every frame.

    ~/pq/bin/python ~/flexnsense_sim/rl/phys_to_parquet.py ~/flexnsense-sim/runs/phys/record/episodes ~/flexnsense-sim/deliver_phys/data

Writes two files with the same episodes and the same pictures, one per arm (an episode in NormaCore's format holds one
arm's eight motors):
  phys_move_arm.parquet   the arm that takes the scale and changes the finger (the trained policy)
  phys_pull_arm.parquet   the arm that pulls to 200 g and returns (a fixed routine)
Columns: episode_start_ns, global_frame_id, timestamp_ns_since_episode_start, joints, images, task, plus
sim_force_n (what the scale's spring reads) and sim_stage (0 to index, 1 pull, 2 to middle, 3 pull).
images = [overhead camera, camera on the move arm], JPEG, 640 x 540.

Synthetic, because there is no real servo: position, goal, range_min, range_max are ticks computed from the simulated
joint angle (4096 per turn, 2048 at the URDF's zero); the eighth motor is the gripper, taken from its jaw's travel;
current_ma and velocity are 0.
"""

import glob
import math
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TASK = {
    "move": "take the spring scale from the table and put its loop on the index finger; after the pull, move the loop "
            "to the middle finger; then go back to rest",
    "pull": "hold the spring scale; when its loop is on a finger, pull until it shows 200 grams, then return to zero",
}
TICKS_PER_RAD = 4096 / (2 * math.pi)


def main(src: str, dst: str):
    files = sorted(glob.glob(os.path.join(src, "ep_*.npz")))
    os.makedirs(dst, exist_ok=True)
    joint_t = pa.struct([("range_min", pa.uint32()), ("range_max", pa.uint32()), ("position", pa.uint32()),
                         ("position_norm", pa.float32()), ("goal", pa.uint32()), ("goal_norm", pa.float32()),
                         ("current_ma", pa.uint32()), ("velocity", pa.uint32())])
    image_t = pa.struct([("jpeg", pa.binary())])
    schema = pa.schema([("episode_start_ns", pa.uint64()), ("global_frame_id", pa.binary()),
                        ("timestamp_ns_since_episode_start", pa.uint64()), ("joints", pa.list_(joint_t)),
                        ("images", pa.list_(image_t)), ("task", pa.string()), ("sim_force_n", pa.float32()),
                        ("sim_stage", pa.int8())])
    base_ns = 1_794_800_000_000_000_000
    for arm in ("move", "pull"):
        out = os.path.join(dst, f"phys_{arm}_arm.parquet")
        writer, frame, rows_total = pq.ParquetWriter(out, schema, compression="zstd"), 0, 0
        for e, f in enumerate(files):
            d = np.load(f, allow_pickle=True)
            pos, goal, lo, hi, dt = d[f"pos_{arm}"], d[f"goal_{arm}"], d[f"lo_{arm}"], d[f"hi_{arm}"], float(d["dt"])
            span = np.where(hi - lo == 0, 1.0, hi - lo)
            norm = lambda x: np.clip((x - lo) / span, 0.0, 1.0).astype(np.float32)
            # the gripper (last column) is a jaw's travel in metres: its ticks are its share of the range
            tick = lambda x: np.concatenate([np.clip(np.rint(2048 + x[..., :7] * TICKS_PER_RAD), 0, 4095),
                                             np.rint(norm(x)[..., 7:] * 2000 + 1000)], axis=-1).astype(np.uint32)
            p_t, g_t, p_n, g_n = tick(pos), tick(goal), norm(pos), norm(goal)
            r_min, r_max = tick(lo[None])[0], tick(hi[None])[0]
            n_t = len(pos)
            cols = {
                "episode_start_ns": [base_ns + e * 60_000_000_000] * n_t,
                "global_frame_id": [int(frame + t).to_bytes(8, "little") for t in range(n_t)],
                "timestamp_ns_since_episode_start": [int(round(t * dt * 1e9)) for t in range(n_t)],
                "joints": [[{"range_min": int(r_min[m]), "range_max": int(r_max[m]), "position": int(p_t[t, m]),
                             "position_norm": float(p_n[t, m]), "goal": int(g_t[t, m]), "goal_norm": float(g_n[t, m]),
                             "current_ma": 0, "velocity": 0} for m in range(8)] for t in range(n_t)],
                "images": [[{"jpeg": bytes(d["top"][t])}, {"jpeg": bytes(d["wrist"][t])}] for t in range(n_t)],
                "task": [TASK[arm]] * n_t,
                "sim_force_n": d["force"].astype(np.float32).tolist(),
                "sim_stage": d["stage"].astype(np.int8).tolist(),
            }
            writer.write_table(pa.table(cols, schema=schema))
            frame += n_t
            rows_total += n_t
        writer.close()
        print(f"wrote {out}: {len(files)} episodes, {rows_total} rows, 2 pictures a row, {os.path.getsize(out) / 1e6:.0f} MB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
