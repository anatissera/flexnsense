"""Turn recorded simulation episodes (.npz from record_rollouts.py) into Parquet with NormaCore's episode columns.

    ~/pq/bin/python ~/flexnsense_sim/rl/rollouts_to_parquet.py ~/flexnsense-sim/dataset/pull.npz ~/flexnsense-sim/hf_dataset/data/pull.parquet

Columns, one row per frame, as in NormaCore's dataset export (docs/datasets/export-parquet in their repository):
  episode_start_ns, global_frame_id, timestamp_ns_since_episode_start, joints (per motor: range_min, range_max,
  position, position_norm, goal, goal_norm, current_ma, velocity), images, task
plus the simulator's own signals as extra columns (sim_*).

What is synthetic here, because there is no real servo: "position", "goal", "range_min" and "range_max" are servo
ticks computed from the simulated joint angle (4096 ticks per turn, 2048 at the URDF's zero angle); "current_ma" and
"velocity" are 0; "images" is an empty list (no camera was rendered).
"""

import math
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TASK_TEXT = {
    "pull": "hold the spring scale and give way when the finger pulls harder than 200 grams",
    "move": "move the spring scale's loop from one finger to the next finger",
    "team_pull": "hold the spring scale and give way when the finger pulls harder than 200 grams, while the other arm moves the scale between fingers",
    "team_move": "after the pull ends, take the spring scale, lift its loop over the finger, put it on the other finger and go back to rest",
}
TICKS_PER_RAD = 4096 / (2 * math.pi)


def main(src: str, dst: str):
    d = np.load(src, allow_pickle=False)
    task = str(d["task"])
    pos, goal, valid, extra = d["pos"], d["goal"], d["valid"], d["extra"]
    lo, hi, dt = d["lo"], d["hi"], float(d["dt"])
    names = [str(n) for n in d["extra_names"]]
    n_ep, n_t, n_m = pos.shape
    tick = lambda rad: np.clip(np.rint(2048 + rad * TICKS_PER_RAD), 0, 4095).astype(np.uint32)
    norm = lambda rad: np.clip((rad - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    r_min, r_max = tick(lo), tick(hi)
    joint_t = pa.struct([("range_min", pa.uint32()), ("range_max", pa.uint32()), ("position", pa.uint32()),
                         ("position_norm", pa.float32()), ("goal", pa.uint32()), ("goal_norm", pa.float32()),
                         ("current_ma", pa.uint32()), ("velocity", pa.uint32())])
    image_t = pa.struct([("jpeg", pa.binary())])
    base_ns = 1_794_700_000_000_000_000  # an arbitrary wall-clock origin in November 2026
    rows = {k: [] for k in ("episode_start_ns", "global_frame_id", "timestamp_ns_since_episode_start", "joints", "images", "task")}
    extras = {f"sim_{n}": [] for n in names}
    frame = 0
    for e in range(n_ep):
        start = base_ns + e * 60_000_000_000
        p_t, g_t, p_n, g_n = tick(pos[e]), tick(goal[e]), norm(pos[e]), norm(goal[e])
        for t in range(n_t):
            if not valid[e, t]:
                break
            rows["episode_start_ns"].append(start)
            rows["global_frame_id"].append(int(frame).to_bytes(8, "little"))
            rows["timestamp_ns_since_episode_start"].append(int(round(t * dt * 1e9)))
            rows["joints"].append([
                {"range_min": int(r_min[m]), "range_max": int(r_max[m]), "position": int(p_t[t, m]),
                 "position_norm": float(p_n[t, m]), "goal": int(g_t[t, m]), "goal_norm": float(g_n[t, m]),
                 "current_ma": 0, "velocity": 0} for m in range(n_m)])
            rows["images"].append([])
            rows["task"].append(TASK_TEXT[task])
            for k, name in enumerate(names):
                extras[f"sim_{name}"].append(float(extra[e, t, k]))
            frame += 1
    table = pa.table({
        "episode_start_ns": pa.array(rows["episode_start_ns"], pa.uint64()),
        "global_frame_id": pa.array(rows["global_frame_id"], pa.binary()),
        "timestamp_ns_since_episode_start": pa.array(rows["timestamp_ns_since_episode_start"], pa.uint64()),
        "joints": pa.array(rows["joints"], pa.list_(joint_t)),
        "images": pa.array(rows["images"], pa.list_(image_t)),
        "task": pa.array(rows["task"], pa.string()),
        **{k: pa.array(v, pa.float32()) for k, v in extras.items()},
    })
    pq.write_table(table, dst, compression="zstd")
    print(f"wrote {dst}: {n_ep} episodes, {table.num_rows} rows, {n_m} motors, {1 / dt:.0f} rows a second")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
