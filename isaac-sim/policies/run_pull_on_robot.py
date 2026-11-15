"""Run the simulated pull-arm policy on the real ElRobot follower. REFERENCE CODE, NEVER RUN ON HARDWARE YET.

It was written from the simulation side on 2026-11-15 and has not moved a real arm. By default it is a DRY RUN: it
reads the arm and the scale, computes what the policy would command, prints it, and sends nothing. Read
policies/README.md ("Putting the pull policy on the robot") before using --send.

Run from the repository root, with Station running and the gauge reader serving the force:

    .venv/bin/python -m gauge_reader.server                      # terminal 1: the scale reading on ws://localhost:8765/force
    .venv/bin/python isaac-sim/policies/run_pull_on_robot.py \
        --policy isaac-sim/policies/pull_v1/policy.onnx \
        --joint_map isaac-sim/policies/joint_map.json          # dry run
    ... --send                                                    # really command the arm (person at the keyboard!)

Needs: onnxruntime, numpy, websockets (pip install onnxruntime websockets).

What it feeds the policy (29 numbers, the order used in training, sim/rl/flex_tasks/pull_env.py):
   0..6   joint angle of motors 1..7 minus the start posture [rad]
   7..13  joint speed of motors 1..7 [rad/s] times 0.1
  14..18  target minus angle for motors 1, 2, 3, 4, 6 [rad]
  19      force one step ago / 1.96 N
  20      (force now - force one step ago) / 1.96 N
  21..23  gripper point minus its starting place, along the scale / across it / up [m], times 10
  24..28  the previous action
What it gets back (5 numbers in -1..1): change of the targets of motors 1, 2, 3, 4, 6, times 0.012 rad per step.
The loop runs 60 times a second, as in training.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "lib" / "station.py").exists())  # the repository root, wherever this folder sits
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent / "tools"))  # seed_pose.py (forward kinematics)

CAP_N = 1.96  # the force limit the policy was trained with (200 gf)
ACTION_SCALE = 0.012  # rad per step
STEP_S = 1.0 / 60.0
TIP = 0.10  # m: gripper origin -> the point between the jaws (same number as in training)
DRIVEN = [1, 2, 3, 4, 6]  # motors the policy moves
START = {1: 0.0, 2: 0.04, 3: 1.057, 4: 0.072, 5: 0.0, 6: -1.169, 7: 0.0}  # rad: the start posture used in training
MAX_STEP_RAD = 0.012  # never ask a joint to move more than this per step
MAX_FROM_START_RAD = 0.9  # stop if any joint target strays further than this from the start posture
FORCE_STALE_S = 0.25  # no fresh scale reading for this long -> release and stop


def load_joint_map(path: str) -> dict[int, dict]:
    """{motor id: {"zero_tick": tick at URDF angle 0, "ticks_per_rad": signed}} measured on YOUR arm (see README)."""
    raw = json.loads(Path(path).read_text())
    return {int(k): v for k, v in raw.items()}


def to_rad(tick: int, m: dict) -> float:
    return (tick - m["zero_tick"]) / m["ticks_per_rad"]


def to_tick(rad: float, m: dict) -> int:
    return int(round(m["zero_tick"] + rad * m["ticks_per_rad"]))


async def read_force(state: dict, url: str):
    """Keep state["force"], state["force_t"] fresh from the gauge reader. A missing reading is None, never a guess."""
    import websockets

    async with websockets.connect(url) as ws:
        async for msg in ws:
            data = json.loads(msg)
            ok = data.get("found") and data.get("newtons") is not None
            state["force"] = float(data["newtons"]) if ok else None
            state["force_t"] = time.monotonic()


async def read_joints(client, bus: str, state: dict):
    from lib.station import follow_motor_state

    async for bus_serial, motor_id, ms in follow_motor_state(client):
        if bus_serial == bus and ms.position is not None:
            state["ticks"][motor_id] = ms.position
            state["ranges"][motor_id] = (ms.range_min, ms.range_max)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="policy.onnx exported by sim/rl/eval_pull.py")
    ap.add_argument("--joint_map", required=True, help="json with zero_tick and ticks_per_rad per motor (measure it first)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--bus", default=None, help="follower bus serial (default: FOLLOWER_BUS_SERIAL from .env)")
    ap.add_argument("--force_url", default="ws://localhost:8765/force")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--send", action="store_true", help="really command the arm; without it nothing is sent")
    args = ap.parse_args()

    import onnxruntime as ort
    from seed_pose import fk  # forward kinematics from NormaCore's URDF (tools/seed_pose.py)

    from lib.env import env_get
    from lib.station import connect, set_position

    jmap = load_joint_map(args.joint_map)
    bus = args.bus or env_get("FOLLOWER_BUS_SERIAL")
    sess = ort.InferenceSession(args.policy)
    in_name = sess.get_inputs()[0].name
    state = {"ticks": {}, "ranges": {}, "force": None, "force_t": 0.0}
    client = await connect(args.host)
    tasks = [asyncio.create_task(read_joints(client, bus, state)), asyncio.create_task(read_force(state, args.force_url))]
    while len(state["ticks"]) < 7:
        await asyncio.sleep(0.05)

    def angles() -> np.ndarray:
        return np.array([to_rad(state["ticks"][m], jmap[m]) for m in range(1, 8)])

    def tip_and_dir(q7: np.ndarray):
        pos, rot_m = fk(q7)  # gripper origin and orientation from the seven joint angles
        return pos + rot_m[:, 1] * TIP  # the gripper points along its own +y

    q0 = angles()
    home = tip_and_dir(q0)
    u = home.copy(); u[2] = 0.0; u /= np.linalg.norm(u)  # along the scale: from the base towards the gripper, level
    side = np.cross([0.0, 0.0, 1.0], u)
    start = np.array([START[m] for m in range(1, 8)])
    target = q0.copy()
    prev_q, prev_force, action = q0.copy(), 0.0, np.zeros(5, dtype=np.float32)
    drive_idx = [m - 1 for m in DRIVEN]
    t_end = time.monotonic() + args.seconds
    try:
        while time.monotonic() < t_end:
            t0 = time.monotonic()
            force = state["force"]
            if force is None or t0 - state["force_t"] > FORCE_STALE_S:
                print("no fresh force reading: stopping (the arm keeps its last target)")
                break
            q = angles()
            off = tip_and_dir(q) - home
            obs = np.concatenate([
                q - start, 0.1 * (q - prev_q) / STEP_S, (target - q)[drive_idx],
                [prev_force / CAP_N, (force - prev_force) / CAP_N],
                10.0 * np.array([off @ u, off @ side, off[2]]), action,
            ]).astype(np.float32)[None]
            action = np.clip(sess.run(None, {in_name: obs})[0][0], -1.0, 1.0).astype(np.float32)
            step = np.clip(ACTION_SCALE * action, -MAX_STEP_RAD, MAX_STEP_RAD)
            target[drive_idx] += step
            if np.abs(target - q0).max() > MAX_FROM_START_RAD:
                print("a joint target strayed too far from the start posture: stopping")
                break
            for k, m in zip(drive_idx, DRIVEN):
                tick = to_tick(float(target[k]), jmap[m])
                lo, hi = state["ranges"][m]
                tick = min(max(tick, lo), hi) if hi > lo else tick
                if args.send:
                    await set_position(client, bus, m, tick)
            print(f"force {force:5.2f} N | along {off @ u * 1000:6.1f} mm | action {np.round(action, 2)} | {'SENT' if args.send else 'dry run'}")
            prev_q, prev_force = q, force
            await asyncio.sleep(max(0.0, STEP_S - (time.monotonic() - t0)))
    finally:
        for t in tasks:
            t.cancel()


if __name__ == "__main__":
    asyncio.run(main())
