"""Run the simulated pull-arm policy on the real ElRobot follower. REFERENCE CODE, NEVER RUN ON HARDWARE YET.

It was written from the simulation side on 2026-11-15 and has not moved a real arm. By default it is a DRY RUN: it
reads the arm and the scale, computes what the policy would command, prints it, and sends nothing. Read
policies/README.md ("How to put the pull policy on the robot") before using --send.

Run from the repository root, with Station running. The script reads the scale itself, from the scale's camera
(the same reader as `python -m gauge_reader run`), so no other terminal is needed:

    .venv/bin/python -m gauge_reader cameras                     # which number or name is the scale's camera
    .venv/bin/python isaac-sim/policies/run_pull_on_robot.py \
        --policy isaac-sim/policies/pull_v1/policy.onnx \
        --joint_map isaac-sim/policies/joint_map.json \
        --camera 0                                               # dry run; --camera takes a number or part of a name
    ... --calib config/scale_calib.json                          # the scale's calibration (this is the default)
    ... --send                                                    # really command the arm (person at the keyboard!)

Needs: onnxruntime, numpy (pip install onnxruntime), and what gauge_reader needs (requirements.txt).

It stops with a message, and sends nothing, when the calibration file is missing, the camera does not open, the
scale gives no reading within 15 s, or one of motors 1 to 7 does not report within 10 s.

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
for _p in (HERE.parent / "tools", HERE, REPO):  # seed_pose.py (forward kinematics), sim_pull_runner.py, lib and gauge_reader
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CAP_N = 1.96  # the force limit the policy was trained with (200 gf)
ACTION_SCALE = 0.012  # rad per step
STEP_S = 1.0 / 60.0
TIP = 0.10  # m: gripper origin -> the point between the jaws (same number as in training)
DRIVEN = [1, 2, 3, 4, 6]  # motors the policy moves
START = {1: 0.0, 2: 0.04, 3: 1.057, 4: 0.072, 5: 0.0, 6: -1.169, 7: 0.0}  # rad: the start posture used in training
MAX_STEP_RAD = 0.012  # never ask a joint to move more than this per step
MAX_FROM_START_RAD = 0.9  # stop if any joint target strays further than this from the start posture
FORCE_STALE_S = 0.25  # no fresh scale reading for this long -> release and stop
ARM_MOTORS = tuple(range(1, 8))  # the seven arm joints; motor 8 (the gripper) also reports but is not used here
JOINTS_WAIT_S = 10.0  # how long to wait for every arm motor to report its position
FORCE_WAIT_S = 15.0  # how long to wait for the first reading of the scale
DEFAULT_CALIB = "config/scale_calib.json"


def load_joint_map(path: str) -> dict[int, dict]:
    """{motor id: {"zero_tick": tick at URDF angle 0, "ticks_per_rad": signed}} measured on YOUR arm (see README)."""
    raw = json.loads(Path(path).read_text())
    return {int(k): v for k, v in raw.items()}


def to_rad(tick: int, m: dict) -> float:
    return (tick - m["zero_tick"]) / m["ticks_per_rad"]


def to_tick(rad: float, m: dict) -> int:
    return int(round(m["zero_tick"] + rad * m["ticks_per_rad"]))


def missing_motors(ticks: dict, needed=ARM_MOTORS) -> list[int]:
    """The motors of `needed` that have not reported a position yet. The count of entries says nothing: the
    gripper (motor 8) reports too, so seven entries can be there with an arm motor still missing."""
    return [m for m in needed if m not in ticks]


def joints_timeout_message(ticks: dict, bus: str | None, waited_s: float = JOINTS_WAIT_S, needed=ARM_MOTORS) -> str:
    seen = sorted(ticks)
    return (
        f"error: after {waited_s:g} s not every arm motor has reported on bus {bus!r}.\n"
        f"  seen:    {seen if seen else 'none'}\n"
        f"  missing: {missing_motors(ticks, needed)}\n"
        + ("  No motor at all: is Station running, and is --bus (or FOLLOWER_BUS_SERIAL in .env) the follower's serial?"
           if not seen else
           "  A missing motor is not answering on the bus: check its cable and its id (Station shows the motors it sees).")
        + "\nNothing was sent to the arm."
    )


async def wait_for_joints(state: dict, timeout_s: float = JOINTS_WAIT_S, needed=ARM_MOTORS,
                          clock=time.monotonic, sleep=asyncio.sleep, poll_s: float = 0.05) -> list[int]:
    """Wait until every motor of `needed` is in state["ticks"]. Returns the motors still missing: [] when all
    reported, the missing ones when `timeout_s` passed first."""
    t_end = clock() + timeout_s
    while True:
        missing = missing_motors(state["ticks"], needed)
        if not missing or clock() >= t_end:
            return missing
        await sleep(poll_s)


async def wait_for_force(force, timeout_s: float = FORCE_WAIT_S, clock=time.monotonic, sleep=asyncio.sleep,
                         poll_s: float = 0.05) -> float | None:
    """Wait for the first reading of the scale. `force` is called for newtons or None and has `.error` (set when
    its background reading died). Returns the reading, or None when the reading died or `timeout_s` passed; the
    caller then prints force.why() and stops. A missing reading is None, never a guess."""
    t_end = clock() + timeout_s
    while True:
        value = force()
        if value is not None:
            return float(value)
        if force.error or clock() >= t_end:
            return None
        await sleep(poll_s)


def no_force_message(force, waited_s: float = FORCE_WAIT_S) -> str:
    return (f"error: no reading of the scale after waiting up to {waited_s:g} s: {force.why()}.\n"
            "Is --camera the scale's camera, is the scale in the picture, and was the calibration made with this camera?\n"
            "Nothing was sent to the arm.")


def resolve_calib(path: str) -> Path | None:
    """The calibration file: as given, or under the repository root when the script is not run from there."""
    for p in (Path(path), REPO / path):
        if p.is_file():
            return p
    return None


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the simulated pull-arm policy on the ElRobot follower (dry run unless --send).")
    ap.add_argument("--policy", required=True, help="policy.onnx exported by sim/rl/eval_pull.py")
    ap.add_argument("--joint_map", required=True, help="json with zero_tick and ticks_per_rad per motor (measure it first)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--bus", default=None, help="follower bus serial (default: FOLLOWER_BUS_SERIAL from .env)")
    ap.add_argument("--camera", required=True,
                    help="the scale's camera: a number, or part of its name (see `python -m gauge_reader cameras`)")
    ap.add_argument("--calib", default=DEFAULT_CALIB, help=f"the scale's calibration (default: {DEFAULT_CALIB})")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--send", action="store_true", help="really command the arm; without it nothing is sent")
    args = ap.parse_args(argv)
    calib = resolve_calib(args.calib)
    if calib is None:
        ap.error(f"no calibration at {args.calib}: run `python -m gauge_reader calibrate` first, or give the file with --calib")
    args.calib = str(calib)
    return args


async def read_joints(client, bus: str, state: dict):
    from lib.station import follow_motor_state

    async for bus_serial, motor_id, ms in follow_motor_state(client):
        if bus_serial == bus and ms.position is not None:
            state["ticks"][motor_id] = ms.position
            state["ranges"][motor_id] = (ms.range_min, ms.range_max)


async def main(argv=None):
    args = parse_args(argv)

    import onnxruntime as ort
    from seed_pose import fk  # forward kinematics from NormaCore's URDF (tools/seed_pose.py)
    from sim_pull_runner import CameraForce  # the scale read through its camera, in this process

    from lib.env import env_get
    from lib.station import connect, set_position

    jmap = load_joint_map(args.joint_map)
    bus = args.bus or env_get("FOLLOWER_BUS_SERIAL")
    sess = ort.InferenceSession(args.policy)
    in_name = sess.get_inputs()[0].name
    # The scale first, in this (the main) thread and before the arm is touched: a missing calibration or a camera
    # that will not open stops the program here, with the real message.
    try:
        read_force = CameraForce(args.camera, args.calib, stale_s=FORCE_STALE_S)
    except RuntimeError as e:
        raise SystemExit(f"error: {e}\nNothing was sent to the arm.") from None
    if await wait_for_force(read_force) is None:
        raise SystemExit(no_force_message(read_force))
    state = {"ticks": {}, "ranges": {}}
    client = await connect(args.host)
    tasks = [asyncio.create_task(read_joints(client, bus, state))]
    try:
        if await wait_for_joints(state):
            raise SystemExit(joints_timeout_message(state["ticks"], bus))

        def angles() -> np.ndarray:
            return np.array([to_rad(state["ticks"][m], jmap[m]) for m in ARM_MOTORS])

        def tip_and_dir(q7: np.ndarray):
            pos, rot_m = fk(q7)  # gripper origin and orientation from the seven joint angles
            return pos + rot_m[:, 1] * TIP  # the gripper points along its own +y

        q0 = angles()
        home = tip_and_dir(q0)
        u = home.copy(); u[2] = 0.0; u /= np.linalg.norm(u)  # along the scale: from the base towards the gripper, level
        side = np.cross([0.0, 0.0, 1.0], u)
        start = np.array([START[m] for m in ARM_MOTORS])
        target = q0.copy()
        prev_q, prev_force, action = q0.copy(), 0.0, np.zeros(5, dtype=np.float32)
        drive_idx = [m - 1 for m in DRIVEN]
        t_end = time.monotonic() + args.seconds
        while time.monotonic() < t_end:
            t0 = time.monotonic()
            force = read_force()  # None when the scale is not found, the reading is older than FORCE_STALE_S, or it died
            if force is None:
                print(f"no fresh force reading ({read_force.why()}): stopping (the arm keeps its last target)")
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
