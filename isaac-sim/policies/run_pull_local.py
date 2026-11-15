#!/usr/bin/env python
"""The simulation-trained pull policy on the real arm, through the team's existing control loop.

It reuses robot_training/inference (branch feature/robot-training): control.py, station_io.py and safety.py do the
reading, the goal limits and the writing. This file only supplies the policy. The options of that loop all apply:
--dry-run, --max-ticks, --station, --bus-serial, --max-step-ticks, ...

    # in the environment that has norma-core's station_py (see robot_training/README.md), from the repository root
    python isaac-sim/policies/run_pull_local.py --policy isaac-sim/policies/pull_bench_v1 \\
        --force-camera 0 --calib config/scale_calib.json --camera-order side,gripper,table --dry-run
    ... then --max-ticks 30, then without a limit.

--force-camera is the camera that looks at the scale (`python -m gauge_reader cameras` lists them); it is opened by
this script. --calib is the scale's calibration on THIS machine (`python -m gauge_reader calibrate` makes it).

NEVER RUN ON HARDWARE YET. Read isaac-sim/README.md first.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="folder with policy.onnx and flex_sim.json")
    p.add_argument("--force-camera", required=True, help="the camera that looks at the scale: a number or part of its name")
    p.add_argument("--calib", default="config/scale_calib.json", help="the scale's calibration file on this machine")
    p.add_argument("--robot-training", default=None,
                   help="path to robot_training/inference (default: found from the repository root)")
    pre, _ = p.parse_known_args()
    inference = Path(pre.robot_training) if pre.robot_training else next(
        (r / "robot_training" / "inference" for r in HERE.parents if (r / "robot_training" / "inference").exists()), None)
    if inference is None:
        sys.exit("robot_training/inference not found: check out or merge feature/robot-training, or pass --robot-training")
    sys.path.insert(0, str(inference))
    from control import add_robot_args, run_control

    repo = next((r for r in HERE.parents if (r / "gauge_reader").is_dir() and (r / "lib").is_dir()), None)
    if repo is None:
        sys.exit("gauge_reader/ and lib/ not found above this folder: run from a checkout of the repository")
    sys.path.insert(0, str(repo))
    from sim_pull_runner import CameraForce, SimPullRunner

    add_robot_args(p)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if not Path(args.calib).is_file():
        sys.exit(f"no calibration at {args.calib}: run `python -m gauge_reader calibrate` on this machine, or pass --calib")
    try:
        force = CameraForce(args.force_camera, args.calib)
    except RuntimeError as e:
        sys.exit(str(e))
    import time

    t_end = time.monotonic() + 10.0
    while force() is None and time.monotonic() < t_end:  # the scale must be read before anything else starts
        if force.error:
            break
        time.sleep(0.1)
    if force() is None:
        sys.exit(f"no reading of the scale after 10 s: {force.why()}")
    runner = SimPullRunner(args.policy, force)
    args.hand_dim = 0
    print(f"policy {runner.cfg.type} | one goal per Station frame | scale reads {force():.2f} N on camera {args.force_camera} | "
          f"motor directions {runner.directions.astype(int).tolist()}")

    async def get_chunk(state, jpeg_by_view, task, prev=None):
        chunk, ms = await asyncio.get_running_loop().run_in_executor(
            None, runner.predict_from_jpegs, state, jpeg_by_view, task, prev)
        s = runner.last
        return chunk, 1, f"(force {s['force_n']:.2f} N, along {s['along_m'] * 1000:+.0f} mm, action {s['action'].round(2).tolist()}, {ms:.1f} ms)"

    asyncio.run(run_control(args, get_chunk, logging.getLogger("run_pull_local")))


if __name__ == "__main__":
    main()
