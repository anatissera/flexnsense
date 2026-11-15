#!/usr/bin/env python
"""The simulation-trained pull policy on the real arm, through the team's existing control loop.

It reuses robot_training/inference (branch feature/robot-training): control.py, station_io.py and safety.py do the
reading, the goal limits and the writing. This file only supplies the policy. The options of that loop all apply:
--dry-run, --max-ticks, --station, --bus-serial, --max-step-ticks, ...

    # terminal 1: the scale reading (opens the camera)
    .venv/bin/python -m gauge_reader.server
    # terminal 2, in the environment that has norma-core's station_py (see robot_training/README.md)
    python isaac-sim/policies/run_pull_local.py --policy isaac-sim/policies/pull_bench_v1 \\
        --camera-order side,gripper,table --dry-run
    ... then --max-ticks 30, then without a limit.

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
    p.add_argument("--force-url", default="ws://localhost:8766/force", help="gauge_reader.server WebSocket")
    p.add_argument("--robot-training", default=None,
                   help="path to robot_training/inference (default: found from the repository root)")
    pre, _ = p.parse_known_args()
    inference = Path(pre.robot_training) if pre.robot_training else next(
        (r / "robot_training" / "inference" for r in HERE.parents if (r / "robot_training" / "inference").exists()), None)
    if inference is None:
        sys.exit("robot_training/inference not found: check out or merge feature/robot-training, or pass --robot-training")
    sys.path.insert(0, str(inference))
    from control import add_robot_args, run_control

    from sim_pull_runner import GaugeForce, SimPullRunner

    add_robot_args(p)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    runner = SimPullRunner(args.policy, GaugeForce(args.force_url))
    args.hand_dim = 0
    print(f"policy {runner.cfg.type} | one goal per Station frame | force from {args.force_url} | "
          f"motor directions {runner.directions.astype(int).tolist()}")

    async def get_chunk(state, jpeg_by_view, task, prev=None):
        chunk, ms = await asyncio.get_running_loop().run_in_executor(
            None, runner.predict_from_jpegs, state, jpeg_by_view, task, prev)
        s = runner.last
        return chunk, 1, f"(force {s['force_n']:.2f} N, along {s['along_m'] * 1000:+.0f} mm, action {s['action'].round(2).tolist()}, {ms:.1f} ms)"

    asyncio.run(run_control(args, get_chunk, logging.getLogger("run_pull_local")))


if __name__ == "__main__":
    main()
