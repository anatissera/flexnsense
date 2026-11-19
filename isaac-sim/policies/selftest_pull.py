#!/usr/bin/env python
"""Test the pull policy's whole inference chain WITHOUT a robot, a camera or Station.

A stand-in arm and scale replace the real ones, at Station's pace (10 frames a second):
  * the arm: each motor moves towards its goal with a lag, in servo ticks, within the follower's calibrated ranges;
  * the scale: a spring between the point between the jaws and a finger that pulls away, holds and comes back;
  * the force reading: in steps of 5 gf, with noise, as the camera gives it.
Every frame goes through the real code: position_norm -> SimPullRunner (the exported network) -> goal_norm ->
ticks -> the control loop's GoalSafety (if arm2/training/inference is found) -> the stand-in arm.

It runs the same pulls twice, with the policy and with the arm holding still, and prints the peak forces.

    python isaac-sim/policies/selftest_pull.py --policy isaac-sim/policies/pull_bench_v1
Exit code 0 if the policy keeps the average peak force under the limit and well under the arm held still.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sim_pull_runner import Arm, SimPullRunner, norm_to_rad, rad_to_norm  # noqa: E402

# the follower's calibrated ranges (config/calibration/follower-5B8E114979.json), motors 1..8
RANGES = [(1085, 3011), (996, 3100), (891, 3205), (898, 3198), (139, 3957), (1024, 3072), (109, 3987), (1242, 2854)]


def find_safety():
    for r in HERE.parents:
        d = r / "robot_training" / "inference"
        if (d / "safety.py").exists():
            sys.path.insert(0, str(d))
            from safety import GoalSafety, goal_norm_to_ticks

            return GoalSafety, goal_norm_to_ticks
    return None, None


def run(policy_dir: str, use_policy: bool, pulls: int, seed: int, lag_s: float, verbose: bool):
    GoalSafety, to_ticks = find_safety()
    arm, rng = Arm(), np.random.default_rng(seed)
    meta = json.loads((Path(policy_dir) / "flex_sim.json").read_text())
    cap, rest, dt = meta["cap_n"], 0.23, 1.0 / meta["rate_hz"]
    lo = np.array([r[0] for r in RANGES], dtype=np.float64)
    hi = np.array([r[1] for r in RANGES], dtype=np.float64)
    follow = 1.0 - math.exp(-dt / lag_s)  # share of the way to its goal a motor covers in one frame
    peaks, overs, drifts, aborts = [], [], [], 0
    for _ in range(pulls):
        k = 58.86 * rng.uniform(0.7, 1.3)
        amp, ramp, hold, delay = rng.uniform(0.03, 0.10), rng.uniform(0.8, 3.0), rng.uniform(1.0, 3.0), rng.uniform(0.3, 1.5)
        q0 = np.array(meta["start_rad"]) + np.array([0, 1, 1, 1, 0, 1, 0]) * rng.uniform(-0.1, 0.1, 7)
        norm = np.concatenate([rad_to_norm(q0, arm.lower, arm.upper, np.ones(7)), [0.05]])
        ticks = lo + norm * (hi - lo)
        state = {"force": 0.0}
        runner = SimPullRunner(policy_dir, lambda: state["force"])
        safety = GoalSafety(n_joints=8, gripper_index=7) if GoalSafety else None
        tip0 = None
        peak, over, idle_drift, idle_n = 0.0, 0.0, 0.0, 0
        for i in range(int(10.0 / dt)):
            t = i * dt - delay
            sm = lambda p: (lambda c: c * c * (3 - 2 * c))(min(1.0, max(0.0, p)))
            pull = amp * (sm(t / ramp) - sm((t - ramp - hold) / ramp))
            norm = (ticks - lo) / (hi - lo)
            q = norm_to_rad(norm[:7], arm.lower, arm.upper, np.ones(7))
            pos, pointing = arm.gripper(q)
            tip = pos + 0.10 * pointing
            if tip0 is None:
                tip0 = tip.copy()
                u = tip0.copy(); u[2] = 0.0; u /= np.linalg.norm(u)
            finger = tip0 + u * (rest + pull)
            force = k * max(0.0, float(np.linalg.norm(finger - tip)) - rest)
            state["force"] = max(0.0, round((force + rng.normal(0, 0.05)) / 0.049) * 0.049)  # what the camera would read
            peak = max(peak, force)
            over += dt * (force > 1.1 * cap)
            if pull < 1e-4:
                idle_drift += abs(float((tip - tip0) @ u)); idle_n += 1
            if use_policy:
                goal_norm, _ = runner.predict_from_jpegs(norm.astype(np.float32), {}, "")
                goal_norm = goal_norm[0]
            else:
                goal_norm = norm
            if to_ticks:
                target = to_ticks(goal_norm, RANGES)
                goal, msg = safety.apply(target, np.rint(ticks).astype(np.int64))
                if goal is None:
                    aborts += 1
                    continue
            else:
                goal = lo + np.clip(goal_norm, 0, 1) * (hi - lo)
            ticks = ticks + follow * (np.asarray(goal, dtype=np.float64) - ticks)
        peaks.append(peak); overs.append(over); drifts.append(idle_drift / max(1, idle_n))
        if verbose:
            print(f"   pull {amp * 100:4.1f} cm, spring {k:5.1f} N/m: peak {peak:4.2f} N (rigid would be {k * amp:4.2f} N)")
    return np.array(peaks), np.array(overs), np.array(drifts), aborts, bool(to_ticks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--pulls", type=int, default=30)
    ap.add_argument("--lag", type=float, default=0.15, help="how slowly the stand-in motors follow their goal [s]")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    meta = json.loads((Path(a.policy) / "flex_sim.json").read_text())
    out = {}
    for name, use in (("policy", True), ("held still", False)):
        peaks, overs, drifts, aborts, with_safety = run(a.policy, use, a.pulls, a.seed, a.lag, a.verbose and use)
        out[name] = peaks
        print(f"{name:10s}: peak force average {peaks.mean():4.2f} N, worst {peaks.max():4.2f} N | pulls more than 10 % over "
              f"the limit: {100 * (peaks > 1.1 * meta['cap_n']).mean():3.0f} % | time over, average {overs.mean():4.2f} s | "
              f"drift at rest {1000 * drifts.mean():4.1f} mm | safety aborts {aborts}")
    print(f"limit {meta['cap_n']:.2f} N | {a.pulls} pulls | motors lag {a.lag:.2f} s | "
          f"{'through robot_training GoalSafety' if with_safety else 'arm2/training/inference not found: without GoalSafety'}")
    ok = out["policy"].mean() < meta["cap_n"] and out["policy"].mean() < 0.75 * out["held still"].mean()
    print("RESULT:", "OK" if ok else "NOT OK")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
