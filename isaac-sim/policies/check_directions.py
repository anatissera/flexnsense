#!/usr/bin/env python
"""Print what the arm's model expects when each motor's position_norm goes UP a little, from the training start posture.

Use it to check, by hand and with torque off, that every motor turns the way the policy assumes: nudge one joint so
that its position_norm rises (Station's web page shows it) and compare where the jaws go with the line printed here.
A motor that goes the opposite way gets -1 in <policy folder>/joint_directions.json.

    python isaac-sim/policies/check_directions.py

Directions are in the arm's own frame: "forward" is from the base towards the jaws in the start posture, "left" is to
the left when standing behind the base and looking forward, "up" is away from the table.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_pull_runner import Arm, norm_to_rad, rad_to_norm  # noqa: E402

START = np.array([0.0, 0.04, 1.057, 0.072, 0.0, -1.169, 0.0])
STEP = 0.03  # of the joint's range


def main() -> None:
    arm, ones = Arm(), np.ones(7)
    norm = rad_to_norm(START, arm.lower, arm.upper, ones)
    pos, pointing = arm.gripper(START)
    tip = pos + 0.10 * pointing
    fwd = tip.copy(); fwd[2] = 0.0; fwd /= np.linalg.norm(fwd)
    left = np.cross([0.0, 0.0, 1.0], fwd)
    print("start posture as position_norm, motors 1..7:", norm.round(2).tolist())
    print(f"jaws at the start: {np.linalg.norm(tip[:2]) * 100:.0f} cm in front of the base, {tip[2] * 100:.0f} cm above the table\n")
    for m in range(7):
        n2 = norm.copy(); n2[m] += STEP
        p2, d2 = arm.gripper(norm_to_rad(n2, arm.lower, arm.upper, ones))
        move = (p2 + 0.10 * d2) - tip
        parts = [(move @ fwd, "forward", "back"), (move @ left, "left", "right"), (move[2], "up", "down")]
        words = [f"{abs(v) * 1000:4.1f} mm {a if v > 0 else b}" for v, a, b in parts if abs(v) > 0.0004]
        turn = np.degrees(np.arccos(np.clip(d2 @ pointing, -1, 1)))
        print(f"motor {m + 1}: position_norm +{STEP:.2f} ({STEP * (arm.upper[m] - arm.lower[m]):.3f} rad) -> jaws move "
              f"{', '.join(words) if words else 'almost nowhere'}; the gripper's direction turns {turn:.1f} deg")


if __name__ == "__main__":
    main()
