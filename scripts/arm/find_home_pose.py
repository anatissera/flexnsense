"""Search the workspace for the pose that pulls straightest, and save it as home.

    .venv/bin/python scripts/arm/find_home_pose.py             # search and report
    .venv/bin/python scripts/arm/find_home_pose.py --save      # and write config/calibration/

Geometry only, from NormaCore's URDF: nothing moves, Station is not needed, and the whole
search runs in the repo's normal .venv in a couple of minutes.

Why a fixed home pose at all: a straight pull is only as good as the pose it starts from. The
same three joints that track a line to half a millimetre from one configuration wander 8.6 mm
from another, and a pose can start well conditioned and drive itself towards a singularity
partway through a 90 mm pull. Picking one good pose and always starting there makes the
control quality repeatable, and it means the pose-dependent bench numbers are measured in the
pose they are used in.

How the search scores a pose, in two stages, because the cheap score alone is not enough:

  1. the smallest singular value of the jacobian, which is the arm's gain in its *worst*
     direction. A pose that is strong along the pull but weak sideways scores badly here,
     which is what we want, since a repetition pulls out and back.
  2. the survivors are then simulated through full pulls in many directions, out and back, and
     ranked by the worst deviation any of them actually produced. Stage one is a proxy
     evaluated at the starting pose; stage two is the thing we care about, measured over the
     whole path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm1.lib.kinematics import load_chain  # noqa: E402

CALIB_DIR = Path("config/calibration")
HOME_PATH = CALIB_DIR / "home-pose.json"


def joint_ranges() -> dict[int, tuple[int, int]]:
    files = sorted(CALIB_DIR.glob("follower-*.json"))
    if not files:
        raise SystemExit(f"no follower calibration in {CALIB_DIR}/; the search needs the "
                         "joint ranges to stay inside.")
    ranges = json.loads(files[0].read_text())["ranges"]
    return {int(k): (int(v[0]), int(v[1])) for k, v in ranges.items()}


def worst_deviation(chain, pose, motors, distance_m, directions, steps=40, give_up_m=0.01):
    """Worst straight-line error over many pull directions, out and back."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(directions):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        for sign in (1, -1):
            err, _ = chain.line_error(motors, pose, d * sign, distance_m, steps)
            worst = max(worst, err)
            if worst > give_up_m:
                return worst          # already bad enough to reject; stop paying for detail
    return worst


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", type=int, default=6000, help="poses to try in stage one")
    p.add_argument("--shortlist", type=int, default=12, help="how many reach stage two")
    p.add_argument("--distance", type=float, default=90.0, help="pull distance in mm")
    p.add_argument("--directions", type=int, default=14,
                   help="pull directions to test per pose (each is tried out and back)")
    p.add_argument("--margin", type=float, default=0.15,
                   help="fraction of each joint's range to keep clear of its limits")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save", action="store_true", help=f"write {HOME_PATH}")
    args = p.parse_args(argv)

    chain = load_chain()
    ranges = joint_ranges()
    motors = [m for m in chain.motor_ids if m != 8]
    distance_m = args.distance / 1000.0

    print(f"stage one: scoring {args.samples} poses by worst-direction gain ...")
    rng = np.random.default_rng(args.seed)
    scored = []
    for _ in range(args.samples):
        pose = {}
        for m in motors:
            lo, hi = ranges[m]
            margin = int((hi - lo) * args.margin)
            pose[m] = int(rng.integers(lo + margin, hi - margin))
        picks = chain.choose_pull_motors(pose)
        sv = np.linalg.svd(chain.jacobian(picks, pose), compute_uv=False)
        if sv[-1] < 1e-9:
            continue
        scored.append((float(sv[-1]), float(sv[0] / sv[-1]), pose, picks))
    if not scored:
        raise SystemExit("no usable pose found; try a smaller --margin.")
    scored.sort(key=lambda r: -r[0])

    print(f"stage two: simulating {args.distance:.0f} mm pulls in {args.directions} directions, "
          "out and back, for the top candidates ...\n")
    print(f"{'cond':>6} {'picks':<12} {'worst deviation':>17}")
    finalists = []
    for sigma, cond, pose, picks in scored[:args.shortlist]:
        dev = worst_deviation(chain, pose, picks, distance_m, args.directions)
        finalists.append((dev, pose, picks, cond, sigma))
        print(f"{cond:>6.2f} {str(picks):<12} {dev*1000:>14.2f} mm")

    finalists.sort(key=lambda r: r[0])
    dev, pose, picks, cond, sigma = finalists[0]
    tip = chain.tip_position(pose)

    print(f"\nbest pose: every direction held to {dev*1000:.2f} mm over {args.distance:.0f} mm")
    print("  ticks: " + ", ".join(f"m{m}={pose[m]}" for m in sorted(pose)))
    print(f"  joints it would pull with: {picks}")
    print(f"  gripper at x={tip[0]*1000:+.0f} y={tip[1]*1000:+.0f} z={tip[2]*1000:+.0f} mm")
    print("  headroom to the joint limits:")
    for m in sorted(pose):
        lo, hi = ranges[m]
        print(f"    m{m}: {pose[m] - lo:>5} down, {hi - pose[m]:>5} up")

    if args.save:
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        HOME_PATH.write_text(json.dumps({
            "ticks": {str(m): int(pose[m]) for m in sorted(pose)},
            "pull_motors": picks,
            "worst_deviation_mm": round(dev * 1000, 3),
            "pull_distance_mm": args.distance,
            "directions_tested": args.directions * 2,
            "condition_number": round(cond, 3),
            "smallest_singular_value": round(sigma, 5),
            "search": {"samples": args.samples, "seed": args.seed, "margin": args.margin},
            "note": ("Found by scripts/arm/find_home_pose.py. Geometry only, from the URDF: this "
                     "has not been checked against the physical arm for reachability with the "
                     "rig in place. Confirm nothing collides before trusting it."),
        }, indent=2) + "\n")
        print(f"\nwrote {HOME_PATH}")
        print("Check the arm can actually reach it with the rig in place before using it.")
    else:
        print("\n(--save to write it to config/calibration/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
