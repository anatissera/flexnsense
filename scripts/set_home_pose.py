"""Capture the arm's current pose, as placed by hand, and save it to one of two named slots.

    .venv/bin/python scripts/lock_arm.py --release   # so the arm can be moved by hand
    # ... move the arm to where it should sit between sessions ...
    .venv/bin/python scripts/set_home_pose.py --save              # --to home, the default
    .venv/bin/python scripts/set_home_pose.py --save --to cpm     # scripts/run_cpm.py's start

`--to` only accepts `session.home_pose.NAMED_POSES`'s two keys, not an arbitrary path: a
captured pose is only trustworthy because someone placed and checked it against the real rig,
so the destinations it may be written to are the same closed, named pair `scripts/go_home.py`
reads from, not any file that happens to have a `ticks` key.

`scripts/find_home_pose.py` searches for a home pose from the URDF alone, which is geometry
that has never been checked against the physical rig: collisions, cables, the leader arm, the
camera mount, none of that is in the model. This is the other way round. Someone places the
arm by hand, knowing the real rig, and this reads that pose back and reports the same
straight-pull numbers `find_home_pose.py` would have computed for a candidate, so the choice
is informed rather than a guess either way.

It does not move anything. Read the pose, report on it, save it if asked.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.env import env_get  # noqa: E402
from lib.kinematics import load_chain  # noqa: E402
from session.home_pose import NAMED_POSES  # noqa: E402

CALIB_DIR = Path("config/calibration")
HOME_PATH = CALIB_DIR / "home-pose.json"


async def read_pose(host: str, bus_serial: str, motor_ids: list[int], seconds: float) -> dict:
    from lib.station import connect, follow_motor_state

    client = await connect(host=host)
    found: dict[int, float] = {}

    async def collect():
        async for bus, mid, motor in follow_motor_state(client):
            if bus == bus_serial and motor.position is not None:
                found[mid] = motor.position
                if all(m in found for m in motor_ids):
                    return

    try:
        await asyncio.wait_for(collect(), timeout=seconds)
    except asyncio.TimeoutError:
        pass
    return found


def worst_deviation(chain, pose, motors, distance_m, directions, seed=7):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(directions):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        for sign in (1, -1):
            err, _ = chain.line_error(motors, pose, d * sign, distance_m, steps=40)
            worst = max(worst, err)
    return worst


def joint_ranges() -> dict[int, tuple[int, int]]:
    files = sorted(CALIB_DIR.glob("follower-*.json"))
    if not files:
        return {}
    ranges = json.loads(files[0].read_text())["ranges"]
    return {int(k): (int(v[0]), int(v[1])) for k, v in ranges.items()}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distance", type=float, default=90.0, help="pull distance to test, mm")
    p.add_argument("--directions", type=int, default=14,
                   help="pull directions to test, each tried out and back")
    p.add_argument("--save", action="store_true", help="write the pose to --to")
    p.add_argument("--to", choices=sorted(NAMED_POSES), default="home",
                   help="which of the two named poses to write. 'home' is the session's "
                        "parking pose, the one every admittance session starts from and "
                        "returns to. 'cpm' is scripts/run_cpm.py's start pose: held under "
                        "tension with the finger already hooked, and therefore NOT somewhere "
                        "the arm should be parked between sessions. Exactly these two, not "
                        "an open path: see session.home_pose.NAMED_POSES")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--station-host", default=None)
    p.add_argument("--bus-serial", default=None)
    args = p.parse_args(argv)

    chain = load_chain()
    motors = [m for m in chain.motor_ids if m != 8]

    bus_serial = args.bus_serial or env_get("FOLLOWER_BUS_SERIAL")
    if not bus_serial:
        raise SystemExit("no follower bus serial: set FOLLOWER_BUS_SERIAL in .env or pass "
                         "--bus-serial.")
    host = args.station_host or env_get("STATION_HOST", "localhost")

    print("reading the arm's current pose ...")
    pose = asyncio.run(read_pose(host, bus_serial, motors, args.seconds))
    missing = [m for m in motors if m not in pose]
    if missing:
        raise SystemExit(f"no reading for motor(s) {missing}. Is Station running and this "
                         "bus connected?")
    pose = {m: int(pose[m]) for m in motors}

    ranges = joint_ranges()
    print(f"\n{'motor':<7}{'ticks':>7}  headroom to limits")
    out_of_range = []
    for m in sorted(pose):
        if m in ranges:
            lo, hi = ranges[m]
            if not (lo < pose[m] < hi):
                out_of_range.append(m)
            print(f"{m:<7}{pose[m]:>7}  {pose[m]-lo:>5} down, {hi-pose[m]:>5} up")
        else:
            print(f"{m:<7}{pose[m]:>7}  (no calibration on file to check against)")
    if out_of_range:
        raise SystemExit(f"\nmotor(s) {out_of_range} are at or beyond their calibrated range. "
                         "Something is wrong with this reading or this pose; not proceeding.")

    distance_m = args.distance / 1000.0
    picks = chain.choose_pull_motors(pose)
    tip = chain.tip_position(pose)
    dev_any = worst_deviation(chain, pose, picks, distance_m, args.directions)

    # The number that actually matters for a real rig: straightness along the directions a
    # tether plausibly runs in, which are each joint's own pull direction (left-right,
    # up-down, forward-back from the gripper), not an arbitrary direction in 3D. "Worst over
    # every direction in space" includes diagonals no tether will ever use, and can look much
    # worse than what this pose will actually be asked to do.
    plausible = {}
    for m in (1, 2, 3, 4, 5, 6):
        d = chain.tip_velocity_direction(m, pose)
        fwd, _ = chain.line_error(picks, pose, d, distance_m, steps=40)
        back, _ = chain.line_error(picks, pose, -d, distance_m, steps=40)
        plausible[m] = max(fwd, back)
    dev_plausible = max(plausible.values())

    print(f"\ngripper at x={tip[0]*1000:+.0f} y={tip[1]*1000:+.0f} z={tip[2]*1000:+.0f} mm")
    print(f"best pull joints from here: {picks}")
    print(f"\nstraightness along each motor's own direction (a proxy for where a real tether "
          f"would run), out and back, {args.distance:.0f} mm:")
    for m, d in sorted(plausible.items()):
        print(f"  along motor {m}'s direction: {d*1000:5.2f} mm")
    print(f"\nworst over {args.directions * 2} directions anywhere in 3D (includes diagonals no "
          f"tether runs along): {dev_any*1000:.2f} mm")
    if dev_plausible < 0.002:
        print(f"\nFor this pose, the directions a tether would plausibly run in track to "
              f"{dev_plausible*1000:.2f} mm, as good as a searched pose. The large number above "
              "is from directions this rig will not actually pull in.")
    elif dev_plausible > 0.003:
        print(f"\nWarning: even the plausible tether directions deviate up to "
              f"{dev_plausible*1000:.1f} mm here. Check the tether's real direction against the "
              "list above before trusting this pose.")
    dev = dev_plausible

    if args.save:
        out = Path(NAMED_POSES[args.to])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "ticks": {str(m): pose[m] for m in sorted(pose)},
            "pull_motors": picks,
            "worst_deviation_mm": round(dev * 1000, 3),
            "pull_distance_mm": args.distance,
            "directions_tested": "plausible tether axes: each of motors 1-6's own pull "
                                 "direction, out and back (12 directions)",
            "worst_deviation_any_direction_mm": round(dev_any * 1000, 3),
            "source": "set_home_pose.py: placed by hand on the real rig, not searched",
            "note": ("Chosen on the physical arm, with the rig, cables and the other arm in "
                     "place, rather than from URDF geometry alone. Re-run "
                     "scripts/measure_ticks_per_metre.py in this pose before trusting it for "
                     "ticks_per_metre or yield_sign, since both are pose-dependent."),
        }, indent=2) + "\n")
        print(f"\nwrote {out}")
    else:
        print(f"\n(--save to write this pose to {NAMED_POSES[args.to]!r} as {args.to!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
