"""Move the arm to one of two named poses, slowly and in steps.

    .venv/bin/python scripts/arm/go_home.py --check         # where is it now? nothing moves
    .venv/bin/python scripts/arm/go_home.py                 # to the session home (the default)
    .venv/bin/python scripts/arm/go_home.py --to cpm         # to the cpm start pose

Exactly two destinations, named rather than an open path, because a typed `--path` would let
a typo or an unrelated file become somewhere the arm actually drives to. `--to` only accepts
`session.home_pose.NAMED_POSES`'s keys, and that dict is the one list of poses anyone may
send this script to.

`home` (the default) is where every session starts and ends: tether slack, nothing hooked, so
the control quality is repeatable rather than a matter of how the arm was last left, and the
pose-dependent bench numbers are used in the pose they were measured in. Why that pose:
`session/home_pose.py`.

`cpm` is `scripts/session/run_cpm.py`'s own start pose: tether already taken up, finger attached. Not
a parking pose, which is exactly why walking to it is also something worth doing on its own,
outside of running a stroke, for example to get the arm back there after it was moved by hand
or after `--to home`.

**This moves the arm.** It asks for a typed confirmation first and then interpolates, rather
than commanding the goal in one jump: a servo told to cross its whole range at once goes as
fast as it can, and the one thing this arm must never do near a hand is move fast. The step
size and the pause between steps are both capped, and Ctrl-C stops at the current waypoint
rather than carrying on.

It will not run with the tether under load. A pose change drags whatever is attached through
an arc, so the hook comes off the finger first.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm1.lib.env import env_get  # noqa: E402
from arm1.session.home_pose import (  # noqa: E402
    NAMED_POSES,
    distance_from_home,
    home_pose_metadata,
    home_waypoints,
    is_home,
    load_home_pose,
)


async def read_pose(client, bus_serial: str, motor_ids: list[int], seconds: float) -> dict:
    from arm1.lib.station import follow_motor_state

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


async def run(args) -> int:
    from arm1.lib.station import connect, set_position

    # args.to is already one of NAMED_POSES's keys (argparse enforces it via choices), so
    # this lookup cannot miss.
    label = args.to
    path = NAMED_POSES[args.to]
    home = load_home_pose(path)
    meta = home_pose_metadata(path)

    bus_serial = args.bus_serial or env_get("FOLLOWER_BUS_SERIAL")
    if not bus_serial:
        raise SystemExit("no follower bus serial: set FOLLOWER_BUS_SERIAL in .env or pass "
                         "--bus-serial.")
    host = args.station_host or env_get("STATION_HOST", "localhost")

    client = await connect(host=host)
    pose = await read_pose(client, bus_serial, sorted(home), args.seconds)
    if not pose:
        raise SystemExit("no motor positions arrived. Is Station running and this bus connected?")

    missing = [m for m in home if m not in pose]
    if missing:
        raise SystemExit(f"no reading for motor(s) {missing}. Moving the rest would twist the "
                         "arm into a pose nobody chose; fix the bus first.")

    gap = distance_from_home(pose, home)
    print(f"{label} holds every pull direction to {meta['worst_deviation_mm']} mm over "
          f"{meta['pull_distance_mm']:.0f} mm (it would pull with joints {meta['pull_motors']})")
    print(f"\n{'motor':<7}{'now':>8}{label[:8]:>8}{'delta':>9}")
    for m in sorted(home):
        print(f"{m:<7}{int(pose[m]):>8}{home[m]:>8}{home[m] - int(pose[m]):>+9}")
    print(f"\nfurthest joint is {gap} ticks from {label} "
          f"({'already there' if is_home(pose, args.tolerance, home) else 'not there yet'})")

    if args.check:
        return 0
    if is_home(pose, args.tolerance, home):
        print("Nothing to do.")
        return 0

    waypoints = home_waypoints(pose, home, args.max_step_ticks)
    print(f"\nThis MOVES the arm to {label}, in {len(waypoints)} steps of at most "
          f"{args.max_step_ticks} ticks.")
    print("The tether must not be under load and no finger may be attached: changing pose "
          "drags whatever is hooked up through an arc.")
    if input("Type 'go home' to proceed: ").strip().lower() != "go home":
        print("aborted; nothing moved.")
        return 1

    try:
        for i, way in enumerate(waypoints, start=1):
            for m in sorted(way):
                await set_position(client, bus_serial, m, way[m])
            print(f"  step {i}/{len(waypoints)}")
            await asyncio.sleep(args.settle)
    except KeyboardInterrupt:
        print(f"\nstopped. The arm is partway to {label}; rerun to finish, or move it by "
              "hand after `scripts/arm/lock_arm.py --release`.")
        return 1

    final = await read_pose(client, bus_serial, sorted(home), args.seconds)
    gap_now = distance_from_home(final, home) if final else None
    if gap_now is None:
        print(f"\nmoved, but could not read the arm back to confirm it reached {label}.")
    elif gap_now <= args.tolerance:
        print(f"\nat {label} (furthest joint {gap_now} ticks off).")
    else:
        print(f"\nstopped {gap_now} ticks from {label}, outside the {args.tolerance} tick "
              "tolerance. A joint may be blocked, or the pose may not be reachable with the "
              "rig in place.")
        return 1
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--to", choices=sorted(NAMED_POSES), default="home",
                   help="which saved pose to walk to. 'home' is the session's parking pose "
                        "(nothing hooked, tether slack); 'cpm' is scripts/session/run_cpm.py's start "
                        "pose (tether taken up, finger attached). Exactly these two, not an "
                        "open path: see session.home_pose.NAMED_POSES")
    p.add_argument("--check", action="store_true", help="report the distance, move nothing")
    p.add_argument("--max-step-ticks", type=int, default=80,
                   help="largest move per step (default 80, about 7 degrees)")
    p.add_argument("--settle", type=float, default=0.35, help="seconds between steps")
    p.add_argument("--tolerance", type=int, default=20,
                   help="ticks within which the arm counts as home")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--station-host", default=None)
    p.add_argument("--bus-serial", default=None)
    args = p.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
