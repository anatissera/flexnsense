"""Nudge exactly one motor by a small, safe tick delta, then put it back. For identifying which
motor moves along the tether axis before running measure_ticks_per_metre.py.

    .venv/bin/python scripts/arm/jog_one_motor.py --motor 4

Run this once per motor id (1 to 8) with the tether rigged but deliberately SLACK, no finger
anywhere near the rig. Watch the arm (or Station's web UI) each time: the motor that moves the
attachment point ALONG the pull direction, rather than sideways or in an arc that does not
change the tether's length, is the one `measure_ticks_per_metre.py` and `run_session.py --motor`
need.

Nothing is under tension while the tether is slack, so a wrong guess here costs nothing - that is
why this step exists, rather than guessing from Station's web UI, which does not expose a jog
control we have confirmed works for this.

Moves a small default (40 ticks, override with --ticks), waits --settle, then returns to the
exact starting position - on the normal path, on a bad keystroke, and on Ctrl-C alike, the same
safe-return pattern as measure_ticks_per_metre.py.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.env import env_get  # noqa: E402


async def run(args) -> int:
    from lib.station import connect, follow_motor_state, set_position

    bus_serial = args.bus_serial or env_get("FOLLOWER_BUS_SERIAL")
    if not bus_serial:
        raise SystemExit("no follower bus serial: set FOLLOWER_BUS_SERIAL in .env or pass "
                         "--bus-serial.")
    host = args.station_host or env_get("STATION_HOST", "localhost")

    print("This moves one motor a small amount with the tether rigged. Keep it SLACK (no "
          "tension) so a wrong motor costs nothing.")
    if input("Type 'no finger attached' to confirm there is none, anywhere near the rig: "
            ).strip().lower() != "no finger attached":
        print("aborted; nothing moved.")
        return 1

    client = await connect(host=host)
    position: dict = {"ticks": None}

    async def watch():
        async for bus, mid, motor in follow_motor_state(client):
            if bus == bus_serial and mid == args.motor and motor.position is not None:
                position["ticks"] = motor.position

    watcher = asyncio.create_task(watch())
    try:
        for _ in range(50):  # up to ~5s at 100ms updates, same cadence as Station's inference queue
            if position["ticks"] is not None:
                break
            await asyncio.sleep(0.1)
        if position["ticks"] is None:
            raise SystemExit(f"never got a position reading for motor {args.motor} on {bus_serial}; "
                             "is Station running and this bus connected?")

        baseline = position["ticks"]
        target = baseline + args.ticks
        print(f"motor {args.motor}: {baseline} -> {target} ({args.ticks:+d} ticks). Watch the arm now.")
        await set_position(client, bus_serial, args.motor, target)
        await asyncio.sleep(args.settle)
        print("returning to baseline ...")
    finally:
        try:
            await set_position(client, bus_serial, args.motor, baseline)
        except Exception as exc:  # noqa: BLE001
            print(f"!! could not return motor {args.motor} to baseline ({baseline} ticks): {exc}")
            print(f"!! release torque by hand: python scripts/arm/lock_arm.py --release "
                  f"--motors {args.motor}, or power down Station.")
        watcher.cancel()

    print("done. Did the attachment point move ALONG the pull direction? If not, try the next motor.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motor", type=int, required=True, help="motor id, 1 to 8")
    p.add_argument("--ticks", type=int, default=40, help="how far to nudge it (default 40, small on purpose)")
    p.add_argument("--settle", type=float, default=1.0)
    p.add_argument("--station-host", default=None)
    p.add_argument("--bus-serial", default=None)
    args = p.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
