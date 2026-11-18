"""Check Station's connection and calibration state for a bus. Does not calibrate.

NormaCore does not publish a scripted calibration procedure for the ElRobot: joint range
(min/max per motor) is set in Station's web UI (*ST3215 bus calibration* page), by hand, once
per physical arm, and then committed to the repo - see config/calibration/README.md.

This script only answers "is Station reachable, and does this bus already have a range for
every motor", which is what you want before trusting any position command.

    python scripts/arm/calibrate.py --bus-serial <SERIAL>
    python scripts/arm/calibrate.py --bus-serial <SERIAL> --motors 8   # expect 8, warn if fewer respond

For the SO-101 via LeRobot instead: scripts/lerobot/calibrate.py.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `lib`


def load_env(path=".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


async def check(host, bus_serial, expected_motors, seconds):
    from arm1.lib.station import connect, follow_motor_state

    print(f"connecting to Station at {host} ...")
    client = await connect(host)
    print(f"connected. watching bus {bus_serial} for {seconds:.0f}s ...")

    seen = {}
    deadline = time.monotonic() + seconds
    async for serial, motor_id, state in follow_motor_state(client):
        if serial == bus_serial:
            seen[motor_id] = state
        if time.monotonic() > deadline:
            break

    if not seen:
        sys.exit(f"error: no motors seen on bus {bus_serial!r} in {seconds:.0f}s. "
                  "Wrong serial, or Station isn't talking to this bus.")

    print(f"\n{'motor':>6s} {'range_min':>10s} {'range_max':>10s} {'status':>14s}")
    uncalibrated = []
    for motor_id in sorted(seen):
        s = seen[motor_id]
        calibrated = s.range_min != 0 or s.range_max != 0
        if not calibrated:
            uncalibrated.append(motor_id)
        print(f"{motor_id:6d} {s.range_min:10d} {s.range_max:10d} "
              f"{'OK' if calibrated else 'NOT CALIBRATED':>14s}")

    if len(seen) < expected_motors:
        missing = sorted(set(range(1, expected_motors + 1)) - set(seen))
        print(f"\nwarning: expected {expected_motors} motors, saw {len(seen)}. "
              f"Missing ids: {missing}")

    if uncalibrated:
        print(f"\n{len(uncalibrated)} motor(s) not calibrated: {uncalibrated}")
        print("Calibrate this bus in Station's web UI (ST3215 bus calibration page), "
              "then commit the result per config/calibration/README.md.")
    else:
        print(f"\nAll {len(seen)} motors report a calibrated range.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--station-host", help="default: STATION_HOST from .env, else localhost")
    parser.add_argument("--bus-serial", help="default: FOLLOWER_BUS_SERIAL from .env")
    parser.add_argument("--motors", type=int, default=8, help="expected motor count (ElRobot: 8)")
    parser.add_argument("--seconds", type=float, default=3.0, help="how long to watch the bus")
    args = parser.parse_args()

    load_env()
    host = args.station_host or os.environ.get("STATION_HOST", "localhost")
    bus_serial = args.bus_serial or os.environ.get("FOLLOWER_BUS_SERIAL")
    if not bus_serial:
        sys.exit("No bus serial: pass --bus-serial or set FOLLOWER_BUS_SERIAL in .env "
                  "(find it on the bus card in Station's web UI)")

    asyncio.run(check(host, bus_serial, args.motors, args.seconds))


if __name__ == "__main__":
    main()
