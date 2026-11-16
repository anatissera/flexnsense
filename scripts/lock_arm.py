"""Lock every joint of a bus in its current position, or release them.

Locking is what turns the arm into a rigid anchor - the state the force-calibration rig needs
(docs/force-calibration.md step 4: "Torque must be enabled and the arm commanded to hold its
pose"). It does not move the arm: it reads each motor's current position, writes that same
value back as its goal, and only then enables torque, so there is no jump when the servo starts
actively holding.

    python scripts/lock_arm.py                  # lock all 8 motors (asks for confirmation)
    python scripts/lock_arm.py --yes             # skip the confirmation
    python scripts/lock_arm.py --release         # let the arm go limp again
    python scripts/lock_arm.py --motors 1,2,5    # only these motor ids

With torque on, the servo's own control loop holds position and draws current to resist being
pushed away from it - that resistance current is what log_force_calibration.py measures. This
does not replace Station's own safety cap (station.yaml current-threshold /
motor-current-thresholds, see docs/elrobot.md) - it is independent of it.
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


async def read_positions(client, bus_serial: str, motor_ids: set[int], seconds: float) -> dict[int, int]:
    from lib.station import follow_motor_state

    positions: dict[int, int] = {}
    deadline = time.monotonic() + seconds
    async for serial, motor_id, state in follow_motor_state(client):
        if serial == bus_serial and motor_id in motor_ids and state.position is not None:
            positions[motor_id] = state.position
        if time.monotonic() > deadline or len(positions) == len(motor_ids):
            break
    return positions


async def lock(host: str, bus_serial: str, motor_ids: list[int], seconds: float):
    from lib.station import connect, set_position, set_torque

    print(f"connecting to Station at {host} ...")
    client = await connect(host)

    print(f"reading current position of {len(motor_ids)} motor(s) on bus {bus_serial} ...")
    positions = await read_positions(client, bus_serial, set(motor_ids), seconds)
    missing = [m for m in motor_ids if m not in positions]
    if missing:
        sys.exit(f"error: no position reading for motor(s) {missing}; bus not responding for them?")

    for motor_id, pos in sorted(positions.items()):
        print(f"  motor {motor_id}: holding at {pos}")
        await set_position(client, bus_serial, motor_id, pos)

    await set_torque(client, bus_serial, motor_ids, enable=True)
    print(f"\nTorque enabled on {len(motor_ids)} motor(s). The arm is now a rigid anchor.")


async def release(host: str, bus_serial: str, motor_ids: list[int]):
    from lib.station import connect, set_torque

    print(f"connecting to Station at {host} ...")
    client = await connect(host)
    await set_torque(client, bus_serial, motor_ids, enable=False)
    print(f"Torque disabled on {len(motor_ids)} motor(s). The arm is limp - support it.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--station-host", help="default: STATION_HOST from .env, else localhost")
    parser.add_argument("--bus-serial", help="default: FOLLOWER_BUS_SERIAL from .env")
    parser.add_argument("--motors", default="1,2,3,4,5,6,7,8", help="comma-separated motor ids (default: all 8)")
    parser.add_argument("--release", action="store_true", help="disable torque instead of locking")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    parser.add_argument("--seconds", type=float, default=3.0, help="how long to wait for a position reading")
    args = parser.parse_args()

    load_env()
    host = args.station_host or os.environ.get("STATION_HOST", "localhost")
    bus_serial = args.bus_serial or os.environ.get("FOLLOWER_BUS_SERIAL")
    if not bus_serial:
        sys.exit("No bus serial: pass --bus-serial or set FOLLOWER_BUS_SERIAL in .env")
    motor_ids = [int(m) for m in args.motors.split(",")]

    action = "release torque on (let go limp)" if args.release else "lock (enable torque, hold position)"
    print(f"About to {action} motor(s) {motor_ids} on bus {bus_serial}.")
    if not args.release:
        print("The arm will actively resist being moved after this. Keep hands/fingers clear "
              "of pinch points while it locks.")

    if not args.yes:
        reply = input("\nType 'yes' to proceed: ").strip().lower()
        if reply != "yes":
            print("Aborted, nothing sent.")
            return

    if args.release:
        asyncio.run(release(host, bus_serial, motor_ids))
    else:
        asyncio.run(lock(host, bus_serial, motor_ids, args.seconds))


if __name__ == "__main__":
    main()
