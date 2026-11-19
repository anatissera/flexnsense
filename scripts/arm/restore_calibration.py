"""Restore a joint-range calibration from a saved config/calibration/<name>.json file.

For recovering a calibration that was lost from the servos' firmware (a reset, a reflash, a
replaced servo) - NOT for a first-time calibration. The numbers in that file came from
physically moving each joint through its real range in Station's web UI once; writing them
back here skips that physical step and trusts the file instead. If the arm's actual mechanical
range changed since the file was recorded (different joint, different build), recalibrate by
hand in the web UI instead of using this.

    python scripts/arm/restore_calibration.py config/calibration/follower-5B8E114979.json
    python scripts/arm/restore_calibration.py config/calibration/follower-5B8E114979.json --yes

Without --yes this only shows what it would write and asks for confirmation; nothing is sent
until you type "yes". After writing, it reads the bus back and reports whether every motor's
range now matches the file, so a mismatch is caught immediately rather than discovered later
during force calibration.
"""
from __future__ import annotations

import argparse
import asyncio
import json
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


def load_ranges(path: Path) -> tuple[str, dict[int, tuple[int, int]]]:
    data = json.loads(path.read_text())
    bus_serial = data["bus_serial"]
    ranges = {int(mid): (lo, hi) for mid, (lo, hi) in data["ranges"].items()}
    return bus_serial, ranges


async def restore(host: str, bus_serial: str, ranges: dict[int, tuple[int, int]], seconds: float):
    from arm1.lib.station import connect, follow_motor_state, freeze_calibration

    print(f"connecting to Station at {host} ...")
    client = await connect(host)
    print(f"connected. writing range for {len(ranges)} motor(s) on bus {bus_serial} ...")

    await freeze_calibration(client, bus_serial, ranges)

    print(f"reading the bus back ({seconds:.0f}s) to confirm ...")
    seen: dict[int, tuple[int, int]] = {}
    deadline = time.monotonic() + seconds
    async for serial, motor_id, state in follow_motor_state(client):
        if serial == bus_serial:
            seen[motor_id] = (state.range_min, state.range_max)
        if time.monotonic() > deadline:
            break

    mismatches = []
    for motor_id, expected in ranges.items():
        actual = seen.get(motor_id)
        if actual is None:
            mismatches.append(f"  motor {motor_id}: no reading (bus not responding for this id?)")
        elif actual != expected:
            mismatches.append(f"  motor {motor_id}: wrote {expected}, bus now reports {actual}")

    if mismatches:
        print(f"\n{len(mismatches)} mismatch(es) after writing:")
        print("\n".join(mismatches))
        sys.exit(1)

    print(f"\nAll {len(ranges)} motor(s) confirmed. Calibration restored.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path, help="config/calibration/<name>.json to restore from")
    parser.add_argument("--station-host", help="default: STATION_HOST from .env, else localhost")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    parser.add_argument("--seconds", type=float, default=3.0, help="how long to watch the bus after writing")
    args = parser.parse_args()

    if not args.file.is_file():
        sys.exit(f"error: {args.file} not found")

    load_env()
    host = args.station_host or os.environ.get("STATION_HOST", "localhost")
    bus_serial, ranges = load_ranges(args.file)

    print(f"About to write this joint-range calibration to bus {bus_serial}:\n")
    for motor_id, (lo, hi) in sorted(ranges.items()):
        print(f"  motor {motor_id}: range=[{lo}, {hi}]")
    print(
        "\nThis replaces whatever calibration is currently on those servos. Only do this if "
        "you intend to restore a known-good range (e.g. after a reset), not as a substitute "
        "for calibrating a joint whose real range you don't actually know."
    )

    if not args.yes:
        reply = input("\nType 'yes' to write this to the servos: ").strip().lower()
        if reply != "yes":
            print("Aborted, nothing written.")
            return

    asyncio.run(restore(host, bus_serial, ranges, args.seconds))


if __name__ == "__main__":
    main()
