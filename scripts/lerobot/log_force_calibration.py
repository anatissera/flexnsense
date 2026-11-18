"""NOT our hardware. Kept in case we ever port to a LeRobot-supported arm (SO-101 etc).

Our arm is the ElRobot, controlled through NormaCore Station - see scripts/measure/log_force_calibration.py
(the default, one level up) and docs/elrobot.md. This file is the original SO-101-via-LeRobot
script, unmodified, parked here rather than deleted.

Log Feetech servo load/current while known forces are applied to arm B.

Used to calibrate servo current against grams read off the PESOLA spring scale.
See docs/force-calibration.md for the rig and the procedure.

    python scripts/lerobot/log_force_calibration.py --probe
    python scripts/lerobot/log_force_calibration.py --tare
    python scripts/lerobot/log_force_calibration.py --out config/measurements/calib_up.csv
    python scripts/lerobot/log_force_calibration.py --fit config/measurements/calib_up.csv

Register names are discovered at runtime rather than assumed, since they vary
across lerobot versions. Run --probe first to see what this arm exposes.
"""

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root, for `lib`
from lib.units import grams_force_to_newtons  # noqa: E402

# Names seen across lerobot/Feetech control tables, most useful first.
CANDIDATE_REGISTERS = [
    "Present_Load",
    "Present_Current",
    "Present_Position",
    "Present_Temperature",
    "Present_Voltage",
]


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


def connect(port, arm_id):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    config = SO101FollowerConfig(port=port, id=arm_id)
    robot = SO101Follower(config)
    robot.connect(calibrate=False)
    return robot


def get_bus(robot):
    """The motor bus lives under different attribute names across versions."""
    for attr in ("bus", "arm", "motors"):
        bus = getattr(robot, attr, None)
        if bus is not None and hasattr(bus, "sync_read"):
            return bus
    raise RuntimeError(
        "Could not find a motor bus with sync_read on the robot object. "
        f"Attributes available: {sorted(vars(robot))}"
    )


def probe(bus):
    """Return the register names this bus can actually read."""
    available = []
    for name in CANDIDATE_REGISTERS:
        try:
            value = bus.sync_read(name)
        except Exception as exc:  # noqa: BLE001 - we want to report, not crash
            print(f"  {name:22s} unavailable ({type(exc).__name__}: {exc})")
            continue
        available.append(name)
        print(f"  {name:22s} OK  {value}")
    return available


def sample(bus, registers, seconds, hz):
    """Collect readings for `seconds`, returned as {f'{motor}.{register}': [values]}."""
    series = {}
    deadline = time.time() + seconds
    period = 1.0 / hz
    while time.time() < deadline:
        for register in registers:
            try:
                reading = bus.sync_read(register)
            except Exception as exc:  # noqa: BLE001
                print(f"  read failed on {register}: {exc}", file=sys.stderr)
                continue
            for motor, value in reading.items():
                series.setdefault(f"{motor}.{register}", []).append(value)
        time.sleep(period)
    return series


def summarize(series):
    """Mean and standard deviation per channel."""
    summary = {}
    for channel, values in series.items():
        if not values:
            continue
        summary[f"{channel}.mean"] = statistics.fmean(values)
        summary[f"{channel}.std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return summary


def run_logging(bus, registers, args):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []

    print(
        "\nHang a weight, let it settle, read the PESOLA, type the grams and press "
        "Enter.\nEmpty input finishes and writes the file.\n"
    )
    while True:
        raw = input("grams (blank to finish): ").strip()
        if not raw:
            break
        try:
            grams = float(raw)
        except ValueError:
            print("  not a number, try again")
            continue

        print(f"  sampling {args.seconds:.1f}s ...", end="", flush=True)
        series = sample(bus, registers, args.seconds, args.hz)
        row = {"grams": grams, "direction": args.direction, "timestamp": time.time()}
        row.update(summarize(series))
        rows.append(row)

        preview = [
            f"{ch.rsplit('.', 1)[0]}={row[ch]:.1f}"
            for ch in sorted(row)
            if ch.endswith("Present_Load.mean")
        ]
        print(f" done  {'  '.join(preview) if preview else ''}")

    if not rows:
        print("nothing recorded")
        return

    fieldnames = sorted({key for row in rows for key in row})
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} points to {args.out}")


def run_fit(path):
    import numpy as np

    with open(path) as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 3:
        sys.exit(f"{path}: need at least 3 points to fit, found {len(rows)}")

    grams = np.array([float(row["grams"]) for row in rows])
    channels = [
        key
        for key in rows[0]
        if key.endswith(".mean") and ("Present_Load" in key or "Present_Current" in key)
    ]

    print(f"{len(rows)} points, {grams.min():.0f}–{grams.max():.0f} g\n")
    print(f"{'channel':34s} {'slope':>12s} {'R^2':>8s} {'g/count':>10s}")
    print("-" * 68)

    results = []
    for channel in channels:
        counts = np.array([float(row[channel]) for row in rows])
        if counts.std() == 0:
            continue
        slope, intercept = np.polyfit(counts, grams, 1)
        predicted = slope * counts + intercept
        ss_res = ((grams - predicted) ** 2).sum()
        ss_tot = ((grams - grams.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
        results.append((r2, channel, slope, intercept))

    for r2, channel, slope, intercept in sorted(results, reverse=True):
        name = channel.replace(".mean", "")
        print(f"{name:34s} {slope:12.4f} {r2:8.3f} {abs(slope):10.3f}")

    if results:
        r2, channel, slope, intercept = max(results)
        # grams = slope * counts + intercept is linear, so the N fit is the same line rescaled
        # (see lib/units.py and docs/contributing.md "Units": control code reads newtons, not grams).
        slope_n = grams_force_to_newtons(slope)
        intercept_n = grams_force_to_newtons(intercept)
        print(
            f"\nbest channel: {channel.replace('.mean', '')}\n"
            f"  grams   = {slope:.4f} * counts + {intercept:.2f}   (R^2 = {r2:.3f})\n"
            f"  newtons = {slope_n:.6f} * counts + {intercept_n:.4f}   <- what the controller reads"
        )
        print(
            "\nNow compare against the opposite-direction sweep: the gap between the "
            "two fits is the hysteresis, and it is your real force resolution."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="list readable registers and exit")
    parser.add_argument("--tare", action="store_true", help="sample the unloaded baseline and exit")
    parser.add_argument("--fit", metavar="CSV", help="fit a recorded sweep and exit (no robot needed)")
    parser.add_argument("--out", default="data/calibration.csv", help="where to write the sweep")
    parser.add_argument(
        "--direction",
        default="up",
        choices=("up", "down"),
        help="label the sweep direction, for hysteresis comparison",
    )
    parser.add_argument("--seconds", type=float, default=2.0, help="sampling window per point")
    parser.add_argument("--hz", type=float, default=20.0, help="sampling rate")
    parser.add_argument("--port", help="serial port (default: ROBOT_PORT from .env)")
    parser.add_argument("--id", default="rehab_follower", help="arm id used at calibration")
    args = parser.parse_args()

    if args.fit:
        run_fit(args.fit)
        return

    load_env()
    port = args.port or os.environ.get("ROBOT_PORT")
    if not port:
        sys.exit("No port: pass --port or set ROBOT_PORT in .env")

    robot = connect(port, args.id)
    try:
        bus = get_bus(robot)

        print("probing registers:")
        registers = probe(bus)
        if not registers:
            sys.exit("No readable registers - check the connection and the lerobot version.")
        if args.probe:
            return

        force_registers = [r for r in registers if r in ("Present_Load", "Present_Current")]
        if not force_registers:
            sys.exit("Neither Present_Load nor Present_Current is readable; no force signal.")

        if args.tare:
            print(f"\nbaseline, nothing attached, {args.seconds:.1f}s:")
            for channel, value in sorted(summarize(sample(bus, registers, args.seconds, args.hz)).items()):
                print(f"  {channel:40s} {value:10.2f}")
            return

        run_logging(bus, registers, args)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
