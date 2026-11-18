"""Log ST3215 servo current, via Station, while known forces are applied to arm B.

Used to calibrate servo current against grams read off the PESOLA spring scale.
See docs/force-calibration.md for the rig and the procedure, docs/elrobot.md for
what Station is and how to set up its Python client.

    python scripts/measure/log_force_calibration.py --probe --bus-serial <SERIAL>
    python scripts/measure/log_force_calibration.py --tare --bus-serial <SERIAL>
    python scripts/measure/log_force_calibration.py --out config/measurements/calib_up.csv --bus-serial <SERIAL>
    python scripts/measure/log_force_calibration.py --fit config/measurements/calib_up.csv

`--fit` reads a CSV and does not need Station or the norma-core client at all.
Everything else is async, because Station's client is.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `lib`
from lib.units import grams_force_to_newtons  # noqa: E402


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


async def probe(client, bus_serial, seconds=3.0):
    """List the motor ids this bus reports, with one current/temperature reading each."""
    from lib.station import follow_motor_state

    seen = {}
    deadline = time.monotonic() + seconds
    async for serial, motor_id, state in follow_motor_state(client):
        if serial == bus_serial:
            seen[motor_id] = state
        if time.monotonic() > deadline:
            break
    for motor_id in sorted(seen):
        s = seen[motor_id]
        print(f"  motor {motor_id}  current={s.current_ma} mA  temp={s.temperature_c} C  "
              f"range=[{s.range_min}, {s.range_max}]")
    return seen


async def sample(client, bus_serial, seconds, hz):
    """Collect readings for `seconds`, returned as {motor_id: {'current_ma': [...], ...}}."""
    from lib.station import follow_motor_state

    series: dict[int, dict[str, list]] = {}
    deadline = time.monotonic() + seconds
    min_period = 1.0 / hz
    last_sample_t: dict[int, float] = {}

    async for serial, motor_id, state in follow_motor_state(client):
        if serial != bus_serial:
            if time.monotonic() > deadline:
                break
            continue
        now = time.monotonic()
        if now - last_sample_t.get(motor_id, 0.0) < min_period:
            if now > deadline:
                break
            continue
        last_sample_t[motor_id] = now
        channels = series.setdefault(motor_id, {"current_ma": [], "temperature_c": []})
        if state.current_ma is not None:
            channels["current_ma"].append(state.current_ma)
        if state.temperature_c is not None:
            channels["temperature_c"].append(state.temperature_c)
        if now > deadline:
            break
    return series


def summarize(series: dict[int, dict[str, list]]) -> dict[str, float]:
    """Mean and standard deviation per motor and channel, flattened to CSV-friendly keys."""
    summary = {}
    for motor_id, channels in series.items():
        for name, values in channels.items():
            if not values:
                continue
            summary[f"motor{motor_id}.{name}.mean"] = statistics.fmean(values)
            summary[f"motor{motor_id}.{name}.std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return summary


async def run_logging(client, bus_serial, args):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []

    print(
        "\nHang a weight, let it settle, read the PESOLA, type the grams and press "
        "Enter.\nEmpty input finishes and writes the file.\n"
    )
    while True:
        raw = await asyncio.to_thread(input, "grams (blank to finish): ")
        raw = raw.strip()
        if not raw:
            break
        try:
            grams = float(raw)
        except ValueError:
            print("  not a number, try again")
            continue

        print(f"  sampling {args.seconds:.1f}s ...", end="", flush=True)
        series = await sample(client, bus_serial, args.seconds, args.hz)
        row = {"grams": grams, "direction": args.direction, "timestamp": time.time()}
        row.update(summarize(series))
        rows.append(row)

        preview = [
            f"motor{mid}={row[ch]:.1f}"
            for ch in sorted(row)
            if ch.endswith(".current_ma.mean")
            for mid in [ch.split(".")[0].removeprefix("motor")]
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
    channels = [key for key in rows[0] if key.endswith(".current_ma.mean")]

    print(f"{len(rows)} points, {grams.min():.0f}-{grams.max():.0f} g\n")
    print(f"{'channel':20s} {'slope':>12s} {'R^2':>8s} {'g/mA':>10s}")
    print("-" * 54)

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
        name = channel.replace(".current_ma.mean", "")
        print(f"{name:20s} {slope:12.4f} {r2:8.3f} {abs(slope):10.3f}")

    if results:
        r2, channel, slope, intercept = max(results)
        # grams = slope * mA + intercept is linear, so the N fit is the same line rescaled
        # (see lib/units.py and docs/contributing.md "Units": control code reads newtons, not grams).
        slope_n = grams_force_to_newtons(slope)
        intercept_n = grams_force_to_newtons(intercept)
        print(
            f"\nbest channel: {channel.replace('.current_ma.mean', '')}\n"
            f"  grams   = {slope:.4f} * mA + {intercept:.2f}   (R^2 = {r2:.3f})\n"
            f"  newtons = {slope_n:.6f} * mA + {intercept_n:.4f}   <- what the controller reads"
        )
        print(
            "\nNow compare against the opposite-direction sweep: the gap between the "
            "two fits is the hysteresis, and it is your real force resolution."
        )


async def main_async(args):
    from lib.station import connect

    load_env()
    host = args.station_host or os.environ.get("STATION_HOST", "localhost")
    bus_serial = args.bus_serial or os.environ.get("FOLLOWER_BUS_SERIAL")
    if not bus_serial:
        sys.exit("No bus serial: pass --bus-serial or set FOLLOWER_BUS_SERIAL in .env "
                  "(find it on the bus card in Station's web UI)")

    print(f"connecting to Station at {host} ...")
    client = await connect(host)
    print(f"connected. bus: {bus_serial}")

    if args.probe:
        await probe(client, bus_serial)
        return

    if args.tare:
        print(f"\nbaseline, nothing attached, {args.seconds:.1f}s:")
        series = await sample(client, bus_serial, args.seconds, args.hz)
        for channel, value in sorted(summarize(series).items()):
            print(f"  {channel:30s} {value:10.2f}")
        return

    await run_logging(client, bus_serial, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", action="store_true", help="list motor ids and current readings, then exit")
    parser.add_argument("--tare", action="store_true", help="sample the unloaded baseline and exit")
    parser.add_argument("--fit", metavar="CSV", help="fit a recorded sweep and exit (no Station needed)")
    parser.add_argument("--out", default="data/calibration.csv", help="where to write the sweep")
    parser.add_argument(
        "--direction", default="up", choices=("up", "down"),
        help="label the sweep direction, for hysteresis comparison",
    )
    parser.add_argument("--seconds", type=float, default=2.0, help="sampling window per point")
    parser.add_argument("--hz", type=float, default=20.0, help="sampling rate")
    parser.add_argument("--station-host", help="Station address (default: STATION_HOST from .env, else localhost)")
    parser.add_argument("--bus-serial", help="which bus (default: FOLLOWER_BUS_SERIAL from .env)")
    args = parser.parse_args()

    if args.fit:
        run_fit(args.fit)
        return

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
