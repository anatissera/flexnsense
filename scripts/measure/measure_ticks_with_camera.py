"""Measure ticks_per_metre and yield_sign using the camera instead of a ruler.

    .venv/bin/python scripts/measure/measure_ticks_with_camera.py --motor 4

No ruler, no typing numbers between moves: the camera reads the PESOLA and the tether's known
stiffness converts the force change into the displacement that caused it, `dx = dF / k`. The
operator rigs the tether once and watches; the script does the rest.

Why this exists alongside `measure_ticks_per_metre.py`, which uses a ruler: that one is the
more accurate of the two and stays the reference. The tether's hysteresis (6 mm worst case,
`config/measurements/tether_stiffness.csv`) is a floor on any displacement inferred from force, which is why
the ruler method was chosen originally. This one trades that accuracy for not needing a person
to measure and type a reading at every step, which matters when Station is dropping commands
and a run has to be repeated several times.

Read the result as a starting value to get the controller moving, not as a final calibration.
It prints the inherited uncertainty alongside the number so that is not forgettable.

Two things it does that the ruler version does not, both learned on the bench:

  * **Retries.** Station intermittently takes its full 30 second timeout on a command that
    normally returns instantly, and a six-trial run that dies on trial one wastes the whole
    rig setup. Each command is retried, and a trial that still fails is skipped rather than
    ending the run.
  * **Checks the move actually happened.** A servo accepts a goal and silently stops short
    when it hits Station's current limit (`current-threshold` in station.yaml). Commanding
    +100 ticks and travelling 27 is a real case from this bench. The achieved tick delta is
    read back and used for the fit, so a short move is still a usable data point rather than a
    wrong one.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm1.lib import bench_calibration_io  # noqa: E402
from arm1.lib.env import env_get  # noqa: E402
from arm1.lib.ticks_calibration import fit_ticks_per_metre, infer_yield_sign  # noqa: E402
from arm1.lib.units import grams_force_to_newtons, newtons_to_grams_force  # noqa: E402

GAUGE_CALIB = "config/scale_calib.json"


async def read_position(client, bus_serial: str, motor_id: int, seconds: float):
    """Latest settled position for one motor, or None if the stream stays quiet."""
    from arm1.lib.station import follow_motor_state

    latest = None
    deadline = time.monotonic() + seconds
    async for serial, mid, state in follow_motor_state(client):
        if serial == bus_serial and mid == motor_id and state.position is not None:
            latest = state.position
        if time.monotonic() > deadline:
            break
    return latest


async def command_with_retry(client, bus_serial: str, motor_id: int, target: int,
                             attempts: int = 3) -> bool:
    """Send a goal position, retrying a timeout. True if Station accepted it.

    Station's own client raises ErrRequestTimeout after 30 seconds with no acknowledgement,
    and on this bench that happens intermittently to commands that otherwise return in
    milliseconds. Retrying is safe here because the goal is absolute, not relative: sending
    the same target twice asks for the same position, never for twice the motion.
    """
    from arm1.lib.station import set_position

    for attempt in range(1, attempts + 1):
        try:
            await set_position(client, bus_serial, motor_id, target)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"    command attempt {attempt}/{attempts} failed "
                  f"({type(exc).__name__}); {'retrying' if attempt < attempts else 'giving up'}")
            if attempt < attempts:
                await asyncio.sleep(1.0)
    return False


def read_force_n(pipeline, source, samples: int, settle_s: float) -> float | None:
    """Median force over several frames, in newtons, or None if the scale was not readable.

    A median rather than one frame: the gauge reader reports a few grams of noise frame to
    frame, and a single bad frame at either end of a trial would land directly in the fit.
    """
    import numpy as np

    time.sleep(settle_s)
    values = []
    deadline = time.monotonic() + 5.0
    while len(values) < samples and time.monotonic() < deadline:
        ok, frame, ts, _ = source.read()
        if not ok:
            continue
        result = pipeline.process(frame, ts)
        if result.found and result.newtons is not None:
            values.append(result.newtons)
    if len(values) < max(3, samples // 2):
        return None
    return float(np.median(values))


async def run(args) -> int:
    from arm1.lib.camera import open_source
    from arm1.lib.station import connect

    calib = bench_calibration_io.load()
    k_n_per_m = calib.get("tether_n_per_m")
    if not k_n_per_m:
        raise SystemExit(
            "tether_n_per_m is not measured, and this method is built on it: the camera gives "
            "a force change and the stiffness turns it into a distance.\n\n"
            "    .venv/bin/python scripts/measure/measure_tether_stiffness.py\n\n"
            "Or use the ruler method instead: scripts/measure/measure_ticks_per_metre.py"
        )
    hysteresis_m = calib.get("displacement_uncertainty_m")

    if not Path(args.calib).is_file():
        raise SystemExit(f"no gauge calibration at {args.calib}; run "
                         "`.venv/bin/python -m gauge_reader calibrate` first.")

    bus_serial = args.bus_serial or env_get("FOLLOWER_BUS_SERIAL")
    if not bus_serial:
        raise SystemExit("no follower bus serial: set FOLLOWER_BUS_SERIAL in .env.")
    host = args.station_host or env_get("STATION_HOST", "localhost")

    print("This moves the arm against the tether. Nothing may be attached to a person.")
    print(f"Tether stiffness in use: {k_n_per_m:.1f} N/m.")
    if input("Type 'no finger attached' to confirm: ").strip().lower() != "no finger attached":
        print("aborted; nothing moved.")
        return 1

    from arm1.gauge_reader.cli import _load_calibration, make_pipeline

    pipeline = make_pipeline(_load_calibration(args.calib))
    source = open_source(camera=args.camera)
    client = await connect(host)

    baseline = await read_position(client, bus_serial, args.motor, args.seconds)
    if baseline is None:
        source.release()
        raise SystemExit(f"no position reading for motor {args.motor}; is the bus connected?")
    print(f"\nbaseline position: {baseline} ticks")

    baseline_force = read_force_n(pipeline, source, args.samples, args.settle)
    if baseline_force is None:
        source.release()
        raise SystemExit("the camera cannot read the scale. Check the markers are visible.")
    print(f"baseline force: {newtons_to_grams_force(baseline_force):.0f} gf")

    tick_points: list[tuple[int, float]] = []
    force_trials: list[tuple[int, float, float]] = []
    ceiling_n = grams_force_to_newtons(args.max_grams)

    try:
        for delta in args.ticks:
            target = baseline + delta
            print(f"\n{delta:+d} ticks -> {target}")

            if not await command_with_retry(client, bus_serial, args.motor, target,
                                            args.attempts):
                print("  Station would not take the command; skipping this delta")
                continue

            achieved = await read_position(client, bus_serial, args.motor, args.settle)
            if achieved is None:
                print("  no position readback; skipping this delta")
                await command_with_retry(client, bus_serial, args.motor, baseline, args.attempts)
                continue
            achieved_delta = achieved - baseline
            if achieved_delta != delta:
                # Not a failure: Station's current limit stops the servo short when the tether
                # resists. The achieved delta is the honest input to the fit.
                print(f"  moved {achieved_delta:+d} of {delta:+d} ticks "
                      "(stopped short, probably the current limit)")

            force = read_force_n(pipeline, source, args.samples, args.settle)
            if force is None:
                print("  lost the scale in the camera; skipping this delta")
            elif abs(achieved_delta) < args.min_ticks:
                print(f"  moved only {achieved_delta:+d} ticks, under the {args.min_ticks} "
                      "minimum; too small to measure against, skipping")
            else:
                dx_m = (force - baseline_force) / k_n_per_m
                print(f"  force {newtons_to_grams_force(force):.0f} gf "
                      f"({newtons_to_grams_force(force - baseline_force):+.0f} gf) "
                      f"-> moved {dx_m * 1000:+.1f} mm")
                tick_points.append((achieved_delta, dx_m * 1000.0))
                force_trials.append((achieved_delta,
                                     newtons_to_grams_force(baseline_force),
                                     newtons_to_grams_force(force)))
                if force > ceiling_n:
                    print(f"  force is over the {args.max_grams:.0f} gf ceiling; stopping")
                    break

            await command_with_retry(client, bus_serial, args.motor, baseline, args.attempts)
            await asyncio.sleep(args.settle)
    finally:
        print(f"\nreturning motor {args.motor} to {baseline} ...")
        if not await command_with_retry(client, bus_serial, args.motor, baseline, args.attempts):
            print(f"!! could not return motor {args.motor} to baseline. Release torque by hand: "
                  f"`python scripts/arm/lock_arm.py --release --motors {args.motor}`")
        source.release()

    if len(tick_points) < 2:
        print(f"\nonly {len(tick_points)} usable trial(s); need at least 2 to fit.")
        return 1

    fit = fit_ticks_per_metre(tick_points)
    sign = infer_yield_sign(force_trials)

    print(f"\nticks_per_metre: {fit.ticks_per_metre:.0f}  (from {fit.n_points} trials, "
          f"worst trial {fit.max_relative_deviation*100:.0f}% off the fit)")
    print(f"yield_sign: {sign:+d}  (the tick direction that lowers the force)")

    if hysteresis_m:
        span_m = max(abs(mm) for _, mm in tick_points) / 1000.0
        if span_m > 0:
            print(f"\nInherited uncertainty: the tether's {hysteresis_m*1000:.0f} mm hysteresis "
                  f"over a {span_m*1000:.0f} mm move is about "
                  f"{hysteresis_m/span_m*100:.0f}% on this number. The ruler method "
                  "(scripts/measure/measure_ticks_per_metre.py) does not carry it.")

    if args.save:
        bench_calibration_io.save({
            "ticks_per_metre": round(fit.ticks_per_metre, 1),
            "yield_sign": sign,
        })
        print(f"\nsaved to {bench_calibration_io.DEFAULT_PATH}")
    else:
        print("\n(--save to write these to the bench calibration)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motor", type=int, required=True)
    p.add_argument("--ticks", type=lambda s: [int(v) for v in s.split(",")],
                   default=[150, 300, 450, -150, -300, -450],
                   help="tick deltas to try (default 150,300,450 and their negatives)")
    p.add_argument("--min-ticks", type=int, default=40,
                   help="ignore a trial that moved less than this; too small to measure")
    p.add_argument("--max-grams", type=float, default=450.0,
                   help="stop if the force passes this (75 percent of the PESOLA's range)")
    p.add_argument("--attempts", type=int, default=3,
                   help="retries per command, for Station's intermittent timeouts")
    p.add_argument("--samples", type=int, default=9,
                   help="camera frames to median per force reading")
    p.add_argument("--settle", type=float, default=1.5)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--camera", default=None)
    p.add_argument("--calib", default=GAUGE_CALIB)
    p.add_argument("--station-host", default=None)
    p.add_argument("--bus-serial", default=None)
    p.add_argument("--save", action="store_true")
    args = p.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
