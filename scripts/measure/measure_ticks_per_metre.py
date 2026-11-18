"""Measure ticks_per_metre and yield_sign on the bench: a ruler against the arm, nothing else.

Fills `ticks_per_metre` and `yield_sign` in `config/calibration/bench.json` (shaped like
`session.calibration.BenchCalibration.to_dict()`; see lib/bench_calibration_io.py for why this
writes those keys by hand instead of importing that class). `control.loop.AxisCalibration`
consumes both once set.

No finger anywhere near this: it moves the arm under tension with nobody protecting a healing
finger from it, so the script requires a separate, explicit confirmation of that before any
bus serial or Station connection is even asked for, independent of the per-move 'yes' prompts.

Why a ruler, not the camera (`dx = dF / k`, no ruler needed): tempting, but the tether's own
hysteresis (`lib.stiffness.compare_directions`, `config/measurements/tether_stiffness.csv`: 6 mm worst case) is
a floor on how well any force-inferred displacement can be known, and it dominates at every
tick delta small enough to stay under the force cap - even a 60 mm move, most of the tether's
usable travel, is only good to 10%. A ruler reading is good to about its own precision
(roughly 1 mm) regardless of how far the arm moves, which beats the force method at every
displacement actually reachable here. Do NOT go back to dx = dF / k: an earlier version of this
procedure recommended it, and the numbers above are why that was wrong.

Setup, needs the real arm:

1. Lock the arm in the exact session pose: `python scripts/arm/lock_arm.py --motors <id>`.
2. Anchor the PESOLA between the gripper (or whatever the tether attaches to) and a fixed point,
   with some pre-tension, so the tether stays taut through a small move without going slack.
   Start the pre-tension low (tens of gf): a bigger tick delta is what gives the ruler reading
   room to be precise, and it will add force on top of whatever you start with.
3. Pick two reference marks, one on the part the motor moves, one on something fixed nearby.
4. Run this script. It reads the motor's current position once (the baseline), then for each
   tick delta you pass: commands baseline + delta, waits for it to settle, and asks for the new
   distance between the marks and the new force reading (read either off the PESOLA or off
   gauge_reader's live overlay - exact precision does not matter here, this is only used to
   tell which direction reduces force). It returns to baseline between deltas.
5. ticks_per_metre comes out of the ruler readings alone. yield_sign comes out of whichever
   delta direction made the force go down.

    python scripts/measure/measure_ticks_per_metre.py --motor 3 --ticks 30,60,90,-30,-60,-90
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `lib`

from lib import bench_calibration_io  # noqa: E402
from lib.ticks_calibration import fit_ticks_per_metre, infer_yield_sign  # noqa: E402
from lib.units import grams_force_to_newtons  # noqa: E402

DEFAULT_TICKS = [30, 60, 90, -30, -60, -90]
#: Order-of-magnitude only, for the force-change preview before a move - not used in the fit.
DEFAULT_TICKS_PER_METRE_GUESS = 3300.0

#: Not the clinical cap (session.Protocol.ceiling_n, ~300 gf): that number exists because a
#: healing finger is in the loop, and this measurement has none - its own instructions require
#: an explicit confirmation of that before anything connects. The binding constraint here is the
#: PESOLA's own range: 450 gf is 75% of its 600 g full scale, leaving 150 gf of headroom above
#: the highest level this script will ever ask for, for whatever overshoots a commanded move.
DEFAULT_MAX_GRAMS = 450.0
DEFAULT_SETTLE_SECONDS = 1.5


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


async def read_settled_state(client, bus_serial: str, motor_id: int, seconds: float):
    """Listen for `seconds` and return the most recent MotorState seen for this motor.

    Not the first reading: the servo may still be moving when the stream happens to deliver one,
    so this keeps whatever arrived last in the window, which is the settled value if `seconds`
    was long enough to cover the move.
    """
    from lib.station import follow_motor_state

    latest = None
    deadline = time.monotonic() + seconds
    async for serial, mid, state in follow_motor_state(client):
        if serial == bus_serial and mid == motor_id and state.position is not None:
            latest = state
        if time.monotonic() > deadline:
            break
    return latest


def preview_force_change(delta_ticks: int, guess_ticks_per_metre: float, k_n_per_m: float | None) -> str:
    if k_n_per_m is None:
        return ""
    dx_m = delta_ticks / guess_ticks_per_metre
    df_n = abs(dx_m) * k_n_per_m
    df_gf = df_n / grams_force_to_newtons(1.0)
    return f" (rough estimate: about {df_gf:.0f} gf of change, using tether_n_per_m={k_n_per_m:.1f})"


async def safe_return_to_baseline(client, bus_serial: str, motor_id: int, baseline_position: int) -> None:
    """Send the motor back to baseline, unconditionally - no reading, no prompt, nothing that
    can fail for the same reason this is being called (a bad keystroke, Ctrl-C). Call this from
    a `finally`, so it runs on the normal path and on any exception alike.

    If the move itself fails there is no further software fallback: the arm may now be at an
    unknown position, still under tension, with the process about to exit. Print something a
    person standing at the bench can act on by hand, not a traceback.
    """
    from lib.station import set_position

    try:
        await set_position(client, bus_serial, motor_id, baseline_position)
    except Exception as exc:
        print(f"\n!! could not return motor {motor_id} to baseline ({baseline_position} ticks): "
              f"{exc}")
        print(f"!! the arm is at an unknown position and may still be under tension - do not "
              f"leave it. Release torque by hand: "
              f"`python scripts/arm/lock_arm.py --release --motors {motor_id}`, "
              f"or power down Station if that does not respond.")


async def run(args) -> int:
    from lib.station import connect, set_position

    print("This measurement moves the arm under tension with --max-grams allowed up to "
          f"{args.max_grams:g} gf, well above the clinical cap - that ceiling is only safe "
          "because nothing is attached to a person (see the module docstring).")
    if input("Type 'no finger attached' to confirm there is none, anywhere near the rig: "
              ).strip().lower() != "no finger attached":
        print("Not confirmed; stopping without connecting to Station.")
        return 1

    load_env()
    host = args.station_host or os.environ.get("STATION_HOST", "localhost")
    bus_serial = args.bus_serial or os.environ.get("FOLLOWER_BUS_SERIAL")
    if not bus_serial:
        sys.exit("No bus serial: pass --bus-serial or set FOLLOWER_BUS_SERIAL in .env")

    calibration = bench_calibration_io.load(args.calibration_out) if not args.no_write else {}
    k_n_per_m = calibration.get("tether_n_per_m")

    print(f"connecting to Station at {host} ...")
    client = await connect(host)

    print(f"reading current position of motor {args.motor} on bus {bus_serial} ...")
    baseline = await read_settled_state(client, bus_serial, args.motor, args.seconds)
    if baseline is None:
        sys.exit(f"error: no position reading for motor {args.motor}; bus not responding?")
    baseline_position = baseline.position
    print(f"  baseline position: {baseline_position} ticks "
          f"(range {baseline.range_min}-{baseline.range_max})")

    try:
        baseline_mm = float(input("\nbaseline distance between the marks, in mm: ").strip())
        baseline_grams = float(
            input("baseline force (gf), read off the PESOLA or gauge_reader: ").strip()
        )
    except (ValueError, KeyboardInterrupt) as exc:
        print(f"\n{type(exc).__name__}: {exc}. Nothing has moved yet; exiting.")
        return 1

    tick_points: list[tuple[int, float]] = []
    force_trials: list[tuple[int, float, float]] = []

    for delta in args.ticks:
        target = baseline_position + delta
        if target < baseline.range_min or target > baseline.range_max:
            print(f"\nskipping delta {delta:+d}: target {target} is outside this motor's "
                  f"calibrated range ({baseline.range_min}-{baseline.range_max})")
            continue

        preview = preview_force_change(delta, args.guess_ticks_per_metre, k_n_per_m)
        print(f"\nabout to command motor {args.motor} to {target} ticks ({delta:+d} from "
              f"baseline){preview}.")
        if input("type 'yes' to proceed, anything else to skip: ").strip().lower() != "yes":
            print("  skipped")
            continue

        crashed = False
        exceeded_ceiling = False
        try:
            # set_position is inside the try on purpose, not before it: it timed out once in
            # practice (ErrRequestTimeout from Station's own client, after 30s with no ack),
            # and when that line was outside this block the exception skipped the finally
            # below entirely, so the one thing standing between "arm possibly mid-move, nobody
            # watching" and a safe state never ran. Whether the command actually reached the
            # servo or not, commanding the known baseline again is always safe.
            await set_position(client, bus_serial, args.motor, target)
            print(f"  waiting {args.settle:.1f}s to settle ...")
            state = await read_settled_state(client, bus_serial, args.motor, args.settle)
            achieved_position = state.position if state is not None else target
            achieved_delta_ticks = achieved_position - baseline_position
            if achieved_position != target:
                print(f"  note: settled at {achieved_position}, not the commanded {target}")

            mm_now = float(input("  distance between the marks now, in mm: ").strip())
            grams_now = float(input("  force now (gf): ").strip())

            tick_points.append((achieved_delta_ticks, mm_now - baseline_mm))
            force_trials.append((achieved_delta_ticks, baseline_grams, grams_now))

            if grams_now > args.max_grams:
                print(f"\nforce is now {grams_now:g} gf, over the safety ceiling "
                      f"({args.max_grams:g} gf).")
                exceeded_ceiling = True
        except (Exception, KeyboardInterrupt) as exc:
            crashed = True
            print(f"\n!! {type(exc).__name__} during this trial ({exc}); stopping.")
        finally:
            # Unconditional: this has to run whether the trial above finished cleanly, raised,
            # or was interrupted - it is the one thing standing between "arm under tension,
            # nobody watching" and a safe state, so it cannot itself depend on a reading.
            print(f"  returning to baseline ({baseline_position} ticks) ...")
            await safe_return_to_baseline(client, bus_serial, args.motor, baseline_position)

        if crashed:
            return 1
        if exceeded_ceiling:
            print("Stopping.")
            break

        await asyncio.sleep(args.settle)

    print(f"\n{len(tick_points)} usable trial(s).")
    if len(tick_points) < 2:
        print("not enough trials to fit ticks_per_metre (need at least 2).")
        return 1

    try:
        tick_fit = fit_ticks_per_metre(tick_points)
    except ValueError as exc:
        print(f"\nticks_per_metre fit failed: {exc}")
        return 1
    print(f"\nticks_per_metre = {tick_fit.ticks_per_metre:.1f} "
          f"(worst per-trial deviation {tick_fit.max_relative_deviation:.1%}, "
          f"{tick_fit.n_points} trials)")
    for (delta, mm), rate in zip(tick_points, tick_fit.per_point_ticks_per_metre):
        print(f"  {delta:+5d} ticks, {mm:+6.2f} mm -> {rate:8.1f} ticks/m")

    try:
        yield_sign = infer_yield_sign(force_trials)
    except ValueError as exc:
        print(f"\nyield_sign could not be determined: {exc}")
        return 1
    print(f"yield_sign = {yield_sign:+d} "
          f"({'positive' if yield_sign > 0 else 'negative'} tick deltas reduce force)")

    if args.no_write:
        print(f"\n--no-write: not touching {args.calibration_out}")
        return 0

    updates = {"ticks_per_metre": tick_fit.ticks_per_metre, "yield_sign": yield_sign}
    if args.pose:
        updates["measured_in_pose"] = args.pose
    bench_calibration_io.save(updates, args.calibration_out)
    print(f"\nwrote ticks_per_metre and yield_sign to {args.calibration_out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motor", type=int, required=True,
                   help="motor id that moves along the tether axis")
    p.add_argument("--ticks", default=",".join(str(t) for t in DEFAULT_TICKS),
                   help=f"comma-separated signed tick deltas to try (default {DEFAULT_TICKS})")
    p.add_argument("--max-grams", type=float, default=DEFAULT_MAX_GRAMS,
                   help=f"abort and return to baseline if force exceeds this, in gf "
                        f"(default {DEFAULT_MAX_GRAMS:g})")
    p.add_argument("--guess-ticks-per-metre", type=float, default=DEFAULT_TICKS_PER_METRE_GUESS,
                   help="order-of-magnitude guess, for the pre-move force preview only "
                        f"(default {DEFAULT_TICKS_PER_METRE_GUESS:g})")
    p.add_argument("--settle", type=float, default=DEFAULT_SETTLE_SECONDS,
                   help=f"seconds to wait for the servo to settle after a move "
                        f"(default {DEFAULT_SETTLE_SECONDS:g})")
    p.add_argument("--seconds", type=float, default=3.0,
                   help="how long to wait for the baseline position reading (default 3.0)")
    p.add_argument("--pose", help="free-text label for the session pose this was measured in")
    p.add_argument("--station-host", help="default: STATION_HOST from .env, else localhost")
    p.add_argument("--bus-serial", help="default: FOLLOWER_BUS_SERIAL from .env")
    p.add_argument("--calibration-out", default=bench_calibration_io.DEFAULT_PATH,
                   help=f"merge ticks_per_metre and yield_sign into this file "
                        f"(default %(default)s)")
    p.add_argument("--no-write", action="store_true", help="print only, don't touch --calibration-out")
    args = p.parse_args()
    args.ticks = [int(t) for t in args.ticks.split(",")]

    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
