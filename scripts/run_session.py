"""Run the admittance controller against the real arm, from the terminal.

This is the piece that was missing between "every layer has tests" and "you can try it".
Nothing had ever fed a real camera and a real arm to the same loop.

    .venv/bin/python scripts/run_session.py --motor 4 --finger index --cap-gf 100 --reps 3

Each cycle: read the camera, measure the PESOLA through the same pipeline as
`python -m gauge_reader run`, pass the force to `control.safety.SafetyGate` and
`session.reps.RepMachine`, and when the gate says YIELD, move the arm by the step
`control.admittance` asks for. Printed live, one line per change.

## Three things have to be true before this can run, and it checks all three

1. **Station running**, with the follower bus reachable. `STATION_HOST` and
   `FOLLOWER_BUS_SERIAL` come from `.env`.
2. **The gauge reader calibrated on this machine** (`python -m gauge_reader calibrate`). Without
   it there is no force reading at all, and a controller with no force input is the one thing
   this project must never run.
3. **`ticks_per_metre` and `yield_sign` measured** (`scripts/measure_ticks_per_metre.py`, see
   docs/bench-measurements.md). The control law decides in metres and the servo takes ticks;
   without the conversion there is nothing to command. `--ticks-per-metre` overrides it for a
   first rough trial, and says so loudly, because a wrong value means the arm moves a different
   distance than the law intended.

## What it will not do

It will not start with the force unreadable, it will not keep going when the gate latches
RELEASE, and on exit, including Ctrl-C and a crash, it unloads the spring before stopping.
The arm giving way is the only motion it ever commands: there is no path in here that
tightens the tether.

Start with a low cap and few reps. `--cap-gf 100` is half the clinical 200, and the first
thing worth confirming is not that the controller is good but that yielding actually loosens
the tether, which is `yield_sign` being right in the real pose.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.admittance import AdmittanceParams  # noqa: E402
from control.loop import AxisCalibration, ControlLoop  # noqa: E402
from control.safety import Action, SafetyGate, SafetyLimits  # noqa: E402
from lib.env import env_get  # noqa: E402
from lib.units import grams_force_to_newtons, newtons_to_grams_force  # noqa: E402
from session.calibration import BenchCalibration, NotMeasured  # noqa: E402
from session.modes import MODES  # noqa: E402
from session.protocol import Block  # noqa: E402
from session.reps import RepInput, RepLimits, RepMachine  # noqa: E402

BENCH_CALIB = "config/calibration/bench.json"
GAUGE_CALIB = "config/scale_calib.json"


def build_pipeline(calib_path: str):
    """The same pipeline `python -m gauge_reader run` uses, so readings match what you see there."""
    from gauge_reader.cli import _load_calibration, make_pipeline

    if not Path(calib_path).is_file():
        raise SystemExit(
            f"no gauge calibration at {calib_path}.\n\n"
            "The gauge reader has to be calibrated on this machine first, because it depends on "
            "where this camera sits:\n\n"
            "    .venv/bin/python -m gauge_reader calibrate\n\n"
            "Without it there is no force reading, and this will not run a controller blind."
        )
    return make_pipeline(_load_calibration(calib_path))


def resolve_axis(args) -> AxisCalibration:
    """ticks_per_metre and yield_sign, from the bench calibration or an explicit override."""
    if args.ticks_per_metre is not None:
        sign = args.yield_sign
        if sign is None:
            raise SystemExit(
                "--ticks-per-metre needs --yield-sign too: +1 or -1, whichever direction of tick "
                "change makes the force go DOWN. Guessing it wrong means the arm tightens the "
                "tether when it means to release, which is the one direction this must never go."
            )
        print("!! using an UNMEASURED ticks_per_metre override. The arm will move a different\n"
              "!! distance than the control law intends, by whatever factor this is wrong by.\n"
              "!! Keep the cap low and stop at the first thing that looks odd.\n")
        return AxisCalibration(motor_id=args.motor, ticks_per_metre=args.ticks_per_metre, sign=sign)

    calib = BenchCalibration.load(BENCH_CALIB)
    try:
        ticks_per_metre, sign = calib.require_axis()
    except NotMeasured as exc:
        raise SystemExit(
            f"{exc}\n\n"
            "That measurement is itself the first arm test, and it is the safer one: it moves "
            "the arm in known steps with nothing attached to a person.\n\n"
            "    .venv/bin/python scripts/measure_ticks_per_metre.py --motor <id> "
            "--ticks 30,60,90,-30,-60,-90\n\n"
            "To try this controller before measuring, pass --ticks-per-metre and --yield-sign "
            "explicitly and accept that the distances are wrong by an unknown factor."
        ) from None
    return AxisCalibration(motor_id=args.motor, ticks_per_metre=ticks_per_metre, sign=sign)


async def read_positions(client, bus_serial: str, motor_id: int, state: dict) -> None:
    """Keep `state['ticks']` fresh from the motor state stream."""
    from lib.station import follow_motor_state

    async for bus, mid, motor in follow_motor_state(client):
        if bus == bus_serial and mid == motor_id and motor.position is not None:
            state["ticks"] = motor.position
            state["at"] = time.monotonic()


async def run(args) -> int:
    from lib.station import connect

    block = Block(
        finger=args.finger,
        mode=args.mode,
        reps=args.reps,
        cap_n=grams_force_to_newtons(args.cap_gf),
        hold_s=args.hold,
        rom_target_m=(args.rom_mm / 1000.0) if MODES[args.mode].needs_rom_target else None,
    )
    bench = BenchCalibration.load(BENCH_CALIB)
    for problem in block.check_against(bench):
        print(f"warning: {problem}\n")

    axis = resolve_axis(args)
    machine = RepMachine(block, RepLimits())
    limits = SafetyLimits(
        f_threshold_n=block.cap_n,
        f_max_n=grams_force_to_newtons(args.ceiling_gf),
        max_travel_m=args.max_travel_mm / 1000.0,
    )
    loop = ControlLoop(
        limits=limits,
        params=AdmittanceParams(
            damping_n_s_per_m=args.damping,
            v_max_m_s=args.v_max_mm_s / 1000.0,
            deadband_n=args.deadband_n,
            max_step_m=args.max_step_mm / 1000.0,
        ),
        axis=axis,
        gate=SafetyGate(limits),
    )
    pipeline = build_pipeline(args.calib)

    host = args.station_host or env_get("STATION_HOST", "localhost")
    bus_serial = args.bus_serial or env_get("FOLLOWER_BUS_SERIAL")
    if not bus_serial:
        raise SystemExit("no follower bus serial: set FOLLOWER_BUS_SERIAL in .env or pass "
                         "--bus-serial.")

    print(f"\n{block.reps} x {block.mode} on the {block.finger}")
    print(f"  cap      {block.cap_gf:.0f} gf ({block.cap_n:.2f} N), ceiling {args.ceiling_gf:.0f} gf")
    print(f"  resting  {newtons_to_grams_force(machine.pretension_n):.0f} gf")
    print(f"  arm      motor {axis.motor_id} on {bus_serial}, "
          f"{axis.ticks_per_metre:.0f} ticks/m, yield sign {axis.sign:+d}")
    print(f"  travel   {args.max_travel_mm:.0f} mm limit, {args.max_step_mm:.1f} mm max step\n")
    print("The arm will MOVE, in the direction that releases tension. Keep a hand on the e-stop.")
    if input("Type 'yes' to run: ").strip().lower() != "yes":
        print("aborted; nothing moved.")
        return 1

    from lib.camera import open_source

    source = open_source(camera=args.camera)
    state: dict = {"ticks": None, "at": 0.0}
    client = await connect(host=host)
    follower = asyncio.create_task(read_positions(client, bus_serial, axis.motor_id, state))
    started = time.monotonic()
    last_line = ""

    print(f"\n{'force':>9s}  {'phase':9s}  {'gate':8s}  {'step':>7s}  reps   instruction")
    print("-" * 86)
    try:
        while True:
            now = time.monotonic()
            if now - started > args.max_seconds:
                print("\nout of time")
                break
            ok, frame, ts, _ = source.read()
            if not ok:
                continue
            result = pipeline.process(frame, ts)
            force_n = result.newtons if result.found else None

            step = machine.step(RepInput(
                force_n=force_n,
                reading_age_s=max(0.0, time.monotonic() - ts),
                travel_m=loop.state.travel_m,
                now=now,
                confident=bool(result.found),
            ))
            decision = await loop.step(
                client, bus_serial, force_n, ts, state["ticks"], now=now
            )

            gf = "   --   " if force_n is None else f"{newtons_to_grams_force(force_n):6.0f} gf"
            moved = loop.state.records[-1].step_m * 1000 if loop.state.records else 0.0
            line = (f"{gf:>9s}  {step.phase.value:9s}  {decision.action.value:8s}  "
                    f"{moved:6.2f}mm  {step.reps_completed}/{block.reps}    {step.instruction}")
            if line != last_line:
                print(line)
                last_line = line

            if decision.action is Action.RELEASE:
                print(f"\n!! gate released: {decision.reason}")
                machine.abandon(decision.reason)
                break
            if step.finished:
                print(f"\n{step.note}")
                break
    except KeyboardInterrupt:
        print("\nstopped by keyboard")
    finally:
        # Unload before anything else, on every path. A stuck arm holding tension against a
        # finger with no force reading is the worst state this machine has.
        print("releasing tension ...")
        try:
            await loop.release(client, bus_serial)
        except Exception as exc:  # noqa: BLE001
            print(f"!! could not release ({exc}). The arm may still be under tension: release "
                  f"torque by hand, `python scripts/lock_arm.py --release --motors {axis.motor_id}`, "
                  "or power down Station.")
        follower.cancel()
        source.release()

    print(f"\ncompleted {machine.reps_completed} of {block.reps}, "
          f"{machine.attempts_abandoned} abandoned, "
          f"arm travelled {loop.state.travel_m * 1000:.1f} mm")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--motor", type=int, required=True,
                   help="the joint that moves along the tether axis. Rig knowledge; see "
                        "docs/bench-measurements.md for how to identify it safely")
    p.add_argument("--finger", default="index")
    p.add_argument("--mode", default="active", choices=sorted(MODES))
    p.add_argument("--cap-gf", type=float, default=100.0,
                   help="force cap in grams-force. Start at 100, not the clinical 200")
    p.add_argument("--ceiling-gf", type=float, default=200.0,
                   help="hard ceiling: the gate latches RELEASE above this")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--hold", type=float, default=2.0)
    p.add_argument("--rom-mm", type=float, default=15.0, help="passive mode only")
    p.add_argument("--damping", type=float, default=15.0,
                   help="B in N.s/m; the simulation says 10-20 for this tether")
    p.add_argument("--v-max-mm-s", type=float, default=20.0)
    p.add_argument("--deadband-n", type=float, default=0.05)
    p.add_argument("--max-step-mm", type=float, default=1.0,
                   help="largest move in one cycle. Keep small for a first run")
    p.add_argument("--max-travel-mm", type=float, default=40.0,
                   help="total travel before the gate releases")
    p.add_argument("--ticks-per-metre", type=float, default=None,
                   help="override the measured value for a first rough trial")
    p.add_argument("--yield-sign", type=int, default=None, choices=[1, -1],
                   help="with --ticks-per-metre: which tick direction lowers the force")
    p.add_argument("--camera", default=None)
    p.add_argument("--calib", default=GAUGE_CALIB)
    p.add_argument("--station-host", default=None)
    p.add_argument("--bus-serial", default=None)
    p.add_argument("--max-seconds", type=float, default=180.0)
    args = p.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
