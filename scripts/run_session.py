"""Run the admittance controller against the real arm, from the terminal.

This is the piece that was missing between "every layer has tests" and "you can try it".
Nothing had ever fed a real camera and a real arm to the same loop.

    .venv/bin/python scripts/run_session.py --motor 4 --finger index --cap-gf 100 --reps 3
    .venv/bin/python scripts/run_session.py --coordinated --finger index --cap-gf 100 --reps 3

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

## When no single joint follows the tether: --coordinated

A revolute joint moves the gripper along an arc, so "the motor that pulls along the tether" is
only ever an approximation, and on this arm it has not been a usable one in any pose tried: the
best-aligned joint reaches about 0.4 of the tether's direction, and the rest of its travel is
sideways. That is not a calibration problem to be measured away. It shows up as a bench
measurement that will not fit, because a long enough move crosses the tether's axis instead of
following it and the force stops rising monotonically with the tick delta.

`--coordinated` moves three joints together instead, choosing them for the live pose
(`lib.kinematics.Chain.choose_pull_motors`), and taking the tether's direction from the
gripper's own axis. It derives `ticks_per_metre` and `yield_sign` from that same geometry, so
it needs no bench measurement of either. It refuses to run if the pose cannot follow the pull
to the end of the travel limit. What it still does not know is backlash: geometry is not a
measurement, and the printed warning says so.

There is no `--step-mm` or `--settle` here, unlike `scripts/jog_coordinated.py`: this loop's
pace comes from the camera's frame rate and the admittance law's own velocity, not from a
fixed schedule, which is already "move at the speed the finger pulls" by construction.
`--coordinated-speed` and `--coordinated-accel` are the part that *does* carry over: the same
RAM_GOAL_SPEED/RAM_ACC scaling `jog_coordinated.py` uses to keep three joints landing
together.

## Where the defaults come from: config/session-tuning.json

Every machine-tuning flag below defaults to the committed profile in
`config/session-tuning.json`, not to a number written in this file, so a normal session needs
no tuning flags at all and the dashboard reads the same values this script does. The profile
holds what the bench actually proved over many sessions, and `session/tuning.py` says what
each number is paying for. `--tuning other.json` runs a different profile; any individual
flag still overrides it.

Three of those values are *looser* than what the code shipped with (the hard ceiling, the
blind-frame count, and the yield-direction check being off). They are grouped under `safety`
in the profile and printed on every run, because a loosened limit nobody is shown on the way
past stops being a decision and becomes an accident.

## What it will not do

It will not start with the force unreadable, it will not keep going when the gate latches
RELEASE, and on exit, including Ctrl-C and a crash, it unloads the spring before stopping.
By default the arm giving way is the only motion it ever commands. `--recover` adds exactly
one more: taking slack back up when the finger eases off, bounded two ways at once (see
`control.loop.ControlLoop` on `recover_gain` and `recover_floor_n`): it never ends up tighter
than where the session was rigged, and `--recover-floor-gf` stops it chasing the force down
to true slack, which on this bench once went far enough to lose the camera's reference
markers and trip a different watchdog (`session.reps.RepMachine`'s 300 ms deadman) than the
one `--max-dropouts` tunes.

`--recover` is a bounded stand-in for one specific case and is **not** the pull law: it is a
force-driven mirror of yielding, used in `active` mode so the arm follows a relaxing finger
back. The real `Law.PULL` now exists (`control.loop.PullPlan`), and this script reads
`step.law` and `step.target` to drive it, which is what makes a device-driven mode run at all.

## The arm doing the moving: --mode passive and --mode cpm

    passive   the arm pulls the finger toward extension; the PATIENT flexes back
    cpm       the arm pulls it out and brings it back; the patient is passive throughout

Both need `--rom-mm`, because under the cap nothing else tells a pulling arm where to stop.
In both, the force cap is a **ceiling** rather than the thing the exercise works against: the
stroke ends at the position target, and the cap only ends it early, when a finger turns out
stiffer than the prescription assumed. That early stop is reported as what it is, a clinical
finding, not a failed repetition.

`cpm` is the mode where nothing in a repetition waits on a person, so every rep is the same
motion and it can run quicker and more smoothly than yielding (`--pull-mm-s`). The cost is
that both halves of the stroke are device-driven, so the deadman governs both: a lost force
reading stops the arm whichever way it was going.

## Afterwards: it returns to home on its own

Releasing moves a short, bounded step first, then, once the tether is slack, the arm walks
back to the saved home pose (`session/home_pose.py`) in small, paced steps, the same pattern
`scripts/go_home.py` uses by hand. That order is deliberate: unloading the tether is urgent,
and a pose change is a second, separate motion that should happen only once the tension is
already gone. This matters because `ticks_per_metre` is pose-dependent, and drifting away
from home across sessions quietly invalidates whichever source (a bench measurement, or
`--from-geometry` re-deriving it from the live pose) the next session uses.

A finger may still be through the hook at this point even though the tether is slack, and
walking back to home drags the attachment point through an arc, so the printed message gives
a moment to remove it and Ctrl-C pauses the return partway (rerun `scripts/go_home.py` to
finish from there). `--no-home-check` skips this entirely and leaves the arm wherever the
session ended, matching the old behaviour exactly.

Start with a low cap and few reps. `--cap-gf 100` is half the clinical 200, and the first
thing worth confirming is not that the controller is good but that yielding actually loosens
the tether, which is `yield_sign` being right in the real pose.

## Reaching the cap before the arm has any speed: --anticipation-gf

`SafetyLimits` only starts yielding once the reading is actually over the cap, and the
admittance law's own velocity is exactly zero right at that point: `yield_velocity`'s
overshoot is the amount *past* the threshold, which is nothing yet. Between a frame being
grabbed and a servo actually moving there is real latency, so a finger that keeps flexing in
that gap pushes the reading well past the cap before the arm has caught up to any speed at
all - on this bench, enough to reach the hard ceiling in a single fast rep even with 10 gf of
room between cap and ceiling.

`--anticipation-gf 10` starts yielding 10 gf below the cap instead of at it, so by the time
the reading reaches the cap the arm already has nonzero velocity. The cost is yielding a
little earlier than the nominal cap, which is why this is a flag rather than a hardcoded
margin: how much headroom is worth trading for how much anticipation is a session-level
choice, not a software default.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.admittance import AdmittanceParams  # noqa: E402
from control.loop import AxisCalibration, ControlLoop, CoordinatedAxis, PullPlan  # noqa: E402
from control.safety import Action, SafetyGate, SafetyLimits  # noqa: E402
from lib.env import env_get  # noqa: E402
from lib.kinematics import ARM_MOTORS  # noqa: E402
from lib.units import grams_force_to_newtons, newtons_to_grams_force  # noqa: E402
from session.calibration import BenchCalibration, NotMeasured  # noqa: E402
from session.modes import MODES, Law, Target  # noqa: E402
from session.protocol import Block, Protocol  # noqa: E402
from session.reps import RepInput, RepLimits, RepMachine  # noqa: E402
from session.tuning import DEFAULT_TUNING_PATH, load_tuning  # noqa: E402

BENCH_CALIB = "config/calibration/bench.json"
GAUGE_CALIB = "config/scale_calib.json"


def build_pipeline(calib_path: str, outlier_persist: int = 3):
    """The same pipeline `python -m gauge_reader run` uses, so readings match what you see
    there, except for `outlier_persist` (see `--outlier-persist`): a run that moves the arm
    on what it reads has more reason to wait out a brief sensor glitch than a read-only
    viewer does, so this is the one knob this script overrides rather than leaving at the
    reader's own default.

    Found on the bench: `lib.filters.ReadingFilter`'s outlier gate holds back a spike for up
    to `outlier_persist` consecutive frames before deciding a *persisting* signal must be a
    real load change and trusting it - by design, so a genuine fast force rise is not smoothed
    away for a second. A sustained misdetection (most likely at a direction reversal, where a
    fast multi-joint turnaround is exactly the kind of moment a frame can blur or the tracker
    can briefly lock onto the wrong markers) can satisfy that same persistence test: on the
    bench, a bogus ~555 gf reading held for exactly 3 frames, the stock default, and was
    trusted, yielding further out for a fraction of a second before a normal reading returned
    and recovery resumed. Raising the default here does not change
    `python -m gauge_reader run`'s own tuning.
    """
    from gauge_reader.cli import _load_calibration, make_pipeline
    from lib.filters import ReadingFilter

    if not Path(calib_path).is_file():
        raise SystemExit(
            f"no gauge calibration at {calib_path}.\n\n"
            "The gauge reader has to be calibrated on this machine first, because it depends on "
            "where this camera sits:\n\n"
            "    .venv/bin/python -m gauge_reader calibrate\n\n"
            "Without it there is no force reading, and this will not run a controller blind."
        )
    return make_pipeline(_load_calibration(calib_path),
                         filt=ReadingFilter(outlier_persist=outlier_persist))


def ticks_per_metre_from_geometry(motor_id: int, pose_ticks: dict[int, float]) -> float:
    """ticks_per_metre for one motor at one pose, computed from NormaCore's URDF.

    A third source, between a bench measurement and a guess. It is geometry, not a fit, so it
    carries none of the tether's hysteresis, but it also assumes the URDF matches this arm and
    that the pose is read correctly. That assumption has been checked: the URDF's joint limits
    agree with the hand-measured firmware ranges within 2 degrees, and motor 7, which the URDF
    says only orients the gripper, measured exactly zero travel per tick on the bench.

    Use it to get the controller moving when a bench measurement is not available. Prefer
    `scripts/measure_ticks_per_metre.py` for the real number: only a measurement catches
    backlash, belt stretch and anything else the model does not know about.
    """
    from lib.kinematics import load_chain

    return 1.0 / load_chain().tip_speed_per_tick(motor_id, pose_ticks)


def resolve_coordinated(args, pose_ticks: dict[int, float],
                        bus_serial: str | None = None) -> tuple[AxisCalibration,
                                                                CoordinatedAxis]:
    """The whole pull, derived from the live pose: joints, direction, distance and sign.

    This is the mode to use when no single joint moves the gripper along the tether, which on
    this arm is every pose tried so far. Measuring `ticks_per_metre` on one motor failed three
    times on the bench for the same reason each time, and the failures looked different enough
    to hide it: the joint travels an arc, so a long enough move crosses the tether's axis
    instead of following it, and the force stops being a monotonic function of the tick delta.
    `lib.ticks_calibration.fit_ticks_per_metre` refuses that data, correctly.

    Nothing here is measured, so none of it knows about backlash. What it does know is the
    geometry, which is the part a single-joint bench measurement was never going to fix.
    """
    import numpy as np

    from lib.kinematics import (
        level_direction,
        load_chain,
        load_tick_ranges,
        tether_tension_direction,
    )

    chain = load_chain()
    tick_ranges = None
    if bus_serial:
        try:
            tick_ranges = load_tick_ranges(bus_serial)
        except FileNotFoundError as exc:
            print(f"!! {exc}\n!! proceeding WITHOUT a joint-range check: a joint running past "
                  "its real travel mid-pull would not be caught before it happens.\n")

    if args.tether_axis:
        direction = np.array([float(v) for v in args.tether_axis.split(",")], dtype=float)
        if np.linalg.norm(direction) < 1e-9:
            raise SystemExit("--tether-axis must not be the zero vector.")
        direction = direction / np.linalg.norm(direction)
        source = "--tether-axis, as given"
    else:
        direction = tether_tension_direction(chain, pose_ticks)
        source = "the gripper's own axis at this pose"

    if args.level:
        # The same levelling `scripts/jog_coordinated.py --level` applies, from the same
        # function, so a pull that tracked straight when jogged tracks the same line here.
        # Without it this follows the gripper's own small tilt out of horizontal, which over
        # a long pull is the arm visibly climbing or dipping away from the tether's line.
        before_z = float(direction[2])
        direction = level_direction(direction)
        source += f", levelled (z was {before_z:+.2f})"

    # Check the stroke this session actually commands, over the distance it can actually
    # travel. Both halves of that matter. The travel limit is what bounds how far the arm
    # gets, so a pose that follows the line for 40 mm but not for 90 is fine for a 40 mm
    # session; and this controller only ever gives way, so the stroke to check is `-direction`,
    # away from tension. The arm is not equally capable in both: at the pose of 2026-11-15 it
    # yields along this line for 90 mm at 54 mm per radian but, travelling the other way, hits
    # a singularity after about 30 mm.
    distance_m = args.max_travel_mm / 1000.0
    try:
        coordinated = CoordinatedAxis.for_pose(pose_ticks, direction, chain,
                                               distance_m=distance_m,
                                               travel_direction=-direction,
                                               tick_ranges=tick_ranges,
                                               speed=args.coordinated_speed,
                                               accel=args.coordinated_accel)
    except ValueError as exc:
        raise SystemExit(
            f"{exc}\n\n"
            "This pose cannot follow that pull. Re-pose the arm (scripts/set_home_pose.py "
            "reports straightness for a pose placed by hand), shorten it with "
            "--max-travel-mm, or give the axis explicitly with --tether-axis."
        ) from None

    tracked = (args.motor if args.motor is not None
               else chain.best_tracked_motor(coordinated.motors, pose_ticks, direction))
    if tracked not in coordinated.motors:
        raise SystemExit(
            f"motor {tracked} is not one of the joints this pull uses ({coordinated.motors}), "
            "so its register would never move and the travel limit would never trip. Drop "
            "--motor and let the pose choose, or re-pose the arm."
        )

    axis = AxisCalibration(
        motor_id=tracked,
        ticks_per_metre=chain.ticks_per_metre_along(tracked, pose_ticks, direction),
        sign=chain.yield_sign(tracked, pose_ticks, direction),
    )
    _, why = chain.can_pull_straight(coordinated.motors, pose_ticks, -direction, distance_m,
                                     tick_ranges=tick_ranges)
    print(f"!! coordinated pull: joints {coordinated.motors}, tracking motor {tracked}.")
    print(f"!! direction x={direction[0]:+.2f} y={direction[1]:+.2f} z={direction[2]:+.2f}, "
          f"from {source}.")
    print(f"!! giving way over {args.max_travel_mm:.0f} mm: {why}")
    print("!! All of this is GEOMETRY, from the URDF and the live pose, not a bench\n"
          "!! measurement: it knows nothing about backlash. Keep the cap low and stop at\n"
          "!! the first thing that looks wrong.\n")
    return axis, coordinated


def resolve_axis(args, pose_ticks: dict[int, float] | None = None) -> AxisCalibration:
    """ticks_per_metre and yield_sign, from the bench calibration or an explicit override."""
    if args.from_geometry:
        if not pose_ticks:
            raise SystemExit("--from-geometry needs the arm's pose, which means Station has to "
                             "be reachable. It could not be read.")
        calib = BenchCalibration.load(BENCH_CALIB)
        sign = args.yield_sign if args.yield_sign is not None else calib.yield_sign
        if sign is None:
            raise SystemExit(
                "--from-geometry gives the distance per tick but not which direction releases "
                "the tether. Measure it, or pass --yield-sign explicitly: getting it backwards "
                "means the arm tightens when it means to release."
            )
        tpm = ticks_per_metre_from_geometry(args.motor, pose_ticks)
        print(f"!! ticks_per_metre = {tpm:.0f} from the URDF, not from a bench measurement.\n"
              "!! It knows the geometry but not the backlash or anything else mechanical.\n"
              "!! Keep the cap low and stop at the first thing that looks wrong.\n")
        return AxisCalibration(motor_id=args.motor, ticks_per_metre=tpm, sign=sign)

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


async def read_pose(client, bus_serial: str, seconds: float,
                    motors: tuple[int, ...] = ARM_MOTORS) -> dict[int, float]:
    """Every arm joint's current position, for the geometry this script derives from the pose.

    Waits for **all** of `motors` by name rather than for a count of readings. Counting is what
    this did first and it was wrong in a way that cannot be seen in the output: eight motors
    stream on this bus, so "seven have arrived" is satisfied by a set that includes the
    gripper's jaws and is missing an arm joint.

    A missing joint is not a small loss of accuracy. `lib.kinematics.Chain` treats an absent
    motor as centred, and the chain is serial, so one wrong angle rotates every link below it.
    Measured on the live arm: dropping motor 2 swings the computed tether direction from
    (-0.19, -0.98, 0.10) to (-0.18, -0.54, 0.82), which is about 60 degrees, and turns a pull
    this pose follows to 0.3 mm into one it cannot follow at all. The caller gets whatever
    arrived and is expected to check it; `missing_joints` is the check.
    """
    from lib.station import follow_motor_state

    found: dict[int, float] = {}

    async def collect():
        async for bus, mid, motor in follow_motor_state(client):
            if bus == bus_serial and motor.position is not None:
                found[mid] = motor.position
                if all(m in found for m in motors):
                    return

    try:
        await asyncio.wait_for(collect(), timeout=seconds)
    except asyncio.TimeoutError:
        pass
    return found


def missing_joints(pose_ticks: dict[int, float] | None) -> list[int]:
    """Arm joints absent from a pose reading. Empty means the geometry can be trusted."""
    return [m for m in ARM_MOTORS if m not in (pose_ticks or {})]


def _grab_and_measure(source, pipeline):
    """One frame and its force reading, as a single unit of blocking work for a thread.

    Both halves belong together off the event loop: the grab blocks on the camera and the
    measurement is heavy CPU on a 1920x1080 frame, and leaving either one inline is enough to
    starve the Station position subscription (see the main loop for what that looked like).
    Returns `(ok, frame, timestamp, result)`, with `result` None when the frame was not
    usable, so the caller's control flow stays exactly as it was when this was inline.
    """
    ok, frame, ts, _ = source.read()
    if not ok:
        return False, None, 0.0, None
    return True, frame, ts, pipeline.process(frame, ts)


async def wait_for_first_position(state: dict, seconds: float) -> bool:
    """Block until the position follower has produced a reading, or `seconds` elapse.

    A session that starts before the first position arrives is a session the controller
    cannot act on: `ControlLoop.step` will not command an arm whose position it does not
    know, so every cycle until that first reading is one the patient spends pulling against
    an arm that silently cannot respond. Waiting here, before the first frame, turns that
    from a silent degradation into either a normal start or a refusal.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if state["ticks"] is not None:
            return True
        await asyncio.sleep(0.02)
    return False


async def read_positions(client, bus_serial: str, motor_id: int, state: dict) -> None:
    """Keep `state['ticks']` and `state['pose']` fresh from the motor state stream.

    The whole pose, not just the tracked joint, because a coordinated pull is solved from
    every joint's current angle: commanding one joint off a stale pose would aim the step
    somewhere the arm is no longer standing.

    `state['pose_at']` records when EACH joint's entry in `state['pose']` was last actually
    updated here, not just the tracked one. Found on the bench: Station can keep delivering
    fresh updates for the tracked motor (so nothing about it looks wrong) while silently going
    quiet on one of the other coordinated joints, and without a per-joint timestamp nothing
    notices - `_command_coordinated` still has a non-empty `pose_ticks` dict to read from, it
    is just wrong for the joint Station stopped reporting, and the geometry mixes a live angle
    for one joint with a stale one for another as if both were current. The caller in `run()`
    is what refuses to use the pose once any coordinated joint's entry is too old; this only
    records the timestamps it needs to do that.
    """
    from lib.station import follow_motor_state

    async for bus, mid, motor in follow_motor_state(client):
        if bus == bus_serial and motor.position is not None:
            state["pose"][mid] = motor.position
            state["pose_at"][mid] = time.monotonic()
            if mid == motor_id:
                state["ticks"] = motor.position
                state["at"] = time.monotonic()


def pull_plan_for(step, machine, args) -> PullPlan | None:
    """The `Law.PULL` motion this cycle asks for, or None to run the yield law instead.

    This is the function that was missing between `session/modes.py` specifying a device-driven
    pull and anything being able to perform one. Before it, this script read neither `step.law`
    nor `step.target` and ran force-gated HOLD/YIELD for every mode, which is why `--mode
    passive` would print its phases while the arm never moved.

    `step.target` is what makes the two directions distinguishable, and it has to be: in `cpm`
    both beats are `Law.PULL` and they aim opposite ways, so a caller holding only the law
    would run the return stroke outward.

    Targets are expressed in `PullPlan`'s signed "how far has this arm given way" quantity,
    measured from the repetition's start, which is the same zero `session.reps.RepMachine`
    uses when it decides a reset has arrived (`RepLimits.home_m`). A stretch is negative
    because stretching a finger toward extension *raises* tension, and raising tension is the
    direction a yielding arm would be moving away from.
    """
    if step.law is not Law.PULL:
        return None
    speed = args.pull_mm_s / 1000.0
    max_step = args.pull_max_step_mm / 1000.0
    if step.target is Target.ROM:
        target = machine.block.rom_target_m
        if target is None:          # refused by Block for a pulling mode, so this is a bug
            return None
        return PullPlan(target_yielded_m=-target, speed_m_s=speed, max_step_m=max_step)
    if step.target is Target.REP_START:
        return PullPlan(target_yielded_m=0.0, speed_m_s=speed, max_step_m=max_step)
    # Target.NONE under a pull is ARMED: bounded by a force, not a position. It stops at the
    # tether's resting tension rather than at the cap, so the finger is pre-tensioned and not
    # already working while the patient is still being cued.
    return PullPlan(
        target_yielded_m=-args.max_travel_mm / 1000.0,
        speed_m_s=min(speed, args.arm_mm_s / 1000.0),
        max_step_m=max_step,
        stop_above_n=machine.pretension_n,
    )


def block_for(args) -> Block:
    """The prescription this run exercises, from a protocol file or from the flags.

    Two sources, and the file wins when it is given, because a protocol is what a clinician
    authored and a flag is what someone typed at a prompt. `--protocol` names a file in
    `config/protocols/`; `--protocol-block` picks which of its blocks to run, since this
    script runs one block and a protocol holds several (sequencing them across fingers is
    `session.orchestrator`'s job, and it needs arm 2 to move the hook between them).

    Individual flags still override the file, so a clinician's protocol can be loaded and one
    number changed for one run without editing the prescription. The override is reported, not
    silent: a session that quietly ran something other than what the file says is the thing
    worth preventing here. `--cap-gf` is deliberately *not* overridable this way, because a
    cap is the number the whole prescription is built around and changing it by flag while
    claiming to run a named protocol is how the log stops meaning anything.
    """
    protocol_path = getattr(args, "protocol", None)
    if protocol_path is None:
        return Block(
            finger=args.finger,
            mode=args.mode,
            reps=args.reps,
            cap_n=grams_force_to_newtons(args.cap_gf),
            hold_s=args.hold,
            rom_target_m=(args.rom_mm / 1000.0) if MODES[args.mode].needs_rom_target else None,
        )

    protocol = Protocol.load(protocol_path)
    index = getattr(args, "protocol_block", 0)
    if not 0 <= index < len(protocol):
        raise SystemExit(
            f"{protocol_path} has {len(protocol)} block(s), so --protocol-block {index} does "
            f"not exist. The blocks are:\n" + "\n".join(
                f"  {i}: {b.reps} x {b.mode} on the {b.finger} at {b.cap_gf:.0f} gf"
                for i, b in enumerate(protocol)
            )
        )
    block = protocol.blocks[index]
    print(f"\n!! protocol {protocol.name or protocol_path}")
    print(f"!! block {index} of {len(protocol)}: {block.reps} x {block.mode} on the "
          f"{block.finger} at {block.cap_gf:.0f} gf")

    # Only the fields a person would reasonably want to shorten for one run. Each is reported
    # so the printed banner and the session log disagree with the file out loud, never quietly.
    overrides: dict[str, object] = {}
    if args.reps != _default_of(args, "reps"):
        overrides["reps"] = args.reps
    if args.hold != _default_of(args, "hold"):
        overrides["hold_s"] = args.hold
    if overrides:
        for name, value in overrides.items():
            print(f"!! overriding {name}: {getattr(block, name)} -> {value} (from the flags)")
        block = replace(block, **overrides)
    print()
    return block


def _default_of(args, name: str):
    """What the parser would have produced for `name` had nobody typed it.

    Needed to tell "the user asked for 5 reps" from "5 is simply the default", which argparse
    does not record. Comparing against the default is the cheapest honest way to do it without
    a second parse or a sentinel on every flag.
    """
    return getattr(args, "_defaults", {}).get(name)


async def run(args) -> int:
    from lib.station import connect

    # Re-read rather than threaded down from main(): every number from it has already become
    # an `args` field by now, so this is only for the profile's own name in the banner, and
    # reading it here keeps run() callable from a test with a plain namespace.
    tuning = load_tuning(getattr(args, "tuning", None))
    block = block_for(args)
    bench = BenchCalibration.load(BENCH_CALIB)
    for problem in block.check_against(bench):
        print(f"warning: {problem}\n")

    host = args.station_host or env_get("STATION_HOST", "localhost")
    bus_serial = args.bus_serial or env_get("FOLLOWER_BUS_SERIAL")
    if not bus_serial:
        raise SystemExit("no follower bus serial: set FOLLOWER_BUS_SERIAL in .env or pass "
                         "--bus-serial.")

    # Connect before resolving the axis: --from-geometry needs the arm's actual pose, because
    # ticks_per_metre depends on it. Connecting early costs nothing on the other paths.
    client = await connect(host=host)
    needs_pose = args.from_geometry or args.coordinated
    pose_ticks = await read_pose(client, bus_serial, 5.0) if needs_pose else None

    if needs_pose and (gaps := missing_joints(pose_ticks)):
        raise SystemExit(
            f"no position arrived for motor(s) {gaps} within 5 s, and the pose is what every "
            "number here is derived from. One missing joint is not a small error: the chain is "
            "serial, so it rotates the computed tether direction by tens of degrees and the "
            "arm would then pull along a line that is not the tether's.\n\n"
            "Check the bus is connected and Station is streaming, then try again."
        )

    coordinated = None
    if args.coordinated:
        axis, coordinated = resolve_coordinated(args, pose_ticks, bus_serial)
    else:
        if args.motor is None:
            raise SystemExit("--motor is required unless --coordinated, which picks the joints "
                             "from the pose itself.")
        axis = resolve_axis(args, pose_ticks)
    machine = RepMachine(block, RepLimits(
        blind_grace_s=args.blind_grace_s,
        slack_blind_grace_s=args.slack_blind_grace_s,
    ))
    limits = SafetyLimits(
        f_threshold_n=block.cap_n,
        f_max_n=grams_force_to_newtons(args.ceiling_gf),
        max_travel_m=args.max_travel_mm / 1000.0,
        max_age_s=args.max_age_s,
        max_dropouts=args.max_dropouts,
        anticipation_n=grams_force_to_newtons(args.anticipation_gf),
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
        coordinated=coordinated,
        yield_check_window_s=args.yield_check_window_s if args.yield_check else None,
        recover_gain=args.recover if args.recover > 0 else None,
        recover_floor_n=(grams_force_to_newtons(args.recover_floor_gf)
                         if args.recover_floor_gf is not None else machine.pretension_n),
        recover_retrace_m_s=args.recover_retrace_mm_s / 1000.0,
    )
    if args.recover > 0:
        floor_gf = (args.recover_floor_gf if args.recover_floor_gf is not None
                   else newtons_to_grams_force(machine.pretension_n))
        retrace_note = (f", finishing the walk home at {args.recover_retrace_mm_s:g} mm/s "
                        "once the force reaches that floor" if args.recover_retrace_mm_s > 0
                        else " (and stopping there - pass --recover-retrace-mm-s to keep "
                             "walking home once the force reaches it)")
        print(f"!! --recover {args.recover:g}: when your finger eases off, the arm follows it\n"
              "!! back toward where it started (never further back than that), stopping at "
              f"{floor_gf:.0f} gf rather than chasing the force to true slack{retrace_note}.\n")
    if not args.yield_check:
        print("!! --no-yield-check: the arm will NOT release on its own if it moves in a\n"
              "!! direction that keeps raising the force instead of lowering it. The hard\n"
              "!! ceiling and the camera-dropout check still apply.\n")
    pipeline = build_pipeline(args.calib, outlier_persist=args.outlier_persist)

    print(f"\n{block.reps} x {block.mode} on the {block.finger}")
    anticipation_note = (f", yields early from {args.cap_gf - args.anticipation_gf:.0f} gf"
                        if args.anticipation_gf > 0 else "")
    print(f"  cap      {block.cap_gf:.0f} gf ({block.cap_n:.2f} N), ceiling {args.ceiling_gf:.0f} gf"
          f"{anticipation_note}")
    print(f"  resting  {newtons_to_grams_force(machine.pretension_n):.0f} gf")
    moving = (f"joints {coordinated.motors}, tracking motor {axis.motor_id}"
              if coordinated else f"motor {axis.motor_id}")
    print(f"  arm      {moving} on {bus_serial}, "
          f"{axis.ticks_per_metre:.0f} ticks/m, yield sign {axis.sign:+d}")
    print(f"  travel   {args.max_travel_mm:.0f} mm limit, {args.max_step_mm:.1f} mm max step")
    print(f"  tuning   {tuning.profile}")
    # The gate's own give-up thresholds, printed every run rather than left in a file. Three
    # of them are looser than what the code shipped with, each for a reason written down in
    # session/tuning.py, and a loosened safety limit that nobody is shown on the way past is
    # the kind that stops being a decision and becomes an accident.
    print(f"  gate     releases over {args.ceiling_gf:.0f} gf, after "
          f"{args.max_dropouts} blind frames, readings stale past {args.max_age_s:g} s"
          f"{'' if args.yield_check else ', yield-direction check OFF'}\n")
    print("The arm will MOVE, in the direction that releases tension. Keep a hand on the e-stop.")
    if input("Type 'yes' to run: ").strip().lower() != "yes":
        print("aborted; nothing moved.")
        return 1

    from lib.camera import open_source

    source = open_source(camera=args.camera)
    # `pose_at` starts empty rather than seeded with `now`: the initial `pose_ticks` came from
    # the startup read, which could have been many seconds ago by the time the operator typed
    # 'yes', and treating that as fresh would mask exactly the staleness this tracks.
    state: dict = {"ticks": None, "at": 0.0, "pose": dict(pose_ticks or {}), "pose_at": {}}
    follower = asyncio.create_task(read_positions(client, bus_serial, axis.motor_id, state))

    def _report_follower_death(task: asyncio.Task) -> None:
        """Surface an exception that killed the position reader.

        `asyncio.create_task` swallows exceptions until the task is awaited or garbage
        collected, and this one is never awaited during the session. A follower that dies on
        its first iteration therefore looks identical to one that is merely slow: the session
        runs to completion, the force and step columns keep updating from the camera, and the
        only trace is that the arm never moved. Anything that stops the position stream has
        to be loud, because every other symptom of it is silence.
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            print(f"\n!! the arm position reader DIED: {type(exc).__name__}: {exc}\n"
                  "!! No position means the controller cannot command the arm at all, so the "
                  "session above moved nothing.\n")

    follower.add_done_callback(_report_follower_death)

    if not await wait_for_first_position(state, args.position_wait_s):
        source.release()
        follower.cancel()
        raise SystemExit(
            f"no arm position arrived from Station within {args.position_wait_s:g} s, so the "
            "controller would have been unable to command the arm for the whole session.\n\n"
            "Nothing has moved and the camera is closed. Check Station is running and the "
            "bus is reachable:\n\n"
            "    .venv/bin/python scripts/go_home.py --check\n"
        )
    started = time.monotonic()
    last_line = ""
    last_stale_warning_at = 0.0
    position_ever_seen = False

    print(f"\n{'force':>9s}  {'phase':9s}  {'gate':8s}  {'step':>7s}  {'travel':>8s}  "
          f"reps   instruction")
    print("-" * 95)
    try:
        while True:
            now = time.monotonic()
            if now - started > args.max_seconds:
                print("\nout of time")
                break
            # Grabbing and processing a frame happens in a worker thread, not inline, so the
            # event loop keeps running for the ~100 ms each takes. Inline, with only an
            # `await asyncio.sleep(0)` between iterations, the loop gets exactly one turn per
            # frame, and one turn per frame is not enough for the Station subscription to even
            # finish starting: measured on this bench, the position stream's first reading
            # arrived 3.4 s into the session with a trivial pipeline, and never at all within
            # 22 s with the real one. `ControlLoop.step` refuses to command the arm without a
            # position, correctly, so the whole session ran with the force and step columns
            # updating from the camera while the arm received nothing and finished with
            # "arm travelled 0.0 mm". Handing the blocking work to a thread is what makes the
            # follower an actually-concurrent task rather than one starved behind the camera.
            ok, frame, ts, result = await asyncio.get_running_loop().run_in_executor(
                None, _grab_and_measure, source, pipeline
            )
            if not ok:
                continue
            force_n = result.newtons if result.found else None

            step = machine.step(RepInput(
                force_n=force_n,
                reading_age_s=max(0.0, time.monotonic() - ts),
                travel_m=loop.state.travel_m,
                now=now,
                confident=bool(result.found),
            ))
            decision = await loop.step(
                client, bus_serial, force_n, ts, state["ticks"], now=now,
                pose_ticks=dict(state["pose"]) if coordinated else None,
                pull=pull_plan_for(step, machine, args),
            )
            if state["ticks"] is not None:
                position_ever_seen = True

            # Visibility for a failure that is otherwise silent: without a real position the
            # force and the printed "step" column above are still computed from the camera
            # alone and keep changing normally, so a dead Station position stream looks
            # exactly like a working session right up until "arm travelled 0.0 mm" at the
            # end. `loop.step()` already refuses to command anything without a position
            # (the correct, safe behaviour), so this adds the warning that behaviour was
            # missing, not a second safety mechanism.
            position_age = now - state["at"] if state["ticks"] is not None else now - started
            stale_after = args.position_stale_warning_s
            if position_age > stale_after and now - last_stale_warning_at > stale_after:
                last_stale_warning_at = now
                print(
                    f"\n!! no real arm position for {position_age:.1f} s"
                    f"{'' if position_ever_seen else ' (none received yet this session)'}: "
                    "the force and step above are real, but the arm is NOT being commanded. "
                    "Check Station and the bus; this will keep reading as a normal session "
                    "otherwise.\n"
                )

            gf = "   --   " if force_n is None else f"{newtons_to_grams_force(force_n):6.0f} gf"
            moved = loop.state.records[-1].step_m * 1000 if loop.state.records else 0.0
            # The real, cumulative distance this *cycle's* sole input to RepMachine's own
            # "has the arm actually come back" check (_reset in session/reps.py) relies on -
            # computed only from the tracked motor's real tick readback, never from the sum
            # of commanded steps above. Printed explicitly rather than left to be inferred
            # from summing "step" by eye, because the two can disagree: on the bench, a rep
            # was counted done by the home_m check within one reset cycle, right after the
            # step column had been printing a sustained 2.00 mm for nearly twenty lines in a
            # row - something between the requested motion and the real tick readback did
            # not add up, and this is what would have shown it directly instead of requiring
            # that arithmetic after the fact.
            line = (f"{gf:>9s}  {step.phase.value:9s}  {decision.action.value:8s}  "
                    f"{moved:6.2f}mm  {loop.state.travel_m*1000:7.1f}mm  "
                    f"{step.reps_completed}/{block.reps}    {step.instruction}")
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

    # Release puts the arm wherever it happened to end, not at home on its own. ticks_per_metre
    # is pose-dependent (--from-geometry recomputes it from the live pose each run, but a bench
    # measurement does not), so drifting away from home across sessions quietly invalidates
    # whichever one is in use - this is what walks it back, every time, not just when a person
    # remembers to run scripts/go_home.py.
    if not args.no_home_check:
        try:
            from lib.station import set_position
            from session.home_pose import (
                distance_from_home,
                home_waypoints,
                is_home,
                load_home_pose,
            )

            home = load_home_pose()
            final = await read_pose(client, bus_serial, 3.0)
            missing = [m for m in home if m not in final]
            if missing:
                print(f"\ncould not read motor(s) {missing}; not attempting to return home. "
                      "Run scripts/go_home.py by hand once the bus is reachable.")
            elif not is_home(final, home=home):
                print(f"\nreturning home ({distance_from_home(final, home)} ticks away) ...")
                print("If a finger is still through the hook, remove it now - Ctrl-C pauses "
                      "this partway, same as scripts/go_home.py.")
                waypoints = home_waypoints(final, home, args.go_home_max_step_ticks)
                try:
                    for way in waypoints:
                        for m, target in way.items():
                            await set_position(client, bus_serial, m, target)
                        await asyncio.sleep(args.go_home_settle)
                    print("at home.")
                except KeyboardInterrupt:
                    print("\nstopped partway home. Rerun scripts/go_home.py to finish.")
        except SystemExit:
            pass   # no home pose saved yet; nothing to return to
    return 0


def main(argv=None) -> int:
    # Two-phase parse so --tuning can choose where every other default comes from. The
    # alternative, reading the profile after parsing, would mean the defaults printed by
    # --help and the defaults actually used could differ, which is the one thing a file of
    # safety-relevant numbers must not do.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--tuning", default=str(DEFAULT_TUNING_PATH))
    tuning = load_tuning(pre.parse_known_args(argv)[0].tuning)
    safety = tuning.safety

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tuning", default=str(DEFAULT_TUNING_PATH),
                   help="the machine tuning profile every default below comes from. The "
                        "committed profile holds what the bench proved, so a normal session "
                        "needs no tuning flags at all; see session/tuning.py for what each "
                        "number is paying for")
    p.add_argument("--motor", type=int, default=None,
                   help="the joint that moves along the tether axis. Required unless "
                        "--coordinated, which picks the joints from the pose instead. Rig "
                        "knowledge; see docs/bench-measurements.md for how to identify it safely")
    p.add_argument("--protocol", default=None,
                   help="run a block from a clinician's protocol file (config/protocols/) "
                        "instead of from --finger/--mode/--cap-gf/--reps. The file is what "
                        "was prescribed, so it wins over those; --reps and --hold still "
                        "override it for one run, and say so out loud when they do")
    p.add_argument("--protocol-block", type=int, default=0, metavar="N",
                   help="which block of --protocol to run, 0-based. This script runs one "
                        "block; sequencing a whole protocol across fingers needs arm 2 to "
                        "move the hook between them, which is session.orchestrator's job")
    p.add_argument("--finger", default="index")
    p.add_argument("--mode", default="active", choices=sorted(MODES))
    p.add_argument("--cap-gf", type=float, default=100.0,
                   help="force cap in grams-force. Start at 100, not the clinical 200")
    p.add_argument("--ceiling-gf", type=float, default=safety.ceiling_gf,
                   help="hard ceiling: the gate latches RELEASE above this, unconditionally. "
                        "200 is the clinical cap; the PESOLA's own full scale, 600, is the "
                        "highest this can usefully be")
    p.add_argument("--anticipation-gf", type=float, default=safety.anticipation_gf,
                   help="start yielding this many grams-force BELOW --cap-gf, so the arm "
                        "already has speed by the time the reading reaches the cap instead "
                        "of starting from a standstill there. 0 (default) is the literal cap. "
                        "Try 10-15 if a fast rep overshoots the cap before the arm reacts")
    p.add_argument("--max-age-s", type=float, default=safety.max_age_s,
                   help="a force reading older than this is treated as stale, not a reading")
    p.add_argument("--max-dropouts", type=int, default=safety.max_dropouts,
                   help="consecutive frames with no reading (stale or missing) before the "
                        "gate releases. Does NOT cover --recover or a device-driven reset; "
                        "see --blind-grace-s for that")
    p.add_argument("--outlier-persist", type=int, default=tuning.outlier_persist,
                   help="how many consecutive held-back-looking readings the gauge reader's "
                        "own outlier gate needs before trusting a sudden jump as real, "
                        "rather than noise. Higher survives a longer sensor glitch before "
                        "acting on it, at the cost of reacting slower to a genuine fast "
                        "force spike. 3 is the gauge reader's own default (python -m "
                        "gauge_reader run); on the bench a bogus ~555 gf reading persisted "
                        "for exactly 3 frames and was trusted, hence the higher default here")
    p.add_argument("--blind-grace-s", type=float, default=safety.blind_grace_s,
                   help="a SEPARATE, stricter tolerance: how long a device-driven phase "
                        "(the --recover walk back, or a passive-mode pull) may go without a "
                        "usable reading before the repetition is abandoned. This is "
                        "session.reps.RepMachine's own deadman, not the gate's; raising "
                        "--max-dropouts alone does not touch it. Default 0.3s is "
                        "conservative because a lost reading here leaves the arm moving the "
                        "finger with no idea how hard")
    p.add_argument("--slack-blind-grace-s", type=float, default=safety.slack_blind_grace_s,
                   help="the same tolerance as --blind-grace-s, but for a device-driven phase "
                        "that can only ever REDUCE tension, which a reset is. Longer on "
                        "purpose: such a phase succeeds by making the tether slack, and a "
                        "slack tether is exactly when the camera loses the PESOLA's markers, "
                        "so the strict grace kept abandoning resets for working")
    p.add_argument("--position-wait-s", type=float, default=5.0,
                   help="how long to wait for Station's first arm position before refusing "
                        "to start. The controller cannot command an arm whose position it "
                        "does not know, so starting without one means a session that reads "
                        "normally while the arm never moves")
    p.add_argument("--position-stale-warning-s", type=float, default=1.0,
                   help="print a loud warning if no real arm position has arrived from "
                        "Station in this many seconds. The force/step columns are driven by "
                        "the camera alone and keep looking normal even when the Station "
                        "position stream is dead and the arm is not moving at all")
    p.add_argument("--yield-check", action=argparse.BooleanOptionalAction,
                   default=safety.yield_check,
                   help="release if yielding does not lower the force within "
                        "--yield-check-window-s. OFF in the committed profile, so "
                        "--yield-check turns it back on: it also fires when a person flexes "
                        "faster than --v-max-mm-s, which is a real but harmless case on "
                        "hardware with no force capable of injury, and it fired on every "
                        "session on this bench. The hard ceiling and the dropout check are "
                        "unaffected either way")
    p.add_argument("--yield-check-window-s", type=float, default=safety.yield_check_window_s,
                   help="with the check enabled: how long yielding may run without the force "
                        "coming down before concluding the direction is wrong")
    p.add_argument("--recover", type=float, default=tuning.recover_gain, metavar="GAIN",
                   help="follow the finger BACK when it eases off: take slack up toward the "
                        "session's starting position (never past it) at this gain relative "
                        "to yielding. 0 (default) keeps the original yield-only behaviour; "
                        "1 chases slack as eagerly as it gives way")
    p.add_argument("--recover-floor-gf", type=float, default=tuning.recover_floor_gf,
                   help="with --recover: stop chasing the force down once it reaches this "
                        "many grams-force, rather than chasing it to true slack. Defaults to "
                        "the session's own resting tension (printed as 'resting'); a tether "
                        "recovered past that point can go slack enough to lose the camera's "
                        "PESOLA markers")
    p.add_argument("--recover-retrace-mm-s", type=float, default=tuning.recover_retrace_mm_s,
                   help="with --recover: once the force reaches --recover-floor-gf and has "
                        "nothing further to say, keep retracing the remaining distance home "
                        "at this fixed, slow speed instead of stopping there. Needed because "
                        "a finger that relaxes faster than --v-max-mm-s can follow slackens "
                        "the tether before the arm has mechanically retraced very far, "
                        "leaving nothing but this to carry it the rest of the way; 0 "
                        "disables it and the arm stops at the floor, which on the bench "
                        "meant it never reliably reached home. Deliberately much slower than "
                        "--v-max-mm-s: there is no force reading left to validate urgency "
                        "once this is the only thing moving the arm")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--hold", type=float, default=2.0)
    p.add_argument("--rom-mm", type=float, default=15.0,
                   help="how far a device-driven pull may stretch the finger, measured from "
                        "where the repetition started. Required by any mode that pulls "
                        "(passive, cpm) and meaningless for active: under the cap nothing "
                        "else tells a pulling arm to stop. Bounded by the finger's measured "
                        "excursion in config/calibration/bench.json")
    p.add_argument("--pull-mm-s", type=float, default=tuning.pull_mm_s,
                   help="how fast a device-driven pull travels. A setting, not a computed "
                        "response: under the cap the force says nothing about how fast to "
                        "move a relaxed finger, so unlike yielding there is no law to derive "
                        "this from. Can be quicker than the yield speed because the path is "
                        "decided before the stroke begins rather than chased frame by frame")
    p.add_argument("--pull-max-step-mm", type=float, default=tuning.pull_max_step_mm,
                   help="largest displacement a pull may take in one cycle. Separate from "
                        "--max-step-mm for the same reason the recovery step is: that number "
                        "bounds the yield step's smoothness against a flexing finger, and a "
                        "pull is not that motion")
    p.add_argument("--arm-mm-s", type=float, default=tuning.arm_mm_s,
                   help="how fast the ARMED phase takes up slack. Slower than --pull-mm-s on "
                        "purpose: this is the one pull that starts with the tether slack and "
                        "the finger already through the hook, so it is the stroke with the "
                        "least information about what it is about to touch")
    p.add_argument("--damping", type=float, default=tuning.damping_n_s_per_m,
                   help="B in N.s/m; the simulation says 10-20 for this tether")
    p.add_argument("--v-max-mm-s", type=float, default=tuning.v_max_mm_s)
    p.add_argument("--deadband-n", type=float, default=tuning.deadband_n)
    p.add_argument("--max-step-mm", type=float, default=tuning.max_step_mm,
                   help="largest move in one cycle. Keep small for a first run")
    p.add_argument("--max-travel-mm", type=float, default=tuning.max_travel_mm,
                   help="total travel before the gate releases")
    p.add_argument("--coordinated", action=argparse.BooleanOptionalAction,
                   default=tuning.coordinated,
                   help="pull along the tether with three joints at once, choosing them, the "
                        "direction, the ticks_per_metre and the yield sign from the URDF and "
                        "the live pose. ON in the committed profile, because no single joint "
                        "on this arm follows the tether; --no-coordinated falls back to one "
                        "motor and then --motor is required")
    p.add_argument("--tether-axis", default=None,
                   help="with --coordinated: the tether's direction as x,y,z in the base "
                        "frame, pointing the way that INCREASES tension. Defaults to the "
                        "gripper's own axis at the live pose, which is usually right")
    p.add_argument("--level", action=argparse.BooleanOptionalAction, default=tuning.level,
                   help="with --coordinated: drop the z component of the tether direction and "
                        "renormalise, so the pull stays horizontal instead of following the "
                        "gripper's own tilt. ON in the committed profile. The same flag "
                        "scripts/jog_coordinated.py has; keep the two in agreement")
    p.add_argument("--coordinated-speed", type=int, default=tuning.coordinated_speed,
                   help="with --coordinated: raw ST3215 goal-speed units, scaled per joint "
                        "each cycle so all three arrive together. 700 tracked a straight "
                        "line with no visible wobble on this bench (scripts/jog_coordinated.py "
                        "is where to re-tune it if the arm's load changes)")
    p.add_argument("--coordinated-accel", type=int, default=tuning.coordinated_accel,
                   help="with --coordinated: raw ST3215 acceleration units, written once")
    p.add_argument("--from-geometry", action="store_true",
                   help="compute ticks_per_metre from the URDF and the live pose, instead of "
                        "a bench measurement. Needs yield_sign measured or passed explicitly")
    p.add_argument("--ticks-per-metre", type=float, default=None,
                   help="override the measured value for a first rough trial")
    p.add_argument("--yield-sign", type=int, default=None, choices=[1, -1],
                   help="with --ticks-per-metre: which tick direction lowers the force")
    p.add_argument("--camera", default=None)
    p.add_argument("--calib", default=GAUGE_CALIB)
    p.add_argument("--station-host", default=None)
    p.add_argument("--bus-serial", default=None)
    p.add_argument("--max-seconds", type=float, default=180.0)
    p.add_argument("--no-home-check", action="store_true",
                   help="do not return the arm to the saved home pose after the session")
    p.add_argument("--go-home-max-step-ticks", type=int, default=80,
                   help="largest per-joint move, in ticks, on the way back to home")
    p.add_argument("--go-home-settle", type=float, default=0.35,
                   help="seconds between steps on the way back to home")
    # What the parser would have produced had nobody typed anything. argparse does not record
    # it, and `block_for` needs it to tell "the user asked for 2 reps" from "2 is simply the
    # default". Registered as a default rather than attached to `args` afterwards, so it is
    # present for every caller of this parser and not only for `main`: a test or the dashboard
    # building args through `parse_args` directly would otherwise see no defaults at all and
    # read every flag as an override.
    p.set_defaults(_defaults={a.dest: a.default for a in p._actions})
    args = p.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
