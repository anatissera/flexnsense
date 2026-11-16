"""Continuous passive motion: walk the finger out along a line and back, over and over.

    .venv/bin/python scripts/run_cpm.py --no-camera          # nothing attached, check the line
    .venv/bin/python scripts/run_cpm.py --reps 5             # with the finger hooked

**This is a different kind of thing from `scripts/run_session.py`, on purpose.**

`run_session.py` is an admittance controller: the patient moves, the camera measures the
force, and the force decides every step the arm takes. That is the right design when the
exercise is driven by the patient's own effort.

CPM is not that exercise. The arm does all the moving, along a path that is the same every
repetition, and the patient contributes nothing. There is no signal to chase, so there is
nothing for a per-step force reading to decide. Driving it off the camera anyway was tried
and it does not work, for a reason worth writing down rather than rediscovering:

- the repetition is supposed to be identical every time, and a force-gated one is not: it is
  whatever the camera happened to see, which on this rig includes bogus readings of 390 and
  502 gf with nothing happening;
- the gate's job is to **give way** when the force is too high, so a reading over the cap made
  the arm yield, which moved it *away* from the path instead of along it;
- and the return stroke raises tension, so an over-cap reading blocked it outright. The arm
  reached 32 mm out with the return permanently refused. It could not come back.

So the motion here comes from `lib.kinematics.Chain.line_trajectory`, computed **once**, before
the first command, exactly as `scripts/jog_coordinated.py` does. The camera is a supervisor
with one job: stop everything if the force passes a ceiling. It never decides a step.

That is also why this can run at a sensible speed. There is no reaction latency to pay for,
because nothing is being reacted to.

## Where it starts, and why that is a second pose

`config/calibration/home-pose.json` is where the arm is **parked**: tether slack, nothing
hooked, safe to leave it there. CPM cannot start from there, because a CPM stroke needs
somewhere to travel *to* and the pose has to be reachable in both directions with the finger
attached and the tether already taking up.

So CPM has its own start pose, `config/calibration/cpm-start-pose.json`, captured the same
way the home pose was:

    .venv/bin/python scripts/lock_arm.py --release
    # ... place the arm by hand, with the finger hooked, where the stroke should begin ...
    .venv/bin/python scripts/set_home_pose.py --save --path config/calibration/cpm-start-pose.json

This script walks the arm to that pose before it starts, in paced steps, and returns it there
at the end, which is also where it starts from, so running it again right after needs no walk
at all. It is a pose that holds a finger under tension, though, so it is not somewhere to
leave the arm between sessions: that choice is on the person at the keyboard, not automatic,
because going from here to the real parking pose with a finger still hooked would drag it
through the arc between the two.

## Direction

By default the first half of the stroke travels in the **yield** direction, away from tension,
which is the only direction that cannot tighten anything on a finger whatever else is wrong.
The return comes back along the same waypoints to exactly where it began. `--toward-tension`
reverses that, so the stroke stretches first and releases second. Which one is the exercise
depends on how the rig is set up, and the printed banner always says which one is about to
happen, in words, before anything moves.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.env import env_get  # noqa: E402
from lib.kinematics import ARM_MOTORS, level_direction  # noqa: E402
from lib.units import grams_force_to_newtons, newtons_to_grams_force  # noqa: E402
from session.tuning import DEFAULT_TUNING_PATH, load_tuning  # noqa: E402

START_POSE_PATH = "config/calibration/cpm-start-pose.json"
GAUGE_CALIB = "config/scale_calib.json"

# Bench-proven for exactly this kind of motion: an open-loop walk through a pre-planned list
# of waypoints, each held for a settle period. These are scripts/jog_coordinated.py's own
# defaults, and its CoordinatedAxis docstring records why: "700/50 tracked a straight line
# with no visible wobble on this bench at a pace that kept up with a hand-paced pull."
#
# `config/session-tuning.json`'s coordinated_speed/coordinated_accel (900/30) are NOT this
# number tuned differently - they are tuned for a different motion entirely:
# `control.loop.ControlLoop`'s admittance cycle, which rewrites the goal tens of times a
# second in small (<=4 mm) force-driven corrections. Defaulting this script to that profile
# was the first version's mistake: with accel=30 and a short settle, the servo never finished
# ramping up before the next 1 mm waypoint overwrote its goal, so every step looked like a
# jerky restart rather than part of one continuous motion. A bigger pre-planned walk needs the
# jog profile's longer settle and firmer accel instead, not the admittance loop's.
JOG_PROVEN_SPEED = 700
JOG_PROVEN_ACCEL = 50
JOG_PROVEN_SETTLE = 0.3


def build_pipeline(calib_path: str, outlier_persist: int = 5):
    """The same pipeline `python -m gauge_reader run` uses, with a longer outlier window.

    Duplicated from `scripts/run_session.py` rather than imported, because `scripts/` is not a
    package and importing one script from another by path is worse than two lines. If a third
    caller appears, this belongs in `gauge_reader/` instead.
    """
    from gauge_reader.cli import _load_calibration, make_pipeline
    from lib.filters import ReadingFilter

    if not Path(calib_path).is_file():
        raise SystemExit(
            f"no gauge calibration at {calib_path}.\n\n"
            "The gauge reader has to be calibrated on this machine first, because it depends "
            "on where this camera sits:\n\n"
            "    .venv/bin/python -m gauge_reader calibrate\n\n"
            "Without it there is no force reading. Pass --no-camera to run the path with "
            "nothing attached instead."
        )
    return make_pipeline(_load_calibration(calib_path),
                         filt=ReadingFilter(outlier_persist=outlier_persist))


async def read_pose(client, bus_serial: str, seconds: float) -> dict[int, float]:
    """Every joint's current tick reading, or whatever arrived before `seconds` ran out.

    The same function `scripts/jog_coordinated.py` and `scripts/go_home.py` use.
    `follow_motor_state` yields one tuple per motor across every bus, so filtering to this
    bus is the caller's job.
    """
    from lib.station import follow_motor_state

    found: dict[int, float] = {}

    async def collect():
        async for bus, mid, motor in follow_motor_state(client):
            if bus == bus_serial and motor.position is not None:
                found[mid] = motor.position
                if all(m in found for m in ARM_MOTORS):
                    return

    try:
        await asyncio.wait_for(collect(), timeout=seconds)
    except asyncio.TimeoutError:
        pass
    return found


class ForceSupervisor:
    """The camera, as a ceiling and nothing else.

    Runs in its own thread and publishes the latest reading. The motion never waits on it and
    never asks it for permission to take a step; it is read between steps to answer one
    question, "has the force passed the ceiling", and one more, "has the camera been blind so
    long that the answer is unknown".

    Both of those stop the stroke, and the second is not a formality. The project's rule is
    that a missing force reading is never evidence of a safe force, so a supervisor that has
    gone blind while an arm moves a finger has to be treated as a supervisor reporting a
    problem. The tolerance is generous compared with `control.loop.ControlLoop`'s, because
    this motion does not depend on the reading cycle by cycle, but it is not infinite.
    """

    def __init__(self, source, pipeline, ceiling_n: float, blind_tolerance_s: float):
        self.source = source
        self.pipeline = pipeline
        self.ceiling_n = ceiling_n
        self.blind_tolerance_s = blind_tolerance_s
        self.latest_n: float | None = None
        self.peak_n = 0.0
        self.last_seen = time.monotonic()
        self.stop_reason: str | None = None
        self._running = True

    def poll_once(self) -> None:
        """Grab and measure one frame. Blocking, so it is called in an executor.

        Both halves belong off the event loop together: the grab blocks on the camera and the
        measurement is heavy CPU on a 1920x1080 frame. Leaving either inline starves the
        Station traffic that is actually moving the arm, which `scripts/run_session.py`
        documents having measured.
        """
        ok, frame, ts, _ = self.source.read()
        if not ok or frame is None:
            return
        result = self.pipeline.process(frame, ts)
        now = time.monotonic()
        if result.found and result.newtons is not None:
            self.latest_n = result.newtons
            self.peak_n = max(self.peak_n, result.newtons)
            self.last_seen = now
            if result.newtons > self.ceiling_n:
                self.stop_reason = (
                    f"force {newtons_to_grams_force(result.newtons):.0f} gf passed the "
                    f"{newtons_to_grams_force(self.ceiling_n):.0f} gf ceiling"
                )
        elif now - self.last_seen > self.blind_tolerance_s:
            self.stop_reason = (
                f"no force reading for {now - self.last_seen:.1f} s while the arm was moving "
                "the finger. A missing reading is not evidence of a safe force"
            )

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while self._running:
            await loop.run_in_executor(None, self.poll_once)
            await asyncio.sleep(0)

    def cancel(self) -> None:
        self._running = False

    @property
    def reading(self) -> str:
        if self.latest_n is None:
            return "  --  "
        return f"{newtons_to_grams_force(self.latest_n):4.0f} gf"


async def run(args) -> int:
    import numpy as np

    from lib.kinematics import load_chain, load_tick_ranges, tether_tension_direction
    from lib.station import (
        RAM_ACC,
        RAM_GOAL_POSITION,
        RAM_GOAL_SPEED,
        connect,
        scaled_speeds,
        set_position,
        sync_write,
    )
    from session.home_pose import home_waypoints, load_home_pose

    tuning = load_tuning(getattr(args, "tuning", None))
    chain = load_chain()
    bus_serial = args.bus_serial or env_get("FOLLOWER_BUS_SERIAL")
    if not bus_serial:
        raise SystemExit("no follower bus serial: set FOLLOWER_BUS_SERIAL in .env or pass "
                         "--bus-serial.")
    host = args.station_host or env_get("STATION_HOST", "localhost")

    try:
        start_pose = load_home_pose(args.start_pose)
    except (SystemExit, FileNotFoundError):
        raise SystemExit(
            f"no CPM start pose saved at {args.start_pose}.\n\n"
            "CPM starts from its own pose, not from the session home: the home pose is where "
            "the arm is parked with nothing hooked, and a CPM stroke needs somewhere to "
            "travel to with the finger attached. Capture one:\n\n"
            "    .venv/bin/python scripts/lock_arm.py --release\n"
            "    # place the arm by hand where the stroke should begin\n"
            f"    .venv/bin/python scripts/set_home_pose.py --save --path {args.start_pose}\n"
        ) from None

    client = await connect(host=host)
    live = await read_pose(client, bus_serial, 5.0)
    missing = [m for m in ARM_MOTORS if m not in live]
    if missing:
        raise SystemExit(
            f"no position arrived for motor(s) {missing} within 5 s. Every direction here is "
            "computed from the pose, so an incomplete one makes all of them wrong. Is Station "
            "running and this bus connected?"
        )

    # --- the line, solved once, from the START pose rather than from wherever the arm is ---
    # Solving from the live pose would make the path depend on where the last run happened to
    # leave the arm, which is exactly the repeatability this mode exists to have.
    pose = {m: float(start_pose.get(m, live[m])) for m in ARM_MOTORS}
    if args.axis:
        direction = np.array([float(v) for v in args.axis.split(",")], dtype=float)
        direction = direction / np.linalg.norm(direction)
        source = "--axis, as given"
    else:
        direction = tether_tension_direction(chain, pose)
        source = "the gripper's own axis at the start pose"
    if args.level:
        before_z = float(direction[2])
        direction = level_direction(direction)
        source += f", levelled (z was {before_z:+.2f})"

    travel_dir = direction if args.toward_tension else -direction
    sense = ("TOWARD tension: the stroke stretches first, then releases"
             if args.toward_tension else
             "AWAY from tension: the stroke releases first, then comes back")

    try:
        tick_ranges = load_tick_ranges(bus_serial)
    except FileNotFoundError as exc:
        tick_ranges = None
        print(f"!! {exc}\n!! proceeding WITHOUT a joint-range check: a joint running past its "
              "real travel mid-stroke would not be caught before it happens.\n")

    motors = chain.choose_pull_motors(pose, direction)
    distance_m = args.distance_mm / 1000.0
    ok, why = chain.can_pull_straight(motors, pose, travel_dir, distance_m,
                                      tick_ranges=tick_ranges)
    print(f"!! cpm along the tether with joints {motors}")
    print(f"!! direction x={direction[0]:+.2f} y={direction[1]:+.2f} z={direction[2]:+.2f}, "
          f"from {source}")
    print(f"!! {why}")
    if not ok:
        raise SystemExit(
            "this start pose cannot follow that stroke; nothing has moved. Re-place the arm "
            "and re-save the start pose, shorten it with --distance-mm, or give the axis "
            "explicitly with --axis."
        )

    steps = max(1, round(args.distance_mm / args.step_mm))
    waypoints = chain.line_trajectory(motors, pose, travel_dir, distance_m, steps=steps)
    # Out along the waypoints, then back along the same ones in reverse, ending exactly where
    # it began. Computed once: `--reps` repeats this list, so no repetition can drift from the
    # line the feasibility check above already approved.
    round_trip = waypoints + list(reversed([pose] + waypoints[:-1]))

    ceiling_n = grams_force_to_newtons(args.ceiling_gf)
    print(f"\n{args.reps} x cpm, {args.distance_mm:.0f} mm out and back")
    print(f"  stroke   {steps} steps of {args.step_mm:.1f} mm, {args.settle:g}s between")
    print(f"  sense    {sense}")
    jog_profile = (args.speed, args.accel) == (JOG_PROVEN_SPEED, JOG_PROVEN_ACCEL)
    print(f"  arm      joints {motors} on {bus_serial}, speed {args.speed}, accel "
          f"{args.accel}{' (jog_coordinated.py profile)' if jog_profile else ''}")
    if args.no_camera:
        print("  force    NOT WATCHED (--no-camera). Run this way with nothing attached.")
    else:
        print(f"  force    camera watching only for a ceiling of {args.ceiling_gf:.0f} gf, "
              f"blind tolerance {args.blind_tolerance_s:g}s")
    print(f"  tuning   {tuning.profile}")

    gap = max((abs(int(live.get(m, pose[m])) - int(pose[m])) for m in pose), default=0)
    if gap > args.start_tolerance_ticks:
        print(f"\n!! the arm is {gap} ticks from the start pose. It will be walked there "
              f"first, in steps of {args.goto_max_step_ticks} ticks.")

    print("\nThe arm will MOVE along a fixed path, with a finger attached unless you are "
          "running --no-camera. Keep a hand on the e-stop.")
    if input("Type 'yes' to run: ").strip().lower() != "yes":
        print("aborted; nothing moved.")
        return 1

    supervisor = None
    supervisor_task = None
    source_cam = None

    # Every joint, not just the three pulling the line: the walk to the start pose below can
    # move any of the seven, and an un-set RAM_ACC leaves a joint on whatever accel a previous
    # run happened to leave it at.
    await sync_write(client, bus_serial, RAM_ACC, {
        m: args.accel.to_bytes(1, byteorder="little") for m in ARM_MOTORS
    })

    async def goto(prev: dict[int, float], target: dict[int, float],
                   move_motors: list[int] | None = None) -> None:
        """One step along the path. Two sync_writes, speed scaled per joint.

        The same two-packet pattern `scripts/jog_coordinated.py` uses, and for the same two
        reasons: one bus transaction so every joint's goal lands together, and a speed scaled
        to each joint's share of this step so they arrive together instead of whichever has
        furthest to go still moving after the others stopped.

        `move_motors` defaults to the three pull joints, which is right for a step of the
        line itself: every other joint is unchanged between waypoints there (`line_trajectory`
        only moves the joints it was given), so moving the rest too would be writing a
        no-op delta. It is NOT right for the walk to the start pose, where any of the seven
        joints may differ from where the arm currently sits, and every one of them has to be
        passed explicitly - `scripts/go_home.py` moves every key of each waypoint for exactly
        this reason, and the walk below has to match it rather than assume the pull joints
        are the only ones that could be out of place.
        """
        move_motors = motors if move_motors is None else move_motors
        deltas = {m: target[m] - prev[m] for m in move_motors}
        speeds = scaled_speeds(deltas, args.speed)
        await sync_write(client, bus_serial, RAM_GOAL_SPEED, {
            m: speeds[m].to_bytes(2, byteorder="little") for m in move_motors
        })
        await sync_write(client, bus_serial, RAM_GOAL_POSITION, {
            m: round(target[m]).to_bytes(2, byteorder="little") for m in move_motors
        })
        await asyncio.sleep(args.settle)

    if not args.no_camera:
        from lib.camera import open_source

        source_cam = open_source(args.camera)
        pipeline = build_pipeline(args.calib, outlier_persist=args.outlier_persist)
        supervisor = ForceSupervisor(source_cam, pipeline, ceiling_n, args.blind_tolerance_s)
        supervisor_task = asyncio.create_task(supervisor.run())
        # One reading before anything moves. Starting a stroke on a finger the camera cannot
        # see is the same mistake as continuing one blind, taken earlier.
        for _ in range(60):
            await asyncio.sleep(0.05)
            if supervisor.latest_n is not None:
                break
        if supervisor.latest_n is None:
            supervisor.cancel()
            supervisor_task.cancel()
            source_cam.release()
            raise SystemExit(
                "the camera never produced a force reading, so the ceiling would not be "
                "watched. Nothing has moved. Check the PESOLA is in frame and calibrated "
                "(python -m gauge_reader run), or pass --no-camera with nothing attached."
            )
        print(f"\nforce before moving: {supervisor.reading}")

    start_tip = chain.tip_position(pose)
    stopped = None
    reps_done = 0
    print(f"\n{'rep':>5s}  {'force':>7s}  {'along':>8s}  {'off line':>9s}  phase")
    print("-" * 56)
    try:
        # Walk to the start pose first, paced, so no servo is ever asked to cross a long way
        # in one jump near a hand.
        prev = {m: float(live[m]) for m in ARM_MOTORS}
        for way in home_waypoints(prev, {m: int(pose[m]) for m in pose},
                                  max_step_ticks=args.goto_max_step_ticks):
            way_f = {m: float(v) for m, v in way.items()}
            # Every joint in the waypoint, not just the three that pull the line: any of the
            # seven could be away from the start pose, and the walk here has to close all of
            # that gap, the same as scripts/go_home.py does.
            await goto(prev, way_f, move_motors=list(way_f))
            prev = {**prev, **way_f}
            if supervisor and supervisor.stop_reason:
                stopped = supervisor.stop_reason
                break
        if stopped is None:
            await goto(prev, pose, move_motors=list(ARM_MOTORS))

        for rep in range(1, args.reps + 1):
            if stopped:
                break
            prev = pose
            for i, way in enumerate(round_trip, start=1):
                if supervisor and supervisor.stop_reason:
                    stopped = supervisor.stop_reason
                    break
                await goto(prev, way)
                prev = way
                travel = chain.tip_position(way) - start_tip
                along = float(np.dot(travel, travel_dir))
                off = float(np.linalg.norm(travel - along * travel_dir))
                phase = "out" if i <= steps else "back"
                force = supervisor.reading if supervisor else "  --  "
                print(f"{rep:3d}/{args.reps}  {force:>7s}  {along*1000:6.1f}mm  "
                      f"{off*1000:7.2f}mm  {phase}")
            else:
                reps_done += 1
    except KeyboardInterrupt:
        stopped = "stopped by keyboard"
    finally:
        if supervisor:
            supervisor.cancel()
        if supervisor_task:
            supervisor_task.cancel()
        if stopped:
            print(f"\n!! {stopped}")
        # Back to the start pose on every path, including a stop. The stroke is bounded and
        # the path is known, so the safe place to end is the one the feasibility check
        # approved, not wherever a stop happened to leave it.
        print("returning to the start pose ...")
        for motor_id in motors:
            try:
                await set_position(client, bus_serial, motor_id, round(pose[motor_id]))
            except Exception as exc:  # noqa: BLE001
                print(f"!! could not return motor {motor_id} ({exc}). Release torque by hand: "
                      f"`python scripts/lock_arm.py --release --motors {motor_id}`.")
        await asyncio.sleep(1.0)
        if source_cam is not None:
            source_cam.release()

    print(f"\ncompleted {reps_done} of {args.reps} repetitions")
    if supervisor and supervisor.peak_n:
        print(f"peak force seen: {newtons_to_grams_force(supervisor.peak_n):.0f} gf "
              f"(ceiling {args.ceiling_gf:.0f} gf)")
    print(
        "\nThe arm is at the CPM start pose, which holds the finger. That is also where this "
        "script itself starts from, so running it again now needs no walk at all: leave the "
        "arm here while you are still testing or exercising.\n\n"
        "Only when you are done with cpm for this session, unhook the finger and park the "
        "arm at the session home instead:\n"
        "    .venv/bin/python scripts/go_home.py\n"
        "Going there with the finger still hooked would drag it through the arc between the "
        "two poses, which is why this is a choice to make on purpose, not a step after every "
        "run."
    )
    return 0


def main(argv=None) -> int:
    tuning = load_tuning(DEFAULT_TUNING_PATH)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--distance-mm", type=float, default=15.0,
                   help="how far the stroke travels before coming back. This is an open-loop "
                        "path, so it goes exactly this far whether or not that turns out to "
                        "be comfortable: shorten it with this flag before lengthening it. "
                        "15 mm matches scripts/jog_coordinated.py's own default")
    p.add_argument("--step-mm", type=float, default=1.0,
                   help="how far each commanded step moves. Smaller is smoother and slower, "
                        "and makes a bad step visible before it has gone far")
    p.add_argument("--settle", type=float, default=JOG_PROVEN_SETTLE,
                   help="seconds between steps. With --step-mm this sets the pace; --speed "
                        "sets how fast the servo chases each goal, which is a different knob")
    p.add_argument("--toward-tension", action="store_true",
                   help="stretch first and release second, instead of the default, which "
                        "travels away from tension first. The default cannot tighten anything "
                        "on a finger whatever else is wrong, which is why it is the default")
    p.add_argument("--start-pose", default=START_POSE_PATH,
                   help="the pose the stroke begins and ends at. Not the session home: that "
                        "is where the arm is parked with nothing hooked. Capture one with "
                        "scripts/set_home_pose.py --save --path ...")
    p.add_argument("--start-tolerance-ticks", type=int, default=20,
                   help="how far from the start pose counts as already there")
    p.add_argument("--goto-max-step-ticks", type=int, default=80,
                   help="largest per-joint move on the way to the start pose")
    p.add_argument("--speed", type=int, default=JOG_PROVEN_SPEED,
                   help="raw ST3215 goal-speed units, scaled per joint each step so all of "
                        "them arrive together. Defaults to scripts/jog_coordinated.py's own "
                        "bench-proven value, not config/session-tuning.json's "
                        "coordinated_speed: that number is tuned for the admittance "
                        "controller's continuous small corrections, a different motion from "
                        "this open-loop waypoint walk, and reusing it here made every step "
                        "look like a jerky restart instead of part of one smooth stroke")
    p.add_argument("--accel", type=int, default=JOG_PROVEN_ACCEL,
                   help="raw ST3215 acceleration units, written once. Same reasoning as "
                        "--speed: jog_coordinated.py's proven value, not the admittance "
                        "loop's")
    p.add_argument("--axis", default=None,
                   help="the tether's direction as x,y,z in the base frame, pointing the way "
                        "that INCREASES tension. Defaults to the gripper's own axis at the "
                        "start pose")
    p.add_argument("--level", action=argparse.BooleanOptionalAction, default=tuning.level,
                   help="drop the z component of the tether direction and renormalise, so the "
                        "stroke stays horizontal instead of following the gripper's tilt")
    p.add_argument("--no-camera", action="store_true",
                   help="run with no force supervision at all. For checking the path with "
                        "NOTHING ATTACHED, the same way scripts/jog_coordinated.py is used")
    p.add_argument("--ceiling-gf", type=float, default=tuning.safety.ceiling_gf,
                   help="stop everything if the force passes this. A ceiling, not a cap: it "
                        "is not a force the stroke works against, it is the number that says "
                        "something has gone wrong")
    p.add_argument("--blind-tolerance-s", type=float, default=2.0,
                   help="stop if the camera has produced no reading for this long while the "
                        "arm is moving. Generous compared with run_session.py because the "
                        "motion does not depend on the reading step by step, but not "
                        "infinite: a missing reading is never evidence of a safe force")
    p.add_argument("--outlier-persist", type=int, default=tuning.outlier_persist,
                   help="how many consecutive held-back readings the gauge reader's outlier "
                        "gate needs before trusting a jump")
    p.add_argument("--camera", default=None)
    p.add_argument("--calib", default=GAUGE_CALIB)
    p.add_argument("--tuning", default=str(DEFAULT_TUNING_PATH))
    p.add_argument("--station-host", default=None)
    p.add_argument("--bus-serial", default=None)
    args = p.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
