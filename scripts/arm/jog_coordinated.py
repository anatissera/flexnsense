"""Walk the gripper along the tether's line and back, with no camera and no force reading.

    .venv/bin/python scripts/arm/jog_coordinated.py

Answers one question on its own, with nothing else in the loop: do the three joints
`scripts/session/run_session.py --coordinated` picked actually move the gripper the way the geometry
says they should, on *this* arm, not just in the URDF. `control.loop.ControlLoop` adds a
force reading, a safety gate and an admittance law on top of exactly this motion; if this
script does not track straight, nothing built on top of it will either, and the camera was
never the reason.

Open-loop: every tick it sends comes from `lib.kinematics.Chain.line_trajectory`, computed
once before the first command, not from a sensor. That is what makes this safe to run with no
camera and no force cap: there is no force to answer to and nothing here increases tension, so
the worst it can do is retrace the planned line, which it refuses to do at all if
`can_pull_straight` says this pose cannot follow it (the same check `run_session.py
--coordinated` runs before moving anything).

Moves slowly and in small steps (`--step-mm`, 1 mm by default, with `--settle` between them) so
a bad step is visible and stoppable well before it has gone far, then retraces the exact same
waypoints to return, rather than recomputing a path back that could drift from the one it took
out.

Speed is two separate knobs, and they are not the same control. `--step-mm` and `--settle`
change how often a new goal position is sent; `--speed` (and `--accel`), written to the ST3215's
own RAM_GOAL_SPEED and RAM_ACC registers, change how fast the servo actually chases a goal.
Lowering `--settle` alone just sends the next goal before the servo finished reaching the last
one, which does not make the arm move faster, only more raggedly.

`--speed` is scaled per joint, per step, not written once at the same value for all three. A
straight pull does not move every joint the same number of ticks per step; at a shared speed,
whichever joint has furthest to go is still moving once the others have stopped, which looked
like vertical vibration on this bench even after fixing the timing skew between joints
(`scaled_speeds`). Repeat the whole walk with `--reps` to watch this settle in, rather than
judging it from a single pass.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm1.lib.env import env_get  # noqa: E402
from arm1.lib.kinematics import ARM_MOTORS, level_direction  # noqa: E402


async def read_pose(client, bus_serial: str, seconds: float) -> dict[int, float]:
    from arm1.lib.station import follow_motor_state

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


async def run(args) -> int:
    import numpy as np

    from arm1.lib.kinematics import load_chain, load_tick_ranges, tether_tension_direction
    from arm1.lib.station import (
        RAM_ACC,
        RAM_GOAL_POSITION,
        RAM_GOAL_SPEED,
        connect,
        scaled_speeds,
        set_position,
        sync_write,
    )

    chain = load_chain()
    bus_serial = args.bus_serial or env_get("FOLLOWER_BUS_SERIAL")
    if not bus_serial:
        raise SystemExit("no follower bus serial: set FOLLOWER_BUS_SERIAL in .env or pass "
                         "--bus-serial.")
    host = args.station_host or env_get("STATION_HOST", "localhost")

    client = await connect(host=host)
    pose = await read_pose(client, bus_serial, 5.0)
    missing = [m for m in ARM_MOTORS if m not in pose]
    if missing:
        raise SystemExit(
            f"no position arrived for motor(s) {missing} within 5 s; the pose has to be "
            "complete or every direction computed from it is wrong. Is Station running and "
            "this bus connected?"
        )

    if not args.no_home_check:
        # This script only ever promises to return to wherever it read `pose` at the top of
        # THIS run - it has no way to know that is not the true home. If a previous run ended
        # a hair off (backlash, a joint not perfectly settled before the readback), the next
        # run treats that slightly-off position as correct and returns to it exactly, and the
        # error has nowhere to go but to repeat. Over several runs that is a one-way walk away
        # from config/calibration/home-pose.json, the actual reference, with nothing in this
        # script's own bookkeeping able to notice it because it never looks at that file.
        try:
            from arm1.session.home_pose import distance_from_home, load_home_pose

            gap = distance_from_home(pose, load_home_pose())
            if gap > 20:
                print(f"!! {gap} ticks from the saved home pose before this run has even "
                      "started. Each run returns to where IT started, not to home, so this "
                      "does not self-correct - run scripts/arm/go_home.py first.\n")
        except SystemExit:
            pass   # no home pose saved; nothing to compare against

    if args.axis:
        direction = np.array([float(v) for v in args.axis.split(",")], dtype=float)
        direction = direction / np.linalg.norm(direction)
        source = "--axis, as given"
    else:
        direction = tether_tension_direction(chain, pose)
        source = "the gripper's own axis at this pose"

    if args.level:
        before_z = float(direction[2])
        direction = level_direction(direction)
        source += f", levelled (z was {before_z:+.2f})"
    yield_dir = -direction  # the sense this walks: away from tension, same as ControlLoop

    try:
        tick_ranges = load_tick_ranges(bus_serial)
    except FileNotFoundError as exc:
        tick_ranges = None
        print(f"!! {exc}\n!! proceeding WITHOUT a joint-range check: a joint running past its "
              "real travel mid-walk would not be caught before it happens.\n")

    motors = chain.choose_pull_motors(pose, direction)
    distance_m = args.distance_mm / 1000.0
    ok, why = chain.can_pull_straight(motors, pose, yield_dir, distance_m,
                                      tick_ranges=tick_ranges)
    print(f"joints {motors}, direction x={direction[0]:+.2f} y={direction[1]:+.2f} "
          f"z={direction[2]:+.2f} (from {source})")
    print(why)
    if not ok:
        raise SystemExit(
            "this pose cannot follow that pull; nothing has moved. Re-pose the arm, shorten "
            "it with --distance-mm, or give the axis explicitly with --axis."
        )

    steps = max(1, round(args.distance_mm / args.step_mm))
    waypoints = chain.line_trajectory(motors, pose, yield_dir, distance_m, steps=steps)
    # The full round trip as one list of targets to visit in order: out along the waypoints,
    # then back along the same ones in reverse, ending exactly at `pose`. One rep is this list
    # once; `--reps` repeats it, out-and-back-and-out-and-back, without ever recomputing the
    # path, so repetition cannot drift from the line the first pass already checked.
    round_trip = waypoints + list(reversed([pose] + waypoints[:-1]))
    print(f"\n{steps} steps of {args.step_mm:.1f} mm out, then the same {steps} back, "
          f"x{args.reps}.")
    print("Nothing is attached to anything. The arm will move, slowly, then return.")
    if input("Type 'nothing attached' to confirm: ").strip().lower() != "nothing attached":
        print("aborted; nothing moved.")
        return 1

    print(f"setting accel {args.accel} on joints {motors} (speed is set per step; see below) ...")
    await sync_write(client, bus_serial, RAM_ACC, {
        motor_id: args.accel.to_bytes(1, byteorder="little") for motor_id in motors
    })

    start = chain.tip_position(pose)

    async def goto(prev: dict[int, float], target: dict[int, float]) -> None:
        # Two sync_write packets, not one, and not per-motor at all: the first fixes the
        # timing skew between joints (every joint's new goal lands in the same bus
        # transaction, not three separate round trips that can arrive a few milliseconds
        # apart), the second fixes a different skew that survives even with that fix, between
        # joints that have to travel different distances this step. A shared speed means
        # whichever joint has further to go is still moving after the others stop, which is
        # the wobble this showed on the bench even after the first fix. Scaling speed to each
        # joint's share of this step's distance (`scaled_speeds`) makes them land together
        # instead, keeping the tip on the line during the step, not just at its two ends.
        deltas = {m: target[m] - prev[m] for m in motors}
        speeds = scaled_speeds(deltas, args.speed)
        await sync_write(client, bus_serial, RAM_GOAL_SPEED, {
            m: speeds[m].to_bytes(2, byteorder="little") for m in motors
        })
        await sync_write(client, bus_serial, RAM_GOAL_POSITION, {
            m: round(target[m]).to_bytes(2, byteorder="little") for m in motors
        })
        await asyncio.sleep(args.settle)

    try:
        for rep in range(1, args.reps + 1):
            prev = pose
            label = f"rep {rep}/{args.reps}" if args.reps > 1 else "out"
            for i, way in enumerate(round_trip, start=1):
                await goto(prev, way)
                prev = way
                if i <= steps:
                    travel = chain.tip_position(way) - start
                    along = float(np.dot(travel, yield_dir))
                    off = float(np.linalg.norm(travel - along * yield_dir))
                    print(f"  {label}  out  {i:3d}/{steps}  given way {along*1000:5.1f} mm  "
                          f"off the line {off*1000:5.2f} mm")
                elif i == steps + 1:
                    print(f"  {label}  returning ...")
    finally:
        print("\nsetting every joint back to its exact starting tick ...")
        for motor_id in motors:
            try:
                await set_position(client, bus_serial, motor_id, round(pose[motor_id]))
            except Exception as exc:  # noqa: BLE001
                print(f"!! could not return motor {motor_id} to {round(pose[motor_id])} ticks "
                      f"({exc}). Release torque by hand: `python scripts/arm/lock_arm.py --release "
                      f"--motors {motor_id}`, or power down Station.")
        # Long enough for the servo's own step response to actually finish, not just for the
        # command to be sent: the readback right after this is the whole point of measuring
        # rather than eyeballing "did it go back", and reading it mid-overshoot would answer
        # the wrong question.
        await asyncio.sleep(1.0)

    final = await read_pose(client, bus_serial, 2.0)
    print("\nback at: " + ", ".join(
        f"m{m}={int(final[m])} ({int(final[m]) - int(pose[m]):+d})" if m in final else f"m{m}=?"
        for m in motors
    ))
    if all(m in final for m in ARM_MOTORS):
        drift_mm = (chain.tip_position(final) - chain.tip_position(pose)) * 1000.0
        print(f"gripper vs. where it started: dx={drift_mm[0]:+.2f} dy={drift_mm[1]:+.2f} "
              f"dz={drift_mm[2]:+.2f} mm")
        if abs(drift_mm[2]) > 1.0:
            print(
                "That dz is backlash, not a bug in the return: the commanded tick target was "
                "exactly the one it started at (see the per-motor line above), but the final "
                "approach came from a different direction than however it got to that tick the "
                "first time, and a gear train does not land in the same physical spot from both "
                "directions. Nothing with only position commands fixes that, including this "
                "script. It is also why the real controller (control.loop.ControlLoop) drives "
                "off the PESOLA's force, not off ticks alone: the force reading corrects for "
                "exactly this kind of mechanical slack every cycle, which this open-loop jog "
                "has no sensor to do."
            )
    else:
        print("(could not read every joint back to check; see the per-motor line above)")

    print("\ndone.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distance-mm", type=float, default=15.0,
                   help="how far to walk out before coming back")
    p.add_argument("--step-mm", type=float, default=1.0,
                   help="distance per commanded step; smaller is slower and safer to watch")
    p.add_argument("--settle", type=float, default=0.3,
                   help="pause after each step, seconds")
    p.add_argument("--reps", type=int, default=1,
                   help="how many times to repeat the full out-and-back, without "
                        "recomputing the path (stretch, return, stretch, return, ...)")
    p.add_argument("--speed", type=int, default=700,
                   help="raw ST3215 goal-speed units, scaled per joint every step. 700 is "
                        "what control.loop.CoordinatedAxis defaults to, tuned here: a "
                        "straight line with no visible wobble at a pace that kept up with a "
                        "hand-paced pull. This is the actual speed control; --settle and "
                        "--step-mm only change how often a new goal is sent, not how fast "
                        "the servo chases one")
    p.add_argument("--accel", type=int, default=50,
                   help="raw ST3215 acceleration units. NormaCore's own autocalibration "
                        "script uses 50; lower ramps up more gently")
    p.add_argument("--axis", default=None,
                   help="the tether's direction as x,y,z, pointing the way that INCREASES "
                        "tension. Defaults to the gripper's own axis at the live pose")
    p.add_argument("--level", action="store_true",
                   help="drop the z component of the direction (default or --axis) and "
                        "renormalise, so the pull stays in the horizontal plane instead of "
                        "following the gripper's own small tilt")
    p.add_argument("--station-host", default=None)
    p.add_argument("--bus-serial", default=None)
    p.add_argument("--no-home-check", action="store_true",
                   help="skip comparing the starting pose against the saved home pose")
    args = p.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
