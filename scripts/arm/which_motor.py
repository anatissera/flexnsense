"""Which motor moves the gripper along the tether, from NormaCore's URDF and the live pose.

    .venv/bin/python scripts/arm/which_motor.py              # read the pose from Station
    .venv/bin/python scripts/arm/which_motor.py --check      # verify the URDF against this arm
    .venv/bin/python scripts/arm/which_motor.py --ticks 1:1970,2:1715,3:2498,4:2755,5:1907,6:1270,7:1282

Replaces identifying the motor by jogging one at a time and judging by eye, which is slow and
cannot reliably tell "moves the gripper along the pull axis" from "swings it sideways". The
URDF ships with the hardware and states each joint's axis exactly; `lib/kinematics.py` turns
that plus the live tick readings into the direction the gripper actually travels.

**Nothing moves.** This only reads positions and does geometry.

What it prints, per motor: the unit direction the gripper moves when that motor turns positive,
in the arm's base frame, and how far it travels per 100 ticks, which is the lever arm at this
pose. Pass `--axis` with the tether's direction to have them ranked by how well they line up.

The frame: x, y and z are the URDF's base frame, the same one Station's 3D view draws. Rather
than guess which way the room is, read the signs relatively: two motors whose directions are
nearly opposite move the gripper along the same line, in opposite senses.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.env import env_get  # noqa: E402
from lib.kinematics import (  # noqa: E402
    CENTRE_TICKS,
    check_against_calibration,
    load_chain,
)

CALIB_DIR = "config/calibration"


def parse_ticks(text: str) -> dict[int, float]:
    out = {}
    for part in text.split(","):
        motor, _, value = part.partition(":")
        out[int(motor)] = float(value)
    return out


async def read_pose(host: str, bus_serial: str, motor_ids: list[int], seconds: float) -> dict[int, float]:
    """Current tick reading per motor, from Station's state stream."""
    from lib.station import connect, follow_motor_state

    client = await connect(host=host)
    found: dict[int, float] = {}

    async def collect():
        async for bus, mid, motor in follow_motor_state(client):
            if bus == bus_serial and motor.position is not None:
                found[mid] = motor.position
                if all(m in found for m in motor_ids):
                    return

    try:
        await asyncio.wait_for(collect(), timeout=seconds)
    except asyncio.TimeoutError:
        pass
    return found


def report(chain, ticks: dict[int, float], axis: np.ndarray | None) -> None:
    tip = chain.tip_position(ticks)
    print(f"\ngripper at x={tip[0]*1000:+.0f} y={tip[1]*1000:+.0f} z={tip[2]*1000:+.0f} mm "
          f"(base frame, {np.linalg.norm(tip)*1000:.0f} mm from the base)\n")

    rows = []
    for motor_id in chain.motor_ids:
        if motor_id == 8:
            continue   # the gripper's own jaws, not an arm joint
        direction = chain.tip_velocity_direction(motor_id, ticks)
        mm_per_100 = chain.tip_speed_per_tick(motor_id, ticks) * 100 * 1000
        alignment = float(abs(np.dot(direction, axis))) if axis is not None else None
        rows.append((motor_id, direction, mm_per_100, alignment))

    if axis is not None:
        rows.sort(key=lambda r: -r[3])
        print("ranked by how closely each motor moves the gripper along your --axis:\n")

    head = f"{'motor':<7}{'direction the gripper moves':<34}{'mm per 100 ticks':<19}"
    print(head + ("alignment" if axis is not None else ""))
    print("-" * (len(head) + (9 if axis is not None else 0)))
    for motor_id, d, mm, alignment in rows:
        vec = f"x={d[0]:+.2f} y={d[1]:+.2f} z={d[2]:+.2f}"
        line = f"{motor_id:<7}{vec:<34}{mm:<19.1f}"
        if alignment is not None:
            mark = "  <-- best match" if alignment == max(r[3] for r in rows) else ""
            line += f"{alignment:.2f}{mark}"
        print(line)

    print("\nA motor with a large 'mm per 100 ticks' moves the gripper a long way per tick: good "
          "resolution of force, but less travel before the joint runs out of range.")
    if axis is None:
        print("Pass --axis x,y,z with the tether's direction to have these ranked for you. The "
              "two motors whose directions are most nearly opposite move along the same line.")


def plot(chain, ticks: dict[int, float], axis: np.ndarray | None) -> None:
    """Draw the arm and an arrow per motor showing which way it moves the gripper.

    matplotlib rather than a simulator: the question is geometric, and this is already a
    dependency. A full physics simulator (ManiSkill, SAPIEN, MuJoCo) would add gigabytes and a
    GPU dependency to draw the same arrows, and would answer the same question no better,
    because nothing here needs contact, dynamics or rendering. The simulation rig in isaac-sim/
    item 4 is a separate question (closed-loop behaviour), and that is where a real simulator
    earns its place.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    # The arm's shape: where each motor's joint sits, base to gripper.
    points = [np.zeros(3)]
    for motor_id in chain.motor_ids:
        if motor_id == 8:
            continue
        points.append(chain.joint_origin(motor_id, ticks)[1])
    tip = chain.tip_position(ticks)
    points.append(tip)
    pts = np.array(points) * 1000

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", color="0.4", linewidth=2, label="arm")
    for i, motor_id in enumerate([m for m in chain.motor_ids if m != 8], start=1):
        ax.text(pts[i, 0], pts[i, 1], pts[i, 2], f" m{motor_id}", fontsize=9)

    scale = 60.0  # mm, arrow length; direction is what matters, not magnitude
    colours = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#9467bd",
               5: "#17becf", 6: "#8c564b", 7: "#7f7f7f"}
    for motor_id in chain.motor_ids:
        if motor_id == 8:
            continue
        d = chain.tip_velocity_direction(motor_id, ticks)
        mm = chain.tip_speed_per_tick(motor_id, ticks) * 100 * 1000
        if mm < 0.01:
            continue   # motor 7 only orients the gripper; an arrow would be misleading
        ax.quiver(tip[0] * 1000, tip[1] * 1000, tip[2] * 1000,
                  d[0] * scale, d[1] * scale, d[2] * scale,
                  color=colours.get(motor_id, "0.3"), linewidth=2,
                  arrow_length_ratio=0.2, label=f"m{motor_id} ({mm:.0f} mm/100 ticks)")
    if axis is not None:
        ax.quiver(tip[0] * 1000, tip[1] * 1000, tip[2] * 1000,
                  axis[0] * scale * 1.4, axis[1] * scale * 1.4, axis[2] * scale * 1.4,
                  color="red", linewidth=2.5, arrow_length_ratio=0.2, label="tether axis")

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("Which way each motor moves the gripper, at this pose")
    lim = float(np.abs(pts).max()) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


def run_check(chain) -> int:
    """Compare the URDF's limits against this arm's hand-measured calibration."""
    files = sorted(Path(CALIB_DIR).glob("follower-*.json"))
    if not files:
        raise SystemExit(f"no follower calibration in {CALIB_DIR}/ to check against.")
    ok_all = True
    for f in files:
        ranges = json.loads(f.read_text()).get("ranges")
        if not ranges:
            continue
        print(f"\n{f.name}: the firmware's measured ranges against the URDF's limits")
        print(f"{'motor':<7}{'measured deg':<15}{'urdf deg':<12}{'diff':<9}")
        for motor_id, measured, urdf_span, agrees in check_against_calibration(chain, ranges):
            note = "" if agrees else ("  <-- gripper, calibrated by where the jaws close"
                                      if motor_id == 8 else "  <-- DIFFERS")
            if not agrees and motor_id != 8:
                ok_all = False
            print(f"{motor_id:<7}{measured:<15.1f}{urdf_span:<12.1f}{measured - urdf_span:+.1f}{note}")
    print("\nThese agreeing is what makes 'rev_motor_0N is Station's motor N' a checked fact "
          "rather than an assumption; docs/elrobot.md flags it as unverified otherwise.")
    return 0 if ok_all else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticks", help="pose as 1:1970,2:1715,... instead of reading Station")
    p.add_argument("--axis", help="the tether's direction as x,y,z; motors get ranked against it")
    p.add_argument("--check", action="store_true",
                   help="verify the URDF against config/calibration and exit")
    p.add_argument("--plot", action="store_true",
                   help="also draw the arm and the per-motor directions (matplotlib)")
    p.add_argument("--station-host", default=None)
    p.add_argument("--bus-serial", default=None)
    p.add_argument("--seconds", type=float, default=5.0)
    args = p.parse_args(argv)

    chain = load_chain()
    if args.check:
        return run_check(chain)

    axis = None
    if args.axis:
        axis = np.array([float(v) for v in args.axis.split(",")])
        axis = axis / np.linalg.norm(axis)

    if args.ticks:
        ticks = parse_ticks(args.ticks)
    else:
        bus_serial = args.bus_serial or env_get("FOLLOWER_BUS_SERIAL")
        if not bus_serial:
            raise SystemExit("no follower bus serial: set FOLLOWER_BUS_SERIAL in .env, pass "
                             "--bus-serial, or give the pose with --ticks.")
        host = args.station_host or env_get("STATION_HOST", "localhost")
        motor_ids = [m for m in chain.motor_ids if m != 8]
        print(f"reading the pose from Station at {host}, bus {bus_serial} ...")
        ticks = asyncio.run(read_pose(host, bus_serial, motor_ids, args.seconds))
        if not ticks:
            raise SystemExit("no motor positions arrived. Is Station running and this bus connected?")
        missing = [m for m in motor_ids if m not in ticks]
        if missing:
            print(f"warning: no reading for motor(s) {missing}; treating them as centred "
                  f"({CENTRE_TICKS} ticks), so their geometry is nominal rather than actual.")
        print("pose: " + ", ".join(f"m{m}={int(t)}" for m, t in sorted(ticks.items())))

    report(chain, ticks, axis)
    if args.plot:
        plot(chain, ticks, axis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
