"""Simulate the tether pull: one joint's arc against a coordinated straight line.

    .venv/bin/python scripts/simulate_pull.py                      # animate, live pose
    .venv/bin/python scripts/simulate_pull.py --save pull.mp4      # write a video
    .venv/bin/python scripts/simulate_pull.py --distance 90 --motors 3,4

Why this exists: a single revolute joint moves the gripper along an **arc**. Over a few
millimetres that is indistinguishable from a line, which is why a first bench test would not
catch it, but one full-range repetition needs 72 to 88 mm of travel
(`config/calibration/bench.md`), and over that distance one joint wanders more than 20 mm off
the tether's axis. The tether does not bend, so that deviation is either side-load on the
finger or slack, and neither is the exercise the controller thinks it is running.

The fix is inverse kinematics: several joints moving together so the tip tracks a line. This
animates both, side by side, so the difference is visible rather than asserted.

Geometry only, from NormaCore's URDF: no contact, no dynamics, no GPU, nothing installed
beyond matplotlib. The physics-level simulator (`sim/`, ManiSkill) is for the closed-loop
question isaac-sim/ exists for, which is a different question from "is the path straight".
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.env import env_get  # noqa: E402
from lib.kinematics import load_chain  # noqa: E402

#: Fallback pose, read off the arm on 2026-11-15, for when Station is not running.
DEFAULT_POSE = {1: 1972, 2: 1714, 3: 2963, 4: 2225, 5: 1901, 6: 1234, 7: 1224}


def parse_ticks(text: str) -> dict[int, float]:
    out = {}
    for part in text.split(","):
        motor, _, value = part.partition(":")
        out[int(motor)] = float(value)
    return out


async def read_pose(host: str, bus_serial: str, motor_ids: list[int], seconds: float) -> dict:
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


def arm_points(chain, ticks) -> np.ndarray:
    """Joint origins from the base out to the tip, for drawing the arm."""
    pts = [np.zeros(3)]
    for motor_id in chain.motor_ids:
        if motor_id == 8:
            continue
        pts.append(chain.joint_origin(motor_id, ticks)[1])
    pts.append(chain.tip_position(ticks))
    return np.array(pts) * 1000


def single_joint_path(chain, motors_single: int, pose, direction, distance_m, steps):
    """What one joint alone does: rotate it until the tip has travelled `distance_m` of arc."""
    per_tick = chain.tip_speed_per_tick(motors_single, pose)
    total_ticks = distance_m / per_tick if per_tick > 1e-9 else 0.0
    out = []
    for i in range(1, steps + 1):
        p = dict(pose)
        p[motors_single] = pose[motors_single] + total_ticks * i / steps
        out.append(p)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motors", default="3,4",
                   help="joints the coordinated pull may use (default 3,4)")
    p.add_argument("--single", type=int, default=4,
                   help="the joint to show moving alone, for comparison (default 4)")
    p.add_argument("--distance", type=float, default=90.0, help="pull distance in mm")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--ticks", help="pose as 1:1972,2:1714,... instead of reading Station")
    p.add_argument("--axis", help="tether direction x,y,z (default: the --single joint's own)")
    p.add_argument("--save", help="write an mp4 or gif here instead of opening a window")
    p.add_argument("--station-host", default=None)
    p.add_argument("--bus-serial", default=None)
    args = p.parse_args(argv)

    chain = load_chain()
    motors = [int(m) for m in args.motors.split(",")]

    if args.ticks:
        pose = parse_ticks(args.ticks)
    else:
        bus_serial = args.bus_serial or env_get("FOLLOWER_BUS_SERIAL")
        host = args.station_host or env_get("STATION_HOST", "localhost")
        pose = {}
        if bus_serial:
            print(f"reading the pose from Station at {host} ...")
            try:
                pose = asyncio.run(read_pose(host, bus_serial,
                                             [m for m in chain.motor_ids if m != 8], 4.0))
            except Exception as exc:  # noqa: BLE001
                print(f"  could not read Station ({exc})")
        if not pose:
            print("using the recorded pose from 2026-11-15 instead.")
            pose = dict(DEFAULT_POSE)
    print("pose: " + ", ".join(f"m{m}={int(t)}" for m, t in sorted(pose.items())))

    direction = (np.array([float(v) for v in args.axis.split(",")]) if args.axis
                 else chain.tip_velocity_direction(args.single, pose))
    direction = direction / np.linalg.norm(direction)
    distance_m = args.distance / 1000.0

    worst, short = chain.line_error(motors, pose, direction, distance_m, args.steps)
    print(f"\npulling {args.distance:.0f} mm along {np.round(direction, 3)}")
    print(f"  motor {args.single} alone     : arc, deviates from the line")
    print(f"  motors {args.motors} together : worst deviation {worst*1000:.2f} mm, "
          f"shortfall {short*1000:.2f} mm")

    coordinated = chain.line_trajectory(motors, pose, direction, distance_m, args.steps)
    single = single_joint_path(chain, args.single, pose, direction, distance_m, args.steps)

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    p0 = chain.tip_position(pose) * 1000
    ideal = np.array([p0 + direction * 1000 * (distance_m * 1000) * t / 1000
                      for t in np.linspace(0, 1, 2)])

    fig = plt.figure(figsize=(13, 6))
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    titles = [f"motor {args.single} alone (an arc)",
              f"motors {args.motors} coordinated (a line)"]
    paths = [single, coordinated]

    lim = float(np.abs(arm_points(chain, pose)).max()) * 1.15
    artists = []
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
        ax.plot(ideal[:, 0], ideal[:, 1], ideal[:, 2], "--", color="red", linewidth=1.5,
                label="the tether's line")
        arm, = ax.plot([], [], [], "-o", color="0.35", linewidth=2.5, markersize=4, label="arm")
        trail, = ax.plot([], [], [], "-", color="#1f77b4", linewidth=2, label="tip path")
        ax.legend(loc="upper left", fontsize=8)
        artists.append((arm, trail))

    trails = [[], []]

    def frame(i):
        for k, (path, (arm, trail)) in enumerate(zip(paths, artists)):
            pts = arm_points(chain, path[i])
            arm.set_data(pts[:, 0], pts[:, 1])
            arm.set_3d_properties(pts[:, 2])
            trails[k].append(chain.tip_position(path[i]) * 1000)
            t = np.array(trails[k])
            trail.set_data(t[:, 0], t[:, 1])
            trail.set_3d_properties(t[:, 2])
        return [a for pair in artists for a in pair]

    anim = FuncAnimation(fig, frame, frames=len(coordinated), interval=40, blit=False)
    plt.tight_layout()
    if args.save:
        print(f"writing {args.save} ...")
        anim.save(args.save, fps=25)
        print("done")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
