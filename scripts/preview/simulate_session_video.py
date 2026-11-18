"""Render the coordinated controller responding to a pull, as a video.

    .venv/bin/python scripts/preview/simulate_session_video.py
    .venv/bin/python scripts/preview/simulate_session_video.py --cap-gf 200 --pull-mm-s 6

**Nothing moves and no camera opens.** This is `control.simulate`'s question, asked of the
coordinated pull and answered in pictures instead of a summary line: a finger pulls away at a
steady speed, the spring tension rises, the gate holds until the cap and then the admittance
law gives way, and three joints move together to follow the tether's line.

What it is for is seeing the two things a number cannot show. First, that the arm holds still
while the force climbs and only then starts moving, which is the gate doing its job rather
than the arm drifting. Second, that the gripper travels *along* the tether rather than arcing
across it: the deviation trace is the whole reason the controller coordinates three joints
instead of driving the single best-aligned one, and an arc shows up there as a rising curve
while a straight pull stays flat.

Honest about what it models. The arm is assumed to reach the position it is commanded, so this
tests the **control law and the geometry**, not the servos' dynamics, exactly as
`control.simulate` does. The force comes from a spring of the bench-measured stiffness acting
along the real 3D distance between the gripper and the finger, so a pull that wanders off the
line does show up as a force that fails to come down, which is the failure worth seeing.

Every decision in the video comes from the real control code: `control.safety.SafetyGate`,
`control.admittance`, and `control.loop.CoordinatedAxis`. There is no second copy of the
control law in here for the animation to flatter.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.admittance import AdmittanceParams, step_from_velocity, yield_velocity  # noqa: E402
from control.loop import CoordinatedAxis  # noqa: E402
from control.safety import Action, SafetyGate, SafetyLimits  # noqa: E402
from control.simulate import SPRING_N_PER_M  # noqa: E402
from lib.kinematics import load_chain, tether_tension_direction  # noqa: E402
from lib.units import grams_force_to_newtons, newtons_to_grams_force  # noqa: E402

#: The pose the arm stood in on the bench on 2026-11-15, and the one the geometry in
#: `scripts/session/run_session.py --coordinated` was checked against.
DEFAULT_POSE = {1: 2045, 2: 1468, 3: 2458, 4: 2775, 5: 1877, 6: 1524, 7: 3274}

#: Slack length of the tether at rest, metres. Only sets where the finger is drawn; the force
#: depends on the stretch, not on this.
REST_LENGTH_M = 0.12


@dataclass
class Frame:
    """One simulated cycle, everything the animation needs to draw it."""

    t: float
    force_n: float
    action: str
    reason: str
    pose: dict[int, float]
    tip: np.ndarray
    finger: np.ndarray
    deviation_m: float
    travel_m: float


@dataclass
class Sim:
    frames: list[Frame] = field(default_factory=list)
    motors: list[int] = field(default_factory=list)
    released: str = ""


def simulate(pose: dict[int, float], limits: SafetyLimits, params: AdmittanceParams,
             pull_speed_m_s: float, k_n_per_m: float, fps: float, duration_s: float,
             max_travel_mm: float) -> Sim:
    """A finger pulls away along the tether; the gate and the admittance law answer each frame.

    The force is computed from the real 3D distance between the gripper and the finger, not
    from the commanded travel. That is the point of simulating the geometry at all: an arm that
    gives way sideways moves its full commanded step and still fails to slacken the tether, and
    only a distance-based force shows that.
    """
    chain = load_chain()
    tension_dir = tether_tension_direction(chain, pose)
    coordinated = CoordinatedAxis.for_pose(pose, tension_dir, chain,
                                           distance_m=max_travel_mm / 1000.0,
                                           travel_direction=-tension_dir)

    gate = SafetyGate(limits)
    pose = dict(pose)
    start_tip = chain.tip_position(pose)
    # The finger sits out along the gripper's axis, which is +(-tension_dir), and is pulled
    # further out from there. Yielding means the gripper follows it.
    pull_dir = -tension_dir
    finger = start_tip + pull_dir * REST_LENGTH_M

    sim = Sim(motors=list(coordinated.motors))
    # The same sub-tick accumulator `ControlLoop` keeps. Simulating without it would draw a
    # straighter arm than the real one, which is the opposite of what this video is for.
    residual: dict[int, float] = {}
    dt = 1.0 / fps
    for i in range(int(duration_s * fps)):
        t = i * dt
        finger = finger + pull_dir * (pull_speed_m_s * dt)
        tip = chain.tip_position(pose)

        stretch = float(np.linalg.norm(finger - tip)) - REST_LENGTH_M
        force_n = k_n_per_m * max(0.0, stretch)

        travel = tip - start_tip
        along = float(np.dot(travel, pull_dir))
        deviation = float(np.linalg.norm(travel - along * pull_dir))

        decision = gate.check(force_n, age_s=0.0, travel_m=abs(along))
        sim.frames.append(Frame(t, force_n, decision.action.value, decision.reason,
                                dict(pose), tip, finger.copy(), deviation, along))
        if decision.action is Action.RELEASE:
            sim.released = decision.reason
            break
        if decision.action is Action.YIELD:
            v = yield_velocity(force_n, limits.f_threshold_n, params)
            step_m = step_from_velocity(v, dt, params)
            if step_m > 0:
                deltas = coordinated.ticks_for(step_m, pose, chain, residual=residual)
                for motor_id, delta in deltas.items():
                    pose[motor_id] = pose[motor_id] + delta
    return sim


def camera_basis(elev_deg: float, azim_deg: float) -> np.ndarray:
    """Rows `(right, up, towards the viewer)` for a camera looking at the origin.

    Projection is done here rather than with a 3D axes because the arm is 22k triangles and
    matplotlib's 3D pipeline redraws all of them per frame. Projecting with numpy and handing
    the result to a flat `PolyCollection` renders a frame in about 0.15 s instead, which is
    the difference between a video that takes a minute and one that takes half an hour.
    """
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    forward = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right = right / np.linalg.norm(right)
    return np.array([right, np.cross(right, forward), forward])


def shade(triangles: np.ndarray, base_rgb, light=(0.3, 0.4, 0.86)) -> np.ndarray:
    """Flat per-face shading, so the arm reads as a solid object rather than a silhouette."""
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    light = np.array(light) / np.linalg.norm(light)
    lit = np.clip(np.abs(normals @ light), 0.0, 1.0)
    return np.array(base_rgb) * (0.35 + 0.65 * lit)[:, None]


def project(triangles: np.ndarray, basis: np.ndarray):
    """`(polygons, order)`: triangles as 2D polygons, back to front.

    Painter's algorithm on the mean depth of each face. It is not exact for interpenetrating
    triangles, but these are rigid parts that do not interpenetrate, so it is correct here and
    needs no depth buffer.
    """
    view = triangles @ basis.T
    order = np.argsort(-view[:, :, 2].mean(1))
    return view[order][:, :, :2] * 1000.0, order


def render(sim: Sim, out: Path, cap_gf: float, ceiling_gf: float, fps: float,
           elev: float = 18.0, azim: float = -62.0, cell_m: float = 0.003) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    from matplotlib.collections import PolyCollection

    from lib.meshes import load_visuals, posed_triangles

    chain = load_chain()
    visuals = load_visuals(cell_m=cell_m)
    basis = camera_basis(elev, azim)
    frames = sim.frames
    t = np.array([f.t for f in frames])
    gf = np.array([newtons_to_grams_force(f.force_n) for f in frames])
    dev_mm = np.array([f.deviation_m * 1000 for f in frames])
    travel_mm = np.array([f.travel_m * 1000 for f in frames])
    ticks = {m: np.array([f.pose[m] for f in frames]) for m in sim.motors}

    fig = plt.figure(figsize=(14, 7.5))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.3, 1], hspace=0.55, wspace=0.18)
    ax3d = fig.add_subplot(grid[:, 0])
    ax_f = fig.add_subplot(grid[0, 1])
    ax_d = fig.add_subplot(grid[1, 1])
    ax_m = fig.add_subplot(grid[2, 1])

    # Pose every frame's mesh up front: it is 5 ms each, and knowing the extent of the whole
    # animation in advance is what keeps the camera still instead of rescaling under the arm.
    posed = [posed_triangles(chain, visuals, f.pose)[0] for f in frames]
    tips2d = np.array([(f.tip @ basis.T)[:2] for f in frames]) * 1000.0
    fingers2d = np.array([(f.finger @ basis.T)[:2] for f in frames]) * 1000.0

    extent = np.concatenate([(posed[0] @ basis.T).reshape(-1, 3)[:, :2] * 1000.0,
                             (posed[-1] @ basis.T).reshape(-1, 3)[:, :2] * 1000.0,
                             fingers2d])
    pad = 25.0
    ax3d.set_xlim(extent[:, 0].min() - pad, extent[:, 0].max() + pad)
    ax3d.set_ylim(extent[:, 1].min() - pad, extent[:, 1].max() + pad)
    ax3d.set_aspect("equal")
    ax3d.axis("off")
    ax3d.set_title("the arm giving way along the tether", fontsize=11, color="0.3")

    mesh = PolyCollection([], edgecolors="none")
    ax3d.add_collection(mesh)
    ideal_line, = ax3d.plot([], [], "--", color="#2ca02c", lw=1.3, zorder=5,
                            label="the tether's line")
    tether_line, = ax3d.plot([], [], "-", color="#d62728", lw=2.5, zorder=6, label="tether")
    trail_line, = ax3d.plot([], [], "-", color="#1f77b4", lw=2.5, zorder=7,
                            label="path of the gripper")
    finger_dot, = ax3d.plot([], [], "o", color="#d62728", ms=9, zorder=8, label="finger")
    ax3d.legend(loc="lower left", fontsize=8, framealpha=0.85)

    for ax, series, label, colour in (
        (ax_f, gf, "force (gf)", "#d62728"),
        (ax_d, dev_mm, "off the tether's line (mm)", "#9467bd"),
    ):
        ax.set_xlim(0, t[-1] if t[-1] > 0 else 1)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.25)
        ax.plot(t, series, color=colour, alpha=0.18, lw=1)

    ax_f.axhline(cap_gf, color="#ff7f0e", ls="--", lw=1.2)
    ax_f.annotate(f"cap {cap_gf:.0f} gf", (0.01, cap_gf), xycoords=("axes fraction", "data"),
                  fontsize=8, color="#ff7f0e", va="bottom")
    ax_f.axhline(ceiling_gf, color="#d62728", ls=":", lw=1.2)
    ax_f.annotate(f"ceiling {ceiling_gf:.0f} gf", (0.01, ceiling_gf),
                  xycoords=("axes fraction", "data"), fontsize=8, color="#d62728", va="bottom")
    ax_f.set_ylim(0, max(ceiling_gf * 1.15, gf.max() * 1.15))
    ax_d.set_ylim(0, max(2.0, dev_mm.max() * 1.3))

    f_line, = ax_f.plot([], [], color="#d62728", lw=2)
    d_line, = ax_d.plot([], [], color="#9467bd", lw=2)

    ax_m.set_xlim(0, t[-1] if t[-1] > 0 else 1)
    ax_m.set_ylabel("joint (ticks from start)", fontsize=9)
    ax_m.set_xlabel("seconds", fontsize=9)
    ax_m.grid(alpha=0.25)
    m_lines = {}
    for motor_id in sim.motors:
        rel = ticks[motor_id] - ticks[motor_id][0]
        ax_m.plot(t, rel, alpha=0.18, lw=1)
        m_lines[motor_id], = ax_m.plot([], [], lw=2, label=f"motor {motor_id}")
    ax_m.legend(loc="upper left", fontsize=8, ncol=3)
    span = max(1.0, max(abs(ticks[m] - ticks[m][0]).max() for m in sim.motors) * 1.25)
    ax_m.set_ylim(-span, span)

    readout = fig.text(0.015, 0.965, "", fontsize=15, family="monospace", va="top")
    status = fig.text(0.015, 0.915, "", fontsize=11, va="top")

    def update(i):
        frame = frames[i]
        polys, order = project(posed[i], basis)
        mesh.set_verts(polys)
        mesh.set_facecolors(shade(posed[i], (0.80, 0.67, 0.30))[order])

        tip, finger = tips2d[i], fingers2d[i]
        tether_line.set_data([tip[0], finger[0]], [tip[1], finger[1]])
        finger_dot.set_data([finger[0]], [finger[1]])
        trail_line.set_data(tips2d[: i + 1, 0], tips2d[: i + 1, 1])
        ideal_line.set_data([tips2d[0, 0], fingers2d[-1, 0]],
                            [tips2d[0, 1], fingers2d[-1, 1]])

        f_line.set_data(t[: i + 1], gf[: i + 1])
        d_line.set_data(t[: i + 1], dev_mm[: i + 1])
        for motor_id, line in m_lines.items():
            line.set_data(t[: i + 1], ticks[motor_id][: i + 1] - ticks[motor_id][0])

        colour = {"hold": "#1f77b4", "yield": "#2ca02c", "release": "#d62728"}
        readout.set_text(f"{gf[i]:5.0f} gf   {frame.action.upper():<8s}")
        readout.set_color(colour.get(frame.action, "0.2"))
        status.set_text(
            f"t {frame.t:4.1f}s   given way {travel_mm[i]:5.1f} mm   "
            f"off the line {dev_mm[i]:4.2f} mm"
            + (f"\n{frame.reason}" if frame.action == "release" else "")
        )
        return ()

    out.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(str(out), writer=FFMpegWriter(fps=fps, bitrate=3200))
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cap-gf", type=float, default=200.0,
                   help="the force the controller holds at; it stays still below this")
    p.add_argument("--ceiling-gf", type=float, default=300.0,
                   help="hard ceiling: the gate latches RELEASE above this")
    p.add_argument("--pull-mm-s", type=float, default=5.0,
                   help="how fast the finger pulls away")
    p.add_argument("--damping", type=float, default=15.0, help="B in N.s/m")
    p.add_argument("--v-max-mm-s", type=float, default=20.0)
    p.add_argument("--deadband-n", type=float, default=0.05)
    p.add_argument("--max-step-mm", type=float, default=1.0)
    p.add_argument("--max-travel-mm", type=float, default=60.0)
    p.add_argument("--stiffness", type=float, default=SPRING_N_PER_M,
                   help="tether N/m; defaults to the bench-measured value")
    p.add_argument("--seconds", type=float, default=16.0)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--pose", default=None, help="1:2045,2:1468,... instead of the bench pose")
    p.add_argument("--elev", type=float, default=18.0, help="camera elevation, degrees")
    p.add_argument("--azim", type=float, default=-62.0, help="camera azimuth, degrees")
    p.add_argument("--mesh-mm", type=float, default=3.0,
                   help="mesh simplification grid; 0 draws the full 240k triangles, slowly")
    p.add_argument("--out", default="recordings/coordinated_session.mp4")
    args = p.parse_args(argv)

    pose = DEFAULT_POSE if not args.pose else {
        int(k): float(v) for k, _, v in (part.partition(":") for part in args.pose.split(","))
    }
    limits = SafetyLimits(
        f_threshold_n=grams_force_to_newtons(args.cap_gf),
        f_max_n=grams_force_to_newtons(args.ceiling_gf),
        max_travel_m=args.max_travel_mm / 1000.0,
    )
    params = AdmittanceParams(
        damping_n_s_per_m=args.damping,
        v_max_m_s=args.v_max_mm_s / 1000.0,
        deadband_n=args.deadband_n,
        max_step_m=args.max_step_mm / 1000.0,
    )

    sim = simulate(pose, limits, params, args.pull_mm_s / 1000.0, args.stiffness,
                   args.fps, args.seconds, args.max_travel_mm)
    peak = max(newtons_to_grams_force(f.force_n) for f in sim.frames)
    print(f"joints {sim.motors}, {len(sim.frames)} cycles, peak {peak:.0f} gf, "
          f"gave way {sim.frames[-1].travel_m*1000:.1f} mm, "
          f"worst {max(f.deviation_m for f in sim.frames)*1000:.2f} mm off the line")
    if sim.released:
        print(f"released: {sim.released}")

    out = Path(args.out)
    render(sim, out, args.cap_gf, args.ceiling_gf, args.fps,
           elev=args.elev, azim=args.azim, cell_m=args.mesh_mm / 1000.0)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
