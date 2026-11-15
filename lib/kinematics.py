"""Forward kinematics for the ElRobot, from NormaCore's own URDF.

    from lib.kinematics import load_chain
    chain = load_chain()                       # norma-core/hardware/elrobot/simulation/
    tip = chain.tip_position({1: 2048, ...})   # metres, in the base frame
    for motor, direction, mm in chain.tip_motion_per_100_ticks(ticks):
        ...                                    # which way each motor moves the gripper

Why this exists: nothing in the repo could say which motor moves the gripper along a given
direction, and `docs/elrobot.md` warns that the id-to-joint mapping is unverified and must not be
assumed. Identifying it by jogging one motor at a time and watching the arm works but is slow and
hard to judge by eye, especially for telling "moves along the pull axis" from "moves sideways".
The URDF answers it exactly, and it ships with the hardware.

Two conventions this relies on, both checked against the arm rather than assumed
(`tests/lib/test_kinematics.py` pins them, and `scripts/which_motor.py --check` re-runs the
comparison against whatever calibration is in `config/calibration/`):

  * `rev_motor_0N` in the URDF is Station's motor N. Confirmed by comparing the URDF's joint
    limits against the hand-measured firmware ranges in `config/calibration/`: motors 2 to 7
    agree within 2 degrees. Motor 1 is 8 degrees narrower (a conservative manual calibration)
    and motor 8 is the gripper, whose range is set by where the jaws close rather than by the
    mechanism, so it is excluded.
  * Tick 2048 is the URDF's zero angle, and 4096 ticks is a full turn. Every calibrated range in
    `config/calibration/` is centred on 2048, which is what makes this more than a guess.

Pure: no Station, no hardware, no simulator. Takes a tick reading, returns geometry.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: A full turn of the ST3215's position register.
TICKS_PER_TURN = 4096
#: The tick value the URDF treats as angle zero. Every calibrated range is centred here.
CENTRE_TICKS = 2048

DEFAULT_URDF = "norma-core/hardware/elrobot/simulation/elrobot_follower.urdf"

#: The link the tether is attached to. The gripper's own jaws move relative to this, but the
#: tether hangs off the gripper as a whole, so this is the point whose motion matters.
TIP_LINK = "Gripper_Base_v1_1"


def ticks_to_radians(ticks: float) -> float:
    """Station's position register to the URDF's joint angle."""
    return (ticks - CENTRE_TICKS) / TICKS_PER_TURN * 2 * np.pi


def radians_to_ticks(rad: float) -> float:
    return rad / (2 * np.pi) * TICKS_PER_TURN + CENTRE_TICKS


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF's fixed-axis roll-pitch-yaw to a rotation matrix."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation of `angle` about `axis` (Rodrigues)."""
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


@dataclass(frozen=True)
class Joint:
    name: str
    type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray | None
    #: Station's motor id, parsed from `rev_motor_0N`. None for fixed joints and the jaws.
    motor_id: int | None
    #: `(lower, upper)` in radians, from the URDF, for revolute joints.
    limits: tuple[float, float] | None = None


class Chain:
    """The arm's kinematic chain, as parsed from the URDF."""

    def __init__(self, joints: list[Joint]):
        self.joints = joints
        self.by_parent: dict[str, list[Joint]] = {}
        for j in joints:
            self.by_parent.setdefault(j.parent, []).append(j)

    @property
    def motor_ids(self) -> list[int]:
        """Station motor ids that drive a revolute joint, in order from the base.

        The gripper's two prismatic jaws (`rev_motor_08_1`, `_2`) are not here: `load_chain`
        only assigns a motor id to the eight `rev_motor_0N` joints, so the jaws carry None.
        """
        return [j.motor_id for j in self.joints
                if j.motor_id is not None and j.type == "revolute"]

    def link_transform(self, link: str, ticks: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
        """`(rotation, position)` of `link` in the base frame, for a set of motor tick readings.

        Motors missing from `ticks` are treated as being at the centre of their range, which is
        the URDF's zero. That is deliberate: a partial reading still gives useful geometry, and
        the alternative, refusing, would make this unusable while a motor is quiet on the bus.
        """
        path = self._path_to(link)
        R = np.eye(3)
        p = np.zeros(3)
        for j in path:
            R_j = _rpy_matrix(j.rpy)
            p = p + R @ j.xyz
            R = R @ R_j
            if j.type == "revolute" and j.axis is not None:
                angle = ticks_to_radians(ticks.get(j.motor_id, CENTRE_TICKS)) if j.motor_id else 0.0
                R = R @ _axis_matrix(j.axis, angle)
        return R, p

    def tip_position(self, ticks: dict[int, float], link: str = TIP_LINK) -> np.ndarray:
        """Where the tether's attachment point sits, in metres, in the base frame."""
        return self.link_transform(link, ticks)[1]

    def joint_origin(self, motor_id: int, ticks: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
        """`(axis_in_world, position_in_world)` of one motor's rotation axis."""
        joint = next(j for j in self.joints
                     if j.motor_id == motor_id and j.type == "revolute"
                     and j.name == f"rev_motor_{motor_id:02d}")
        path = self._path_to(joint.parent)
        R = np.eye(3)
        p = np.zeros(3)
        for j in path:
            p = p + R @ j.xyz
            R = R @ _rpy_matrix(j.rpy)
            if j.type == "revolute" and j.axis is not None:
                angle = ticks_to_radians(ticks.get(j.motor_id, CENTRE_TICKS)) if j.motor_id else 0.0
                R = R @ _axis_matrix(j.axis, angle)
        p = p + R @ joint.xyz
        R = R @ _rpy_matrix(joint.rpy)
        return R @ joint.axis, p

    def tip_velocity_direction(self, motor_id: int, ticks: dict[int, float],
                               link: str = TIP_LINK) -> np.ndarray:
        """Unit vector the tip moves along when `motor_id` turns positive, in the base frame.

        A revolute joint moves a point at `v = omega x (p_tip - p_joint)`, so this is exact
        rather than a finite difference, and it is what distinguishes "moves the gripper along
        the pull axis" from "swings it sideways".
        """
        axis, origin = self.joint_origin(motor_id, ticks)
        tip = self.tip_position(ticks, link)
        v = np.cross(axis, tip - origin)
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    def tip_speed_per_tick(self, motor_id: int, ticks: dict[int, float],
                           link: str = TIP_LINK) -> float:
        """Metres the tip moves per tick of `motor_id`, at this pose. The lever arm, in effect.

        This is what `ticks_per_metre` in `session.calibration` measures on the bench; comparing
        the two is a real cross-check of both.
        """
        axis, origin = self.joint_origin(motor_id, ticks)
        tip = self.tip_position(ticks, link)
        v = np.cross(axis, tip - origin)          # metres per radian
        return float(np.linalg.norm(v)) * (2 * np.pi / TICKS_PER_TURN)

    # ------------------------------------------------------------------ many joints at once
    def jacobian(self, motors: list[int], ticks: dict[int, float],
                 link: str = TIP_LINK) -> np.ndarray:
        """3 x N matrix: metres the tip moves per radian of each motor, at this pose.

        Column `i` is `omega_i x (p_tip - p_i)`, the same quantity `tip_velocity_direction`
        returns for one motor, assembled so several joints can be solved for together.
        """
        tip = self.tip_position(ticks, link)
        cols = []
        for motor_id in motors:
            axis, origin = self.joint_origin(motor_id, ticks)
            cols.append(np.cross(axis, tip - origin))
        return np.array(cols).T

    def ik_step(self, motors: list[int], ticks: dict[int, float], target_delta_m: np.ndarray,
                link: str = TIP_LINK, damping: float = 1e-3) -> dict[int, float]:
        """Tick changes that move the tip by `target_delta_m`, sharing the work across `motors`.

        Damped least squares (Levenberg-Marquardt): `dq = J^T (J J^T + lambda^2 I)^-1 dx`. The
        damping matters near a singularity, where a plain pseudo-inverse asks for enormous joint
        velocities to produce a small tip motion. Next to a healing finger that is not a
        numerical curiosity, so `damping` is deliberately not zero.

        One step is a linearisation and so is only accurate for small `target_delta_m`;
        `line_trajectory` is what chains steps into a straight path.
        """
        J = self.jacobian(motors, ticks, link)
        JJt = J @ J.T + (damping ** 2) * np.eye(3)
        dq_rad = J.T @ np.linalg.solve(JJt, target_delta_m)
        return {m: float(dq_rad[i] / (2 * np.pi) * TICKS_PER_TURN)
                for i, m in enumerate(motors)}

    def line_trajectory(self, motors: list[int], start_ticks: dict[int, float],
                        direction: np.ndarray, distance_m: float, steps: int = 60,
                        link: str = TIP_LINK, damping: float = 1e-3) -> list[dict[int, float]]:
        """Tick commands that walk the tip along a straight line, one waypoint per step.

        Why this exists: a single revolute joint moves the tip along an **arc**. Over the few
        millimetres a first bench test uses that is indistinguishable from a line, but one
        full-range repetition needs 72 to 88 mm of travel (`config/calibration/bench.md`), and
        over that distance a single joint wanders tens of millimetres off the tether's axis.
        Coordinating several joints is what keeps the pull straight.

        Re-linearises at every waypoint rather than solving once, so the error does not
        accumulate: each step asks "from where I am now, which way do I go".
        """
        if steps < 1:
            raise ValueError(f"steps must be at least 1, got {steps}")
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        per_step = d * (distance_m / steps)

        current = dict(start_ticks)
        out = []
        for _ in range(steps):
            dq = self.ik_step(motors, current, per_step, link, damping)
            current = {m: current.get(m, CENTRE_TICKS) + dq.get(m, 0.0) for m in current}
            for m, delta in dq.items():
                if m not in start_ticks:
                    current[m] = CENTRE_TICKS + delta
            out.append(dict(current))
        return out

    def line_error(self, motors: list[int], start_ticks: dict[int, float],
                   direction: np.ndarray, distance_m: float, steps: int = 60,
                   link: str = TIP_LINK) -> tuple[float, float]:
        """`(worst deviation from the line, shortfall along it)`, both in metres.

        What `line_trajectory` is for is straightness, so this measures whether it delivered,
        and it is what `tests/lib/test_kinematics.py` asserts on rather than trusting the solver.
        """
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        p0 = self.tip_position(start_ticks, link)
        worst = 0.0
        last_along = 0.0
        for way in self.line_trajectory(motors, start_ticks, d, distance_m, steps, link):
            disp = self.tip_position(way, link) - p0
            along = float(np.dot(disp, d))
            perp = float(np.linalg.norm(disp - along * d))
            worst = max(worst, perp)
            last_along = along
        return worst, distance_m - last_along

    def _path_to(self, link: str) -> list[Joint]:
        """Joints from `base_link` down to `link`, in order."""
        parent_of = {j.child: j for j in self.joints}
        path: list[Joint] = []
        cur = link
        while cur in parent_of:
            j = parent_of[cur]
            path.append(j)
            cur = j.parent
        return list(reversed(path))


def load_chain(urdf_path: str | Path = DEFAULT_URDF) -> Chain:
    """Parse the ElRobot URDF. Raises with the setup hint if the clone is not there."""
    p = Path(urdf_path)
    if not p.is_file():
        raise SystemExit(
            f"URDF not found at {p}.\n\n"
            "It ships with the hardware, in NormaCore's repo, and the sparse clone in SETUP.md "
            "already includes hardware/elrobot/simulation. If the clone predates that, add it:\n\n"
            "    cd norma-core && git sparse-checkout add hardware/elrobot/simulation"
        )
    root = ET.parse(p).getroot()
    joints = []
    for j in root.findall("joint"):
        name = j.get("name") or ""
        origin = j.find("origin")
        xyz = np.array([float(v) for v in (origin.get("xyz") if origin is not None else "0 0 0").split()])
        rpy = np.array([float(v) for v in (origin.get("rpy") if origin is not None else "0 0 0").split()])
        axis_el = j.find("axis")
        axis = np.array([float(v) for v in axis_el.get("xyz").split()]) if axis_el is not None else None
        motor_id = None
        if name.startswith("rev_motor_") and len(name) == len("rev_motor_01"):
            motor_id = int(name.rsplit("_", 1)[1])
        lim_el = j.find("limit")
        limits = ((float(lim_el.get("lower")), float(lim_el.get("upper")))
                  if lim_el is not None and lim_el.get("lower") is not None else None)
        joints.append(Joint(name, j.get("type") or "fixed",
                            j.find("parent").get("link"), j.find("child").get("link"),
                            xyz, rpy, axis, motor_id, limits))
    return Chain(joints)


def check_against_calibration(chain: Chain, ranges: dict[str, list[int]],
                              tolerance_deg: float = 12.0) -> list[tuple[int, float, float, bool]]:
    """Compare each motor's calibrated tick range against the URDF's limit for that joint.

    Returns `(motor_id, measured_span_deg, urdf_span_deg, agrees)` per motor. This is the check
    that turns "rev_motor_0N is probably motor N" into something verified against this physical
    arm, and it is worth re-running after any recalibration.

    Motor 8 is reported like the rest but will usually disagree: the gripper's calibrated range
    is set by where its jaws meet, not by the mechanism's travel.
    """
    limits = {j.motor_id: j.limits for j in chain.joints
              if j.motor_id is not None and j.type == "revolute" and j.limits is not None}
    out = []
    for motor_id in sorted(int(k) for k in ranges):
        if motor_id not in limits:
            continue
        lo, hi = ranges[str(motor_id)]
        measured = (hi - lo) / TICKS_PER_TURN * 360
        lo_u, hi_u = limits[motor_id]
        urdf_span = float(np.degrees(hi_u - lo_u))
        out.append((motor_id, measured, urdf_span, abs(measured - urdf_span) < tolerance_deg))
    return out
