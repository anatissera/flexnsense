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

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The seven joints that move the gripper. Motor 8 is the gripper's own jaws: it streams on
#: the same Station bus but carries no arm geometry, so callers that need a complete pose for
#: forward kinematics (not the gripper's open/close state) wait for these seven, not eight.
ARM_MOTORS = (1, 2, 3, 4, 5, 6, 7)

#: A full turn of the ST3215's position register.
TICKS_PER_TURN = 4096
#: The tick value the URDF treats as angle zero. Every calibrated range is centred here.
CENTRE_TICKS = 2048

DEFAULT_URDF = "norma-core/hardware/elrobot/simulation/elrobot_follower.urdf"

#: The link the tether is attached to. The gripper's own jaws move relative to this, but the
#: tether hangs off the gripper as a whole, so this is the point whose motion matters.
TIP_LINK = "Gripper_Base_v1_1"

#: Column of the tip link's rotation matrix that points out of the jaws, along the tether.
#: Which of the three local axes this is comes from the URDF's own frame convention, so it is
#: checked rather than assumed: `Chain.gripper_axis` holds 0.94 against the independently
#: computed forearm direction (motor 6's joint to the tip) in every pose tried, including
#: folded and fully extended ones, and the other two columns do not.
GRIPPER_FORWARD_COLUMN = 1


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

    # ------------------------------------------------------------------ the tether's own line
    def gripper_axis(self, ticks: dict[int, float], link: str = TIP_LINK) -> np.ndarray:
        """Unit vector pointing out of the gripper's jaws, in the base frame.

        This is the line the tether runs along: it leaves the gripper and goes to the finger,
        so the finger sits somewhere along **plus** this direction. It rotates with the arm,
        which is the point. Asking a person to read a direction off Station's 3D view and type
        x,y,z is where this went wrong in practice: the view's axes are not the ones a person
        standing at the bench is looking at, and a wrong direction only shows up as a pull that
        wanders off the tether's line.

        Taken from the tip link's own orientation rather than from "where the arm happens to be
        pointing", so it stays correct when the wrist is turned.
        """
        R = self.link_transform(link, ticks)[0]
        v = R[:, GRIPPER_FORWARD_COLUMN]
        return v / np.linalg.norm(v)

    def ticks_per_metre_along(self, motor_id: int, ticks: dict[int, float],
                              direction: np.ndarray, link: str = TIP_LINK) -> float:
        """Ticks of `motor_id` per metre of tip travel **along `direction`**, at this pose.

        Different from `1 / tip_speed_per_tick`, and the difference is the whole reason this
        exists: a joint that moves the tip 26 mm per 100 ticks but mostly sideways to the
        tether contributes far less than 26 mm of the travel the controller is counting.
        Dividing by the total speed would make the loop believe it had given way further than
        it had, which is the wrong direction for a travel limit to be wrong in.

        Raises if this joint barely moves the tip along `direction` at all, because the
        reciprocal of nearly zero is a number that would sail through every later check.
        """
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        along = self.tip_speed_per_tick(motor_id, ticks, link) * abs(
            float(np.dot(self.tip_velocity_direction(motor_id, ticks, link), d))
        )
        if along < 1e-7:                       # under 0.1 mm per 1000 ticks
            raise ValueError(
                f"motor {motor_id} moves the tip {along*1e6:.2f} micrometres per tick along "
                "this direction, which is not enough to measure travel against. Track a joint "
                "that actually moves along the tether"
            )
        return 1.0 / along

    def yield_sign(self, motor_id: int, ticks: dict[int, float], direction: np.ndarray,
                   link: str = TIP_LINK) -> int:
        """Which tick direction on `motor_id` *reduces* tension, given the pull `direction`.

        `direction` points the way that increases tension. Positive ticks move the tip along
        `tip_velocity_direction`, so the sign that gives way is whichever opposes that.

        Geometry can answer this because it is a direction, not a magnitude: it needs the URDF
        and the pose to be right, but none of the mechanical slop that makes the *distance*
        worth measuring on the bench. `ControlLoop._check_yield_direction` still checks the
        result against the measured force within half a second of the first yield, so a wrong
        answer here costs a bounded step, not a session.
        """
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        along = float(np.dot(self.tip_velocity_direction(motor_id, ticks, link), d))
        if abs(along) < 1e-6:
            raise ValueError(
                f"motor {motor_id} moves the tip perpendicular to this pull, so there is no "
                "tick direction that releases it. Track a joint that moves along the tether"
            )
        return -1 if along > 0 else 1

    def best_tracked_motor(self, motors: list[int], ticks: dict[int, float],
                           direction: np.ndarray, link: str = TIP_LINK) -> int:
        """Of `motors`, the one whose register best measures travel along `direction`.

        A coordinated pull moves three joints, but the travel limit and the yield-direction
        check both read one register (see `control.loop.ControlLoop`). The best one to read is
        the joint that converts the most tip motion along the tether per tick: it has the
        finest resolution on the quantity being limited, and the least of its motion is
        sideways travel that the bookkeeping would otherwise miscount.
        """
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        best, best_speed = None, 0.0
        for motor_id in motors:
            speed = self.tip_speed_per_tick(motor_id, ticks, link) * abs(
                float(np.dot(self.tip_velocity_direction(motor_id, ticks, link), d))
            )
            if speed > best_speed:
                best, best_speed = motor_id, speed
        if best is None:
            raise ValueError(f"none of {motors} moves the tip along this direction at all")
        return best

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

    def choose_pull_motors(self, ticks: dict[int, float], direction: np.ndarray | None = None,
                           candidates: list[int] | None = None, want: int = 3,
                           link: str = TIP_LINK) -> list[int]:
        """Pick the joints best able to pull a straight line from *this* pose.

        Which joints to use is a property of the pose, not of the arm: the same three that
        track a line beautifully in one pose can be nearly coplanar in another. Rather than
        fixing a list once and hoping the pose never changes, this re-derives it.

        Chosen to maximise the **smallest** singular value of the resulting jacobian, by
        exhaustive search over the candidate sets (there are only 35 ways to pick 3 of 7, so
        there is no reason to be clever and risk a greedy mistake). That is deliberately not
        "whichever joints pull hardest along `direction`": the smallest singular value is the
        gain in the arm's *worst* direction, and the worst direction is exactly what the return
        stroke of a repetition runs along. An earlier greedy version scored by alignment with
        the pull and chose sets that tracked the outward stroke to 0.4 mm and the return to
        14.8 mm, which is the failure this now rules out by construction.

        `direction` is therefore unused for the choice and kept only so callers can pass it;
        it is used by `pull_quality` for reporting. Motor 8 is never a candidate, being the
        gripper's own jaws.
        """
        from itertools import combinations

        if candidates is None:
            candidates = [m for m in self.motor_ids if m != 8]
        want = min(want, len(candidates))

        best, best_score = None, -1.0
        for combo in combinations(sorted(candidates), want):
            J = self.jacobian(list(combo), ticks, link)
            sv = np.linalg.svd(J, compute_uv=False)
            score = float(sv[-1])          # gain in the worst direction
            if score > best_score:
                best, best_score = list(combo), score
        return best if best is not None else []

    def pull_quality(self, motors: list[int], ticks: dict[int, float], direction: np.ndarray,
                     link: str = TIP_LINK) -> tuple[int, float]:
        """`(rank, condition number)` of these joints at this pose.

        Rank below 3 means the set spans only a plane, so some direction, typically the return
        stroke, cannot be followed. A very large condition number means it can be followed only
        by asking for large joint motions, which the damping in `ik_step` will refuse to do.
        """
        J = self.jacobian(motors, ticks, link)
        rank = int(np.linalg.matrix_rank(J, tol=1e-6))
        sv = np.linalg.svd(J, compute_uv=False)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 1e-15 else float("inf")
        return rank, cond

    def direction_gain(self, motors: list[int], ticks: dict[int, float],
                       direction: np.ndarray, link: str = TIP_LINK) -> float:
        """How well these joints can move the tip along `direction`, as metres per radian.

        The quantity that decides whether a pull is followable at all. Not every direction is
        reachable from every pose: at some configurations the arm can move the tip hard along
        one axis and barely at all along another, and no amount of damping, step refinement or
        extra joints fixes it, because the limit is the mechanism's, not the solver's.

        A low gain is what a silently bad pull looks like before it happens, which is why
        `CoordinatedAxis` checks it rather than discovering it on a finger.
        """
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        J = self.jacobian(motors, ticks, link)
        # Least-norm joint velocity that produces unit tip speed along d, inverted: the gain is
        # how much tip motion a unit of joint motion buys in that direction.
        JJt = J @ J.T + 1e-12 * np.eye(3)
        dq = J.T @ np.linalg.solve(JJt, d)
        n = float(np.linalg.norm(dq))
        return 1.0 / n if n > 1e-12 else 0.0

    def can_pull_straight(self, motors: list[int], ticks: dict[int, float],
                          direction: np.ndarray, distance_m: float = 0.090,
                          min_gain: float = 0.03, max_deviation_m: float = 0.002,
                          steps: int = 40, link: str = TIP_LINK,
                          tick_ranges: dict[int, tuple[float, float]] | None = None
                          ) -> tuple[bool, str]:
        """Whether a straight pull of `distance_m` is actually followable from this pose.

        Returns `(ok, reason)`, and checks the **whole planned path**, not just the starting
        pose. That distinction is the point of this method: a pull can start from a
        well-conditioned pose and drive itself into a singularity on the way. Measured on this
        arm, a pose with a healthy 82 mm per radian at the start degraded to 15 mm per radian
        by the end of a 90 mm pull, with the condition number rising from 3.9 to 25.5 and the
        path wandering 8.6 mm off the line. Checking only the start called that fine.

        Four ways a pull fails, all of which look correct for the first few millimetres:

          * fewer than three joints, so the reachable set is a plane and the return stroke misses;
          * rank dropping below three somewhere along the path;
          * the gain along `direction` falling too low to follow without large joint motion;
          * a joint running past its real travel partway along the path (see `tick_ranges`).

        `min_gain` is metres of tip motion per radian of joint motion: the default 0.03 means
        a joint turning one radian, about 650 ticks, must buy at least 30 mm along the pull.
        `max_deviation_m` is the straightness the finished path has to achieve.

        `tick_ranges`, `{motor_id: (lo, hi)}`, is deliberately not looked up from the URDF
        automatically: nothing else in this class does, so a pull can be planned, pass every
        other check, and still ask a joint to travel past a limit this class has no way to
        know about on its own. The URDF's own `j.limits` is one legitimate source (pass
        `{j.motor_id: j.limits_in_ticks for j in chain.joints if ...}`), but the hand-measured
        calibration in `config/calibration/follower-*.json` is the one that matches the
        physical rig, cabling and the mechanism's actual stops, which can be narrower than the
        URDF believes. Without this, two joints quietly compensating for a third that has
        pinned against its stop is exactly how a coordinated pull drifts off its line while
        reporting perfect geometry the whole time: every number above is computed from the
        *commanded* ticks, which is not the same thing as where the servo actually is once one
        of them cannot go any further.
        """
        if len(motors) < 3:
            return False, (f"{len(motors)} joints span at most a plane; three independent ones "
                           "are needed to follow an arbitrary direction")

        worst_gain = float("inf")
        worst_cond = 0.0
        pose = dict(ticks)
        for pose in [ticks] + self.line_trajectory(motors, ticks, direction, distance_m,
                                                   steps, link):
            if tick_ranges is not None:
                for motor_id in motors:
                    lo, hi = tick_ranges[motor_id]
                    value = pose[motor_id]
                    if value < lo or value > hi:
                        over = lo - value if value < lo else value - hi
                        return False, (
                            f"motor {motor_id} would be commanded to {value:.0f} ticks, "
                            f"{over:.0f} past its [{lo:.0f}, {hi:.0f}] range, partway along "
                            "this pull: it would stop at its hard limit while the other two "
                            "joints kept moving, which bends the path off the line instead of "
                            "stopping it outright"
                        )
            rank, cond = self.pull_quality(motors, pose, direction, link)
            if rank < 3:
                return False, (f"joints {motors} drop to rank {rank} partway along this pull, "
                               "so the path cannot be followed to the end")
            worst_gain = min(worst_gain, self.direction_gain(motors, pose, direction, link))
            worst_cond = max(worst_cond, cond)

        if worst_gain < min_gain:
            return False, (f"the gain along this direction falls to {worst_gain*1000:.0f} mm "
                           f"per radian during the pull, under the {min_gain*1000:.0f} mm "
                           "minimum: this pose drives itself towards a singularity. Re-pose "
                           "the arm, shorten the pull, or pull along another axis")

        deviation, shortfall = self.line_error(motors, ticks, direction, distance_m, steps, link)
        if deviation > max_deviation_m:
            return False, (f"the path wanders {deviation*1000:.1f} mm off the line, over the "
                           f"{max_deviation_m*1000:.0f} mm limit")
        return True, (f"worst gain {worst_gain*1000:.0f} mm per radian, worst condition "
                      f"{worst_cond:.0f}, deviation {deviation*1000:.2f} mm over "
                      f"{distance_m*1000:.0f} mm")

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


def level_direction(direction: np.ndarray) -> np.ndarray:
    """`direction` with its z component dropped and the rest renormalised.

    The gripper's own axis tilts a little out of horizontal (on this bench, 0.1 to 0.12 of
    the unit vector), and over a long pull that is several millimetres of real rise or fall:
    visible on the bench as the gripper climbing or dipping instead of tracking the straight
    line the tether actually runs along. Levelling trades following the gripper's exact tilt
    for matching a tether that is close to horizontal, which is what the rig's photos show.

    Shared rather than owned by one script because `scripts/jog_coordinated.py` and
    `scripts/run_session.py` have to agree on it exactly: a pull that tracks straight when
    jogged and then climbs under the controller is the same rig behaving two ways, and the
    only honest way to rule that out is for both to compute the direction with one function.

    Raises if the direction is already (near) purely vertical: there would be nothing left in
    the horizontal plane to renormalise, and silently returning a zero vector would pass a
    degenerate pull on to `Chain.choose_pull_motors`.
    """
    d = np.asarray(direction, dtype=float)
    horizontal = np.array([d[0], d[1], 0.0])
    norm = float(np.linalg.norm(horizontal))
    if norm < 1e-6:
        raise ValueError(
            "this direction is (close to) straight up or down, so there is nothing left in "
            "the horizontal plane once z is dropped. Levelling does not apply to a vertical pull"
        )
    return horizontal / norm


def tether_tension_direction(chain: Chain, ticks: dict[int, float],
                             link: str = TIP_LINK) -> np.ndarray:
    """The direction the gripper has to travel to **increase** tension, at this pose.

    The rig's one physical fact, written down once: the tether leaves the gripper's jaws and
    runs to the finger, so the finger lies along `+gripper_axis`. Moving the gripper that way
    closes the gap and slackens the line; retracting, along `-gripper_axis`, stretches it. That
    is the sense `control.loop.CoordinatedAxis.direction` is defined in.

    Checked against the bench rather than only reasoned about. The camera measurement of
    2026-11-15 found yield_sign -1 for motor 4 at pose m1=2049 m2=1469 m3=2468 m4=2772
    m5=1879 m6=1525 m7=3275, meaning negative ticks lowered the force. Negative ticks move the
    tip along `-tip_velocity_direction(4)`, which dots **+0.49** with `gripper_axis` at that
    pose: the motion that measurably slackened the tether does have a positive component along
    the gripper's forward axis, exactly as this says it should.
    """
    return -chain.gripper_axis(ticks, link)


def load_tick_ranges(bus_serial: str,
                     calib_dir: str | Path = "config/calibration"
                     ) -> dict[int, tuple[float, float]]:
    """The hand-measured per-motor travel for one follower arm, as `{motor_id: (lo, hi)}`.

    Reads `config/calibration/follower-<bus_serial>.json`, the same file
    `check_against_calibration` compares against the URDF, and reshapes it into what
    `Chain.can_pull_straight`'s `tick_ranges` wants.

    This is the rig's *measured* range, from moving each joint through its real travel by
    hand, not the URDF's. They mostly agree (`tests/lib/test_kinematics.py`), but the
    firmware's is the one actually enforced by the servo, so it is the one a coordinated pull
    has to respect to avoid a joint pinning against its stop mid-pull.

    Raises if the file is missing, rather than returning an empty range that `can_pull_straight`
    would silently never trigger on: a coordinated pull with no range check at all is worse
    than refusing to plan one, for the reason this function exists.
    """
    path = Path(calib_dir) / f"follower-{bus_serial}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no calibration at {path}. A coordinated pull needs the real joint ranges to "
            "check against; see docs/bench-measurements.md for how this file is produced, "
            "or pass tick_ranges explicitly if you have them some other way."
        )
    ranges = json.loads(path.read_text())["ranges"]
    return {int(motor_id): (float(lo), float(hi)) for motor_id, (lo, hi) in ranges.items()}


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
