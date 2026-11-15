"""The simulation-trained pull policy as a runner for robot_training/inference/control.py.

It has the same surface as the runners in robot_training/inference/policy_runner.py: `predict_from_jpegs(state,
jpeg_by_view, task, prev)` takes the eight `position_norm` values of a Station frame and returns goals as
`goal_norm`, so the existing control loop, its Station code and its tick-level safety layer are used unchanged.
Station's camera pictures are ignored; the force comes from gauge_reader reading the scale's own camera (CameraForce).

Units. The policy was trained in radians from the URDF's zero pose. A Station frame gives each joint as a fraction of
its calibrated range. The follower's calibrated ranges match the URDF joint limits (motor 6: 2048 ticks = 3.1416 rad,
the URDF's span exactly; the others within 1 %, motor 1 within 5 %), so
    angle = lower + position_norm * (upper - lower)
for a motor that turns the way the URDF axis does, and mirrored for one that turns the other way. Which way each
motor turns on the real arm is NOT verified: see `directions` and the README.

What the policy gets (29 numbers, the order used in training, sim/rl/flex_tasks/pull_env.py):
   0..6   angle of motors 1..7 minus the training start posture [rad]
   7..13  change of angle since the last frame / time between frames [rad/s], times 0.1
  14..18  target minus angle for motors 1, 2, 3, 4, 6 [rad]
  19      force at the last frame / 1.96 N
  20      (force now - force at the last frame) / 1.96 N
  21..23  the point between the jaws minus where it was at the start: along the scale, across it, up [m], times 10
  24..28  the previous action
What it returns (5 numbers in -1..1): change of the targets of motors 1, 2, 3, 4, 6, times `action_scale` rad.

NEVER RUN ON HARDWARE YET. Start with --dry-run (see README).
"""

from __future__ import annotations

import json
import math
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_URDF = HERE.parent / "assets" / "elrobot" / "elrobot_follower.urdf"
ARM_JOINTS = [f"rev_motor_0{i}" for i in range(1, 8)]
DRIVEN = [0, 1, 2, 3, 5]  # motors 1, 2, 3, 4, 6 (index in the list of seven arm motors)


class ForceLost(RuntimeError):
    """No fresh reading of the scale. The loop must stop: a missing force is never replaced by a guess."""


# ─────────────────────────────── the arm's geometry, from NormaCore's URDF ─────────────────────────────

def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    return _rot(np.array([0, 0, 1.0]), y) @ _rot(np.array([0, 1.0, 0]), p) @ _rot(np.array([1.0, 0, 0]), r)


class Arm:
    """Joint limits and forward kinematics of the seven arm joints, read from the URDF."""

    def __init__(self, urdf: str | Path = DEFAULT_URDF):
        root = ET.parse(str(urdf)).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, link = [], "Gripper_Base_v1_1"
        while link in by_child:
            chain.append(by_child[link])
            link = by_child[link].find("parent").get("link")
        self.chain = list(reversed(chain))
        named = {j.get("name"): j for j in root.findall("joint")}
        self.lower = np.array([float(named[n].find("limit").get("lower")) for n in ARM_JOINTS])
        self.upper = np.array([float(named[n].find("limit").get("upper")) for n in ARM_JOINTS])

    def gripper(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Position of the gripper's origin and the direction it points (its own +y), in the base frame."""
        rot, pos, k = np.eye(3), np.zeros(3), 0
        for j in self.chain:
            o = j.find("origin")
            xyz = np.array([float(v) for v in o.get("xyz", "0 0 0").split()])
            pos, rot = pos + rot @ xyz, rot @ _rpy(*[float(v) for v in o.get("rpy", "0 0 0").split()])
            if j.get("type") == "revolute":
                rot = rot @ _rot(np.array([float(v) for v in j.find("axis").get("xyz").split()]), float(q[k]))
                k += 1
        return pos, rot[:, 1]


def norm_to_rad(norm: np.ndarray, lower: np.ndarray, upper: np.ndarray, directions: np.ndarray) -> np.ndarray:
    n = np.where(directions > 0, norm, 1.0 - norm)
    return lower + n * (upper - lower)


def rad_to_norm(rad: np.ndarray, lower: np.ndarray, upper: np.ndarray, directions: np.ndarray) -> np.ndarray:
    n = (rad - lower) / (upper - lower)
    return np.clip(np.where(directions > 0, n, 1.0 - n), 0.0, 1.0)


# ─────────────────────────────── the force, from the scale reader ──────────────────────────────────────

class CameraForce:
    """Reads the scale with gauge_reader, in this process, from its camera.

    The calibration is loaded and the camera opened in the caller's thread, so a missing calibration file or a camera
    that will not open stops the program at once with the real message. (gauge_reader reports those with SystemExit,
    which vanishes without a trace when raised inside a thread.) Reading then continues in a background thread; an
    error there is kept and shown by `why()`.
    """

    def __init__(self, camera: int | str, calib: str = "config/scale_calib.json", stale_s: float = 0.3):
        try:
            from gauge_reader.cli import _load_calibration, make_pipeline
            from lib.camera import open_camera

            self.pipe = make_pipeline(_load_calibration(calib))
            self.source = open_camera(int(camera) if str(camera).lstrip("-").isdigit() else camera)
        except SystemExit as e:
            raise RuntimeError(f"scale reading: {e}") from None
        self.stale_s, self.error, self.reason = stale_s, None, None
        self._value: float | None = None
        self._t = 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            while True:
                ok, frame, ts, _ = self.source.read()
                if not ok:
                    self.error = "the scale camera stopped giving pictures"
                    break
                result = self.pipe.process(frame, ts)
                found = bool(result.found) and result.newtons is not None
                self._value, self.reason = (float(result.newtons) if found else None), getattr(result, "reason", None)
                self._t = time.monotonic()
        except BaseException as e:  # including SystemExit: keep the cause instead of losing it with the thread
            self.error = f"{type(e).__name__}: {e}"
        self._value = None

    def __call__(self) -> float | None:
        return self._value if self.error is None and time.monotonic() - self._t < self.stale_s else None

    def why(self) -> str:
        """Why there is no reading right now, in words."""
        if self.error:
            return self.error
        if time.monotonic() - self._t >= self.stale_s:
            return f"no picture from the scale camera in the last {self.stale_s} s"
        return f"the scale was not found in the picture ({self.reason or 'no reason given'})"


def _load_policy(path: Path):
    """The exported network as a function (1, 29) -> (1, 5). TorchScript if torch is installed (it is in the
    robot_training environment), otherwise ONNX with onnxruntime. Both files hold the same network, including the
    scaling of its inputs."""
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and (path / "policy.pt").exists():
        torch.set_num_threads(2)
        model = torch.jit.load(str(path / "policy.pt"), map_location="cpu").eval()

        def infer(obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return model(torch.from_numpy(obs)).numpy()

        return infer
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path / "policy.onnx"), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return lambda obs: sess.run(None, {name: obs})[0]


# ─────────────────────────────── the runner ────────────────────────────────────────────────────────────

class SimPullRunner:
    chunk_size = 1
    n_action_steps = 1  # one goal per Station frame: the policy decides again at every frame
    n_obs_steps = 1
    state_dim = 8
    device = "cpu"

    def __init__(self, policy_dir: str | Path, force, urdf: str | Path = DEFAULT_URDF, session=None,
                 directions: list[int] | None = None, max_lead_rad: float = 0.6):
        """policy_dir: folder with policy.pt / policy.onnx and flex_sim.json. force: a callable giving newtons or None.
        directions: +1 or -1 per arm motor 1..7 (default: joint_directions.json in policy_dir, else all +1)."""
        path = Path(policy_dir)
        self.meta = json.loads((path / "flex_sim.json").read_text())
        self.infer = _load_policy(path) if session is None else (
            lambda obs, s=session, n=session.get_inputs()[0].name: s.run(None, {n: obs})[0])
        self.arm, self.force = Arm(urdf), force
        if directions is None:
            f = path / "joint_directions.json"
            directions = json.loads(f.read_text()) if f.exists() else [1] * 7
        self.directions = np.asarray(directions, dtype=np.float64)
        assert self.directions.shape == (7,) and set(np.abs(self.directions)) == {1.0}, "directions: seven values, +1 or -1"
        self.start = np.asarray(self.meta["start_rad"], dtype=np.float64)
        self.cap, self.tip, self.scale = self.meta["cap_n"], self.meta["tip_m"], self.meta["action_scale"]
        self.period, self.max_lead = 1.0 / self.meta["rate_hz"], max_lead_rad
        self.cfg = SimpleNamespace(type="flex_sim_pull")
        self._first = True
        self.last = {}

    def _tip(self, q: np.ndarray) -> np.ndarray:
        pos, pointing = self.arm.gripper(q)
        return pos + pointing * self.tip

    def predict_from_jpegs(self, state, jpeg_by_view=None, task: str = "", prev=None) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        state = np.asarray(state, dtype=np.float64)
        q = norm_to_rad(state[:7], self.arm.lower, self.arm.upper, self.directions)
        force = self.force()
        if force is None:
            why = self.force.why() if hasattr(self.force, "why") else "no fresh reading"
            raise ForceLost(f"no reading of the scale ({why}): stopping. The arm keeps its last goal")
        if self._first:
            self.hold = state.copy()  # motors 5, 7 and the gripper stay where they are now
            self.target, self.q_prev, self.force_prev = q.copy(), q.copy(), float(force)
            self.home = self._tip(q)
            u = self.home.copy()
            u[2] = 0.0
            self.u = u / np.linalg.norm(u)  # along the scale: from the base towards the jaws, level
            self.side = np.cross([0.0, 0.0, 1.0], self.u)
            self.action = np.zeros(5, dtype=np.float32)
            self._first = False
        off = self._tip(q) - self.home
        obs = np.concatenate([
            q - self.start,
            0.1 * (q - self.q_prev) / self.period,
            (self.target - q)[DRIVEN],
            [self.force_prev / self.cap, (force - self.force_prev) / self.cap],
            10.0 * np.array([off @ self.u, off @ self.side, off[2]]),
            self.action,
        ]).astype(np.float32)[None]
        self.action = np.clip(np.asarray(self.infer(obs))[0], -1.0, 1.0).astype(np.float32)
        self.target[DRIVEN] += self.scale * self.action
        self.target = np.clip(self.target, self.arm.lower, self.arm.upper)
        self.target = np.clip(self.target, q - self.max_lead, q + self.max_lead)  # a target never runs far ahead of the arm
        goal = self.hold.copy()
        goal[:7] = rad_to_norm(self.target, self.arm.lower, self.arm.upper, self.directions)
        for k in (4, 6):  # the two roll motors are held, as in training
            goal[k] = self.hold[k]
        self.q_prev, self.force_prev = q, float(force)
        self.last = {"force_n": float(force), "along_m": float(off @ self.u), "action": self.action.copy()}
        return goal[None].astype(np.float32), (time.perf_counter() - t0) * 1000
