"""The bench with its two follower arms, each doing its own job.

  * PULL ARM  (holds the ring at the top of the scale): stays put, and gives way along the scale while the force is
    above the limit. This is the arm the overhead camera watches to read the force.
  * MOVE ARM  (at the side of the scale): comes down on the scale from above, near the loop, lifts it, carries the loop
    to the next finger, sets it down and lets go.

One cycle: pull on the index finger -> the move arm carries the loop to the middle finger -> pull there -> back.

Runs ON the workstation:

    source ~/isaac-env.sh && cd ~/IsaacLab
    uv run --extra isaacsim python ~/flexnsense_sim/07_two_arms.py --tag two_arms
    PUBLIC_IP=<vm-ip> LIVESTREAM=1 uv run --extra isaacsim python ~/flexnsense_sim/07_two_arms.py --cycles 100000 --no_record --viz kit

Everything is SCRIPTED with inverse kinematics; no trained policy runs here. The hand and the scale are drawn only; the
force is a virtual spring between the pull arm's jaw tips and the finger that wears the loop. The overhead camera's view is rendered as the second picture, the move arm's wrist camera as the third.

The scene is laid out from the bench photos in reference/bench-photos/: a finite table (the arms cannot pass through
it) on legs, the cardboard sheet, and the camera post with its boom and overhead camera. Only the two arms in use are
in the scene; the two leader arms are for recording demonstrations and are left out. Positions are estimated from the photos, not measured.

Writes ~/flexnsense-sim/runs/two_arms/<tag>/ : frame_*.png (bench | overhead camera | wrist camera), log.csv, summary.json.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Two ElRobot followers: one pulls with a force limit, one moves the scale.")
parser.add_argument("--cap", type=float, default=1.96, help="Force limit [N] (1.96 N = 200 gf).")
parser.add_argument("--gain", type=float, default=1.0, help="Admittance gain B [m/s per N].")
parser.add_argument("--pull", type=float, default=0.06, help="How far the finger pulls away [m].")
parser.add_argument("--tag", default="two_arms", help="Output folder name under runs/two_arms/.")
parser.add_argument("--cycles", type=int, default=1, help="Cycles to run (index -> middle -> back to index).")
parser.add_argument("--no_record", action="store_true", help="Do not save frames (for the live stream).")
parser.add_argument("--hero", default=None, help="Path to a 2:1 panorama image. Renders ONE clean portrait view (no caption) "
                    "with that image as the room around the bench, for the pitch deck's title slide.")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

import csv
import json
import math
import os

import torch
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import instantiate
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

ROOT = os.path.expanduser("~/flexnsense-sim")
USD = f"{ROOT}/assets/elrobot/usd/elrobot_follower.usd/elrobot_follower/elrobot_follower.usda"
OUT = f"{ROOT}/runs/two_arms/{args_cli.tag}"

FULL_SCALE = 5.886  # N: the PESOLA reads up to 600 gf
TRAVEL = 0.10  # m: ASSUMED marker travel at full scale
K_SPRING = FULL_SCALE / TRAVEL
SCALE_LEN = 0.23  # m: ring (in the pull arm's jaws) to the finger loop, at zero force
# The PESOLA, from the gripper to the finger: swivel and rod (it lengthens under load) - tube - white cap - wire loop.
# The loop is the C-shaped bail at the top of the tube; it goes round the finger. Lengths estimated from the photo.
TUBE_LEN, CAP_LEN, BAIL_LEG, BAIL_R = 0.165, 0.012, 0.032, 0.013  # SCALE_LEN = rod 0.021 + tube + cap + bail leg
FINGER_UP, KNUCKLE_Z = 0.062, 0.018  # m: the raised part of each finger, and the height of the flat part
TIP = 0.10  # m: ASSUMED gripper origin -> point between the jaw tips
REACH, HEIGHT = 0.20, 0.05  # m: the pull arm's gripper, from its base and above the table
GRASP_AT = 0.185  # m along the scale from the ring: where the move arm takes hold
B_BASE = (0.24, 0.40, 0.0)  # m: the move arm's base, beside the top half of the scale (the pull arm's base is the origin)
B_YAW = math.pi / 2  # the move arm faces the scale
SPACING, LIFT, HOVER, FINGER_LEN = 0.024, 0.04, 0.04, 0.075
SETTLE, REACH_T = 1.0, 4.0
PULL_T, HOLD_T, RELAX_T, REST_T = 2.0, 1.5, 1.2, 0.8
DOWN_T, GRIP_T, LIFT_T, SHIFT_T, LOWER_T, OPEN_T, UP_T = 1.3, 0.5, 1.0, 1.2, 1.0, 0.5, 1.2
AIM_WEIGHT = 0.15  # how much one radian of aim error counts against one metre of position error
RETURN_SPEED = 0.02  # m/s: how fast the pull arm returns to its place once the loop is off the finger
FINGERS = ["index", "middle", "ring", "little"]


def arm_cfg(path: str, pos=(0.0, 0.0, 0.0), yaw: float = 0.0) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=path,
        # Clamped to the table. No collisions between the arm's own parts: the converted model's collision shapes are
        # coarse hulls, and the gripper's small gear presses on the forearm's hull, which turned the wrist whenever
        # the gripper motor was driven.
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD, fix_root_link=True, articulation_props=[PhysxArticulationCfg(enabled_self_collisions=False)]
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=pos, rot=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))),
        actuators={
            "servos": ImplicitActuatorCfg(
                joint_names_expr=["rev_motor_0[1-7]"], joint_effort_limit=2.94, joint_velocity_limit=4.71,
                stiffness=40.0, damping=2.0,
            ),
            # The gripper motor and its jaws are NOT driven: in this converted model, driving motor 8 turns the wrist
            # (motor 7 spun continuously). They stay where the start posture puts them, so the move arm's jaws do
            # not visibly close on the scale. To fix properly, the model's gripper gearing has to be rebuilt.
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["rev_motor_08.*"], joint_effort_limit=2.94, joint_velocity_limit=4.71,
                stiffness=0.0, damping=0.5,
            ),
        },
    )


def ease(p: float) -> float:
    p = min(1.0, max(0.0, p))
    return 2 * p * p if p < 0.5 else 1 - (-2 * p + 2) ** 2 / 2


def pull_at(t: float) -> float:
    if t < PULL_T:
        return args_cli.pull * ease(t / PULL_T)
    if t < PULL_T + HOLD_T:
        return args_cli.pull
    if t < PULL_T + HOLD_T + RELAX_T:
        return args_cli.pull * (1 - ease((t - PULL_T - HOLD_T) / RELAX_T))
    return 0.0


def quat_z_to(d: torch.Tensor) -> torch.Tensor:
    """Quaternion (x, y, z, w) that turns the +Z axis onto the unit vector d."""
    z = torch.tensor([0.0, 0.0, 1.0], device=d.device)
    axis = torch.linalg.cross(z, d)
    s = axis.norm()
    if s < 1e-8:
        return torch.tensor([0.0, 0.0, 0.0, 1.0], device=d.device)
    ang = math.atan2(float(s), float(torch.dot(z, d)))
    return torch.cat([axis / s * math.sin(ang / 2), torch.tensor([math.cos(ang / 2)], device=d.device)])


class Arm:
    """One ElRobot driven by differential inverse kinematics, all in the world frame."""

    POINT = (0.0, 1.0, 0.0)  # the gripper points along its local +y (URDF)
    CLOSE = (1.0, 0.0, 0.0)  # the jaws close along its local x (URDF)

    def __init__(self, robot: Articulation, dev):
        self.robot, self.dev = robot, dev
        # Inverse kinematics drives motors 1, 2, 3, 4 and 6. Motors 5 and 7 are the two roll joints: in the team's real
        # demonstrations they hardly move (std 0.012 and 0.003 in normalised units), so here they are held still.
        # Letting them move made the gripper twist in ways the real arm never does.
        self.arm_ids, _ = robot.find_joints("rev_motor_0[12346]", preserve_order=True)
        self.roll_ids, _ = robot.find_joints("rev_motor_0[57]", preserve_order=True)
        self.all_arm, _ = robot.find_joints("rev_motor_0[1-7]", preserve_order=True)
        # Motor 8 and the two jaws are commanded together: the converter dropped the URDF's gearing between them
        # (mimic joints), so the jaws would not follow the motor by themselves.
        self.grip_ids, self.grip_names = robot.find_joints("rev_motor_08.*")
        self.ee = robot.find_bodies("Gripper_Base_v1_1")[0][0]
        self.jac_body = self.ee - 1 if robot.is_fixed_base else self.ee
        self.jac_joints = [j + robot.num_base_dofs for j in self.arm_ids]
        self.limits = robot.data.joint_pos_limits.torch[0]
        mid, half = (self.limits[:, 0] + self.limits[:, 1]) / 2, (self.limits[:, 1] - self.limits[:, 0]) / 2
        self.start = (mid + 0.2 * half * torch.sin(torch.arange(robot.num_joints, device=dev) * 0.6)).unsqueeze(0)
        self.point = torch.tensor(self.POINT, device=dev)
        close = torch.tensor(self.CLOSE, device=dev)
        self.local = torch.stack([self.point, close, torch.linalg.cross(self.point, close)], dim=1)
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=1, device=dev
        )
        self.q_min = self.q_max = None  # how far each arm motor travels while driven (for the report)
        self.grip(0.9)

    def seed(self, angles: dict):
        """Replace the start posture: {joint name: angle [rad]} for the joints given, the others unchanged."""
        names = list(self.robot.joint_names)
        for n, v in angles.items():
            self.start[0, names.index(n)] = v

    def grip(self, closed: float):
        """closed = 0 open ... 1 shut. Motor 8 up brings the jaws together (jaw 08_1 down, jaw 08_2 up)."""
        way = {"rev_motor_08": 1.0, "rev_motor_08_1": 0.0, "rev_motor_08_2": 1.0}
        frac = lambda n: closed if way[n] == 1.0 else 1.0 - closed
        self.grip_des = torch.stack(
            [self.limits[j, 0] + (self.limits[j, 1] - self.limits[j, 0]) * frac(n) for j, n in zip(self.grip_ids, self.grip_names)]
        ).unsqueeze(0)

    def pose(self):
        p = self.robot.data.body_pose_w.torch[:, self.ee]
        return p[:, 0:3].clone(), p[:, 3:7].clone()

    def pointing(self):
        _, q = self.pose()
        return matrix_from_quat(q)[0] @ self.point

    def tip(self):
        p, _ = self.pose()
        return p[0] + self.pointing() * TIP

    def quat_for(self, point_dir: torch.Tensor, close_dir: torch.Tensor) -> torch.Tensor:
        """Orientation with the gripper pointing along point_dir and its jaws closing along close_dir."""
        d = point_dir / point_dir.norm()
        c = close_dir - d * torch.dot(close_dir, d)
        c = c / c.norm()
        world = torch.stack([d, c, torch.linalg.cross(d, c)], dim=1)
        return quat_from_matrix((world @ self.local.T).unsqueeze(0))

    def hold_start(self):
        self.robot.set_joint_position_target_index(target=self.start)

    def drive(self, pos_w: torch.Tensor, point_dir: torch.Tensor):
        """Move the gripper to pos_w and aim it along point_dir. Its roll about that direction is left alone.

        Five motors (1, 2, 3, 4, 6) for five things to control: three of position, two of aim. Damped least squares
        on the simulator's Jacobian. Motors 5 and 7 stay where the start posture put them.
        """
        p, q = self.pose()
        cur = matrix_from_quat(q)[0] @ self.point
        goal = point_dir / point_dir.norm()
        # Position counts for more than aim: with the rolls held, an arm that has to turn sideways cannot hold an
        # exact aim, and should then keep its place and let the aim be a few degrees off.
        err = torch.cat([pos_w.reshape(3) - p[0], AIM_WEIGHT * torch.linalg.cross(cur, goal)])  # no roll part
        jac = self.robot.data.body_link_jacobian_w.torch[0, self.jac_body][:, self.jac_joints].clone()  # 6 x 5
        jac[3:] *= AIM_WEIGHT
        dq = jac.T @ torch.linalg.solve(jac @ jac.T + (0.05**2) * torch.eye(6, device=self.dev), err)
        q_now = self.robot.data.joint_pos.torch[:, self.arm_ids]
        lo, hi = self.limits[self.arm_ids, 0], self.limits[self.arm_ids, 1]
        des = torch.clamp(q_now + dq.unsqueeze(0), lo, hi)
        # one full-length command per step: motors 1-4 and 6 from the solver, 5 and 7 held, the gripper as asked
        cmd = self.start.clone()
        cmd[:, self.arm_ids] = des
        cmd[:, self.grip_ids] = self.grip_des
        self.robot.set_joint_position_target_index(target=cmd)
        q = self.robot.data.joint_pos.torch[0, self.all_arm]
        self.q_min = q.clone() if self.q_min is None else torch.minimum(self.q_min, q)
        self.q_max = q.clone() if self.q_max is None else torch.maximum(self.q_max, q)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)
        if args_cli.hero:
            # the room is a panorama picture wrapped round the scene: it is both the background and the light
            cfg = sim_utils.DomeLightCfg(intensity=1100.0, color=(1.0, 1.0, 1.0), texture_file=os.path.expanduser(args_cli.hero))
        else:
            floor = sim_utils.GroundPlaneCfg()
            floor.func("/World/defaultGroundPlane", floor, translation=(0.0, 0.0, -0.74))  # the hall floor
            cfg = sim_utils.DomeLightCfg(intensity=900.0, color=(0.9, 0.93, 1.0))
        cfg.func("/World/Light", cfg)
        surf = lambda c, r: sim_utils.PreviewSurfaceCfg(diffuse_color=c, roughness=r)
        wood = sim_utils.CuboidCfg(size=(1.05, 1.20, 0.03), visual_material=surf((0.72, 0.56, 0.36), 0.8),
                                   collision_props=sim_utils.UsdPhysicsCollisionCfg())
        wood.func("/World/TableTop", wood, translation=(0.0, 0.38, -0.015))  # top surface at z = 0
        leg = sim_utils.CuboidCfg(size=(0.05, 0.05, 0.71), visual_material=surf((0.20, 0.33, 0.27), 0.6))
        for k, (lx, ly) in enumerate([(-0.45, -0.15), (0.45, -0.15), (-0.45, 0.91), (0.45, 0.91)]):
            leg.func(f"/World/TableLeg{k}", leg, translation=(lx, ly, -0.385))
        black = surf((0.03, 0.03, 0.035), 0.5)
        post = sim_utils.CuboidCfg(size=(0.05, 0.05, 1.45), visual_material=black)
        post.func("/World/CamPost", post, translation=(-0.58, 0.40, -0.015))  # the tripod post beside the table
        boom = sim_utils.CuboidCfg(size=(0.60, 0.045, 0.03), visual_material=black)
        boom.func("/World/CamBoom", boom, translation=(-0.30, 0.40, 0.70))
        camblock = sim_utils.CuboidCfg(size=(0.06, 0.05, 0.035), visual_material=black)
        camblock.func("/World/CamHead", camblock, translation=(-0.01, 0.40, 0.70))
        card = sim_utils.CuboidCfg(size=(0.62, 0.56, 0.003), visual_material=surf((0.66, 0.52, 0.37), 0.95))
        card.func("/World/Cardboard", card, translation=(-0.02, 0.45, 0.0015))
        robot_a = Articulation(cfg=arm_cfg("/World/PullArm"))
        robot_b = Articulation(cfg=arm_cfg("/World/MoveArm", B_BASE, B_YAW))
        leaders = []  # the two leader arms are only used to record demonstrations: they are not on the bench in use

        def make_camera(path: str, width: int = 640, height: int = 540):
            sim_utils.create_prim(path, "Xform")
            return instantiate(
                CameraCfg(
                    prim_path=f"{path}/Camera",
                    update_period=0,
                    height=height,
                    width=width,
                    data_types=["rgb"],
                    renderer_cfg=IsaacRtxRendererCfg(),
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0)
                    ),
                )
            )

        cam_side = make_camera("/World/CamSide", 720, 1112) if args_cli.hero else make_camera("/World/CamSide")
        cam_top, cam_wrist = make_camera("/World/CamTop"), make_camera("/World/CamWrist")
        mat = lambda c, m=0.0: sim_utils.PreviewSurfaceCfg(diffuse_color=c, metallic=m)
        skin = (0.93, 0.72, 0.60)
        rig = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/rig",
                markers={
                    "tip": sim_utils.SphereCfg(radius=0.0095, visual_material=mat(skin)),
                    "bone_flat": sim_utils.CylinderCfg(radius=0.0095, height=FINGER_LEN, visual_material=mat(skin)),
                    "bone_up": sim_utils.CylinderCfg(radius=0.0095, height=FINGER_UP, visual_material=mat(skin)),
                    "palm": sim_utils.CuboidCfg(size=(0.10, 0.105, 0.030), visual_material=mat(skin)),
                    "wire": sim_utils.SphereCfg(radius=0.0018, visual_material=mat((0.62, 0.64, 0.68), 0.7)),
                    "tube": sim_utils.CylinderCfg(radius=0.0075, height=TUBE_LEN, visual_material=mat((0.80, 0.82, 0.85), 0.5)),
                    "band": sim_utils.CylinderCfg(radius=0.0078, height=0.022, visual_material=mat((0.38, 0.20, 0.47))),
                    "cap": sim_utils.CylinderCfg(radius=0.0055, height=CAP_LEN, visual_material=mat((0.93, 0.93, 0.93))),
                    "marker": sim_utils.CylinderCfg(radius=0.0082, height=0.005, visual_material=mat((0.90, 0.12, 0.10))),
                    "limit": sim_utils.CylinderCfg(radius=0.0080, height=0.002, visual_material=mat((0.05, 0.07, 0.10))),
                },
            )
        )
        N_ROD, N_BAIL, nf = 8, 26, len(FINGERS)
        marker_idx = torch.tensor([0] * nf + [1] * nf + [2] * nf + [3] + [4] * (N_ROD + N_BAIL) + [5, 6, 7, 8, 9], device=sim.device)
        sim.set_camera_view([-0.45, 0.25, 0.5], [0.0, 0.42, 0.05])
        sim.reset()
        os.makedirs(OUT, exist_ok=True)
        dev, dt = sim.device, sim.get_physics_dt()
        ident = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=dev)
        up = torch.tensor([0.0, 0.0, 1.0], device=dev)
        A, B = Arm(robot_a, dev), Arm(robot_b, dev)
        cam_off = torch.tensor([0.0, -0.10, 0.04], device=dev)   # the camera sits beside the wrist, towards the pull arm
        cam_look = torch.tensor([0.0, 0.05, 0.0], device=dev)    # and looks slightly towards the hand
        B.grip(0.1)  # the move arm starts with its jaws open
        # the pull arm starts in the pose it works in (from tools/seed_pose.py): low, level, no roll
        A.seed({"rev_motor_01": 0.0, "rev_motor_02": 0.04, "rev_motor_03": 1.057, "rev_motor_04": 0.072,
                "rev_motor_05": 0.0, "rev_motor_06": -1.169, "rev_motor_07": 0.0,
                "rev_motor_08": 1.98, "rev_motor_08_1": -0.0229, "rev_motor_08_2": 0.0229})  # jaws closed on the ring
        robot_a.write_joint_position_to_sim_index(position=A.start)
        robot_a.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(A.start))
        # Start posture from tools/seed_pose.py: jaw tips over the scale near the loop, 9 cm up, tilted 30 degrees forward of
        # straight down, jaws straddling the scale. Straight down runs the wrist joint into its limit during the carry.
        B.seed({"rev_motor_01": -0.598, "rev_motor_02": -0.069, "rev_motor_03": -0.126, "rev_motor_04": 0.195,
                "rev_motor_05": -0.293, "rev_motor_06": 1.155, "rev_motor_07": -0.905,
                "rev_motor_08": 0.9, "rev_motor_08_1": -0.0104, "rev_motor_08_2": 0.0104})  # jaws half open
        robot_b.write_joint_position_to_sim_index(position=B.start)
        robot_b.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(B.start))
        for lead in leaders:
            lead.set_joint_position_target_index(target=lead.data.default_joint_pos.torch.clone())

        def step_sim():
            for r in (robot_a, robot_b, *leaders):
                r.write_data_to_sim()
            sim.step()
            for r in (robot_a, robot_b, *leaders):
                r.update(dt)
            # the camera on the move arm's wrist rides with its gripper and looks at what the jaws reach for
            pb_w, _ = B.pose()
            back = B.pointing()
            eye_w = pb_w[0] - back * 0.02 + cam_off
            cam_wrist.set_world_poses_from_view(eye_w.unsqueeze(0), (pb_w[0] + back * (TIP + 0.03) + cam_look).unsqueeze(0))
            for c in (cam_side, cam_top, cam_wrist):
                c.update(dt=dt)

        for _ in range(int(1.5 * SETTLE / dt)):
            A.hold_start()
            B.hold_start()
            step_sim()

        r3 = lambda v: [round(float(x), 3) for x in v]
        print("[TWO] move arm root pose (pos, quat):", r3(robot_b.data.root_pose_w.torch[0]), "| gripper at", r3(B.pose()[0][0]),
              "| pointing", r3(B.pointing()), "| joints", r3(robot_b.data.joint_pos.torch[0]), flush=True)
        print("[TWO] pull arm gripper at", r3(A.pose()[0][0]), "| pointing", r3(A.pointing()), flush=True)
        # --- the bench's directions: u along the scale (from the pull arm outwards), h across it
        pa, _ = A.pose()
        u = pa[0].clone()
        u[2] = 0.0
        u = u / u.norm()
        h = torch.linalg.cross(up, u)
        qa_goal = A.quat_for(u, h)  # horizontal, jaws closing sideways on the ring
        pa_goal = u * REACH + up * HEIGHT
        tip_goal = pa_goal + u * TIP
        base_tip = lambda j: tip_goal + u * SCALE_LEN + h * SPACING * j  # where finger j's tip rests
        # the move arm comes down on the scale from its side: gripper tilted 30 degrees forward of straight down,
        # jaws straddling the scale (closing across it)
        b_fwd = torch.tensor([-math.sin(B_YAW), math.cos(B_YAW), 0.0], device=dev)  # the way the move arm faces
        b_point = -up * math.cos(math.radians(30)) + b_fwd * math.sin(math.radians(30))
        qb_goal = B.quat_for(b_point, torch.linalg.cross(b_point, u))
        grasp_on = lambda j: tip_goal + (base_tip(j) - tip_goal) * (GRASP_AT / SCALE_LEN)  # point on the scale
        b_origin = lambda tip_point: tip_point - b_point * TIP  # gripper origin that puts the jaw tips on tip_point
        park = lambda j: grasp_on(j) + up * HOVER

        pa_start, (pb_start, _) = pa.clone(), B.pose()
        for i in range(int(REACH_T / dt)):
            a = ease(i * dt / (REACH_T * 0.8))
            A.drive(pa_start[0] + (pa_goal - pa_start[0]) * a, u)
            B.drive(pb_start[0] + (b_origin(park(0)) - pb_start[0]) * a, b_point)
            step_sim()
        pa0, _ = A.pose()
        pb0, _ = B.pose()
        print("[TWO] move arm wanted gripper at", r3(b_origin(park(0))), "| got", r3(pb0[0]), "| pointing", r3(B.pointing()), flush=True)
        deg = lambda v, w: float(torch.rad2deg(torch.acos(torch.clamp(torch.dot(v, w), -1, 1))))
        print(f"[TWO] pull arm in place: {float((pa0[0]-pa_goal).norm())*1000:.1f} mm, {deg(A.pointing(), u):.1f} deg | "
              f"move arm in place: {float((pb0[0]-b_origin(park(0))).norm())*1000:.1f} mm, {deg(B.pointing(), b_point):.1f} deg",
              flush=True)

        # cameras: a side view of both arms, and the overhead camera that watches the scale
        centre = tip_goal + u * 0.08 - h * 0.04
        eye = centre - u * 0.30 + h * 0.62 + up * 0.50
        if args_cli.hero:  # like the phone clip: from behind the pull arm, along the scale, the hall beyond
            centre = tip_goal + u * 0.17 - h * 0.05 + up * 0.02
            eye = tip_goal - u * 0.50 + h * 0.20 + up * 0.36
        cam_side.set_world_poses_from_view(eye.unsqueeze(0), centre.unsqueeze(0))
        sim.set_camera_view(eye.tolist(), centre.tolist())
        top_c = tip_goal + u * 0.13
        cam_top.set_world_poses_from_view((top_c + up * 0.60 - u * 0.012).unsqueeze(0), top_c.unsqueeze(0))

        from PIL import Image, ImageDraw, ImageFont

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
            small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except OSError:
            font = small = ImageFont.load_default()

        pull_len = PULL_T + HOLD_T + RELAX_T + REST_T
        carry = [("down", DOWN_T), ("grip", GRIP_T), ("lift", LIFT_T), ("shift", SHIFT_T), ("lower", LOWER_T),
                 ("open", OPEN_T), ("up", UP_T)]
        plan = [("pull", pull_len, 0, 0)] + [(k, d, 0, 1) for k, d in carry] + \
               [("pull", pull_len, 1, 1)] + [(k, d, 1, 0) for k, d in carry]  # (kind, seconds, from finger, to finger)
        cycle_len = sum(d for _, d, _, _ in plan)
        label = {"pull": "Pull arm: {t} finger pulls, the arm gives way above the limit",
                 "down": "Move arm: coming down on the scale", "grip": "Move arm: taking hold",
                 "lift": "Move arm: lifting the loop off the {f} finger", "shift": "Move arm: carrying the loop to the {t} finger",
                 "lower": "Move arm: putting the loop on the {t} finger", "open": "Move arm: letting go",
                 "up": "Move arm: clear"}
        target_a = pa_goal.clone()
        rows, frames, last_kind = [], 0, None
        n_steps = int(args_cli.cycles * cycle_len / dt)
        for step in range(n_steps):
            t = step * dt
            tc = t % cycle_len
            for kind, dur, f_from, f_to in plan:
                if tc < dur:
                    break
                tc -= dur
            a = ease(tc / dur)
            tip_a = A.tip()
            pulls = [0.0] * nf
            force = 0.0
            if kind == "pull":
                pulls[f_to] = pull_at(tc)
                loop_end = base_tip(f_to) + u * pulls[f_to]
                d = loop_end - tip_a
                dist = d.norm()
                dirn = d / dist
                force = K_SPRING * max(0.0, float(dist) - SCALE_LEN)
                if force > args_cli.cap and float((target_a - pa_goal).norm()) < 0.08:
                    target_a = target_a + dirn * args_cli.gain * (force - args_cli.cap) * dt
                b_tip_target = park(f_to)
                B.grip(0.1)
            else:
                # after a pull the arm has given way a few centimetres: it goes back slowly, never in one jump
                back = pa_goal - target_a
                target_a = target_a + back * min(1.0, RETURN_SPEED * dt / max(float(back.norm()), 1e-9))
                g_from, g_to = grasp_on(f_from), grasp_on(f_to)
                if kind == "down":
                    b_tip_target = park(f_from) + (g_from - park(f_from)) * a
                    B.grip(0.1)
                elif kind == "grip":
                    b_tip_target = g_from
                    B.grip(0.1 + 0.8 * a)
                elif kind == "lift":
                    b_tip_target = g_from + up * LIFT * a
                elif kind == "shift":
                    b_tip_target = g_from + (g_to - g_from) * a + up * LIFT
                elif kind == "lower":
                    b_tip_target = g_to + up * LIFT * (1 - a)
                elif kind == "open":
                    b_tip_target = g_to
                    B.grip(0.9 - 0.8 * a)
                else:  # up
                    b_tip_target = g_to + (park(f_to) - g_to) * a
                    B.grip(0.1)
                held = kind in ("grip", "lift", "shift", "lower")
                # while the move arm holds it, the scale runs from the ring through that arm's jaws
                through = B.tip() if held else (g_from if kind == "down" else g_to)
                d = through - tip_a
                dirn = d / d.norm()
                dist = SCALE_LEN
            if kind != last_kind:
                pb, _ = B.pose()
                print(f"[TWO] t={t:5.2f} {kind:5s} {f_from}->{f_to} | move arm tip error "
                      f"{float((B.tip() - b_tip_target).norm())*1000:5.1f} mm, tilt {deg(B.pointing(), b_point):4.1f} deg | "
                      f"pull arm error {float((A.pose()[0][0] - target_a).norm())*1000:5.1f} mm", flush=True)
            last_kind = kind
            robot_a.permanent_wrench_composer.set_forces_and_torques_index(
                forces=(dirn * force).reshape(1, 1, 3), positions=tip_a.reshape(1, 1, 3), body_ids=[A.ee], is_global=True
            )
            A.drive(target_a, u)
            B.drive(b_origin(b_tip_target), b_point)
            step_sim()

            # --- draw the hand: palm on the table, each finger flat for a stretch and then standing up
            flat = lambda v: torch.stack([v[0], v[1], torch.zeros((), device=dev)])
            stand = [flat(base_tip(k) + u * pulls[k]) for k in range(nf)]  # where each finger stands (the loop goes round it)
            tips = torch.stack([st + up * (KNUCKLE_Z + FINGER_UP) for st in stand])
            bones_up = torch.stack([st + up * (KNUCKLE_Z + FINGER_UP / 2) for st in stand])
            bones_flat = torch.stack([st + u * (FINGER_LEN / 2) + up * KNUCKLE_Z for st in stand])
            palm = (flat(tip_goal) + u * (SCALE_LEN + FINGER_LEN + 0.05) + h * SPACING * (nf - 1) / 2 + up * 0.015).unsqueeze(0)
            # --- draw the scale along its line: rod from the jaws, tube, purple band, white cap, wire loop
            along = lambda sdist: tip_a + dirn * sdist
            cap_end = float(dist) - BAIL_LEG          # the loop's two legs start here
            tube_end = cap_end - CAP_LEN
            tube_start = tube_end - TUBE_LEN
            lat = torch.linalg.cross(up, dirn)
            lat = lat / lat.norm()
            rod = torch.stack([along(v) for v in torch.linspace(0.0, max(tube_start, 0.004), N_ROD).tolist()])
            n_leg = (N_BAIL - 10) // 2
            legs = [along(v) + lat * (sgn * BAIL_R) for sgn in (-1.0, 1.0) for v in torch.linspace(cap_end, float(dist), n_leg).tolist()]
            arc = [along(float(dist) + BAIL_R * math.sin(a_)) + lat * (BAIL_R * math.cos(a_))
                   for a_ in torch.linspace(0.0, math.pi, N_BAIL - 2 * n_leg).tolist()]
            bail = torch.stack(legs + arc)
            # the indicator starts by the purple band and runs towards the gripper as the force grows
            window = TUBE_LEN - 0.045
            mark_at = lambda f: along(tube_end - 0.030 - window * min(1.0, f / FULL_SCALE))
            qd = quat_z_to(dirn)
            trans = torch.cat([tips, bones_flat, bones_up, palm, rod, bail,
                               along((tube_start + tube_end) / 2).unsqueeze(0), along(tube_end - 0.014).unsqueeze(0),
                               along(tube_end + CAP_LEN / 2).unsqueeze(0), mark_at(force).unsqueeze(0),
                               mark_at(args_cli.cap).unsqueeze(0)])
            orient = ident.repeat(trans.shape[0], 1)
            orient[nf : 2 * nf] = quat_z_to(u)
            orient[3 * nf] = quat_from_matrix(torch.stack([u, h, up], dim=1).unsqueeze(0))[0]
            orient[-5:] = qd
            rig.visualize(translations=trans, orientations=orient, marker_indices=marker_idx)

            rows.append((round(t, 4), kind, round(force, 4), round(float(torch.dot(A.pose()[0][0] - pa_goal, u)), 5)))
            if step % 2 == 0 and args_cli.hero and not args_cli.no_record:
                Image.fromarray(cam_side.data.output["rgb"][0, ..., :3].cpu().numpy()).save(f"{OUT}/frame_{frames:04d}.png")
                frames += 1
            elif step % 2 == 0 and not args_cli.no_record:
                side = Image.fromarray(cam_side.data.output["rgb"][0, ..., :3].cpu().numpy())
                top = Image.fromarray(cam_top.data.output["rgb"][0, ..., :3].cpu().numpy())
                wrist = Image.fromarray(cam_wrist.data.output["rgb"][0, ..., :3].cpu().numpy())
                im = Image.new("RGB", (side.width * 3, side.height + 118), (14, 20, 31))
                im.paste(side, (0, 118))
                im.paste(top, (side.width, 118))
                im.paste(wrist, (side.width * 2, 118))
                dr = ImageDraw.Draw(im)
                dr.text((28, 14), label[kind].format(f=FINGERS[f_from], t=FINGERS[f_to]), font=font, fill=(244, 247, 251))
                dr.text(
                    (28, 66),
                    f"force {force:4.2f} N  ({force / 9.80665 * 1000:3.0f} gf)     limit {args_cli.cap:.2f} N (200 gf)     "
                    f"scripted motion, simulated     t = {t:4.1f} s",
                    font=small,
                    fill=(255, 110, 100) if force > args_cli.cap * 1.05 else (150, 230, 190),
                )
                dr.text((24, 132), "bench", font=small, fill=(244, 247, 251))
                dr.text((side.width + 24, 132), "overhead camera (reads the scale)", font=small, fill=(244, 247, 251))
                dr.text((side.width * 2 + 24, 132), "camera on the move arm", font=small, fill=(244, 247, 251))
                im.save(f"{OUT}/frame_{frames:04d}.png")
                frames += 1

        with open(f"{OUT}/log.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "phase", "force_N", "pull_arm_give_m"])
            w.writerows(rows)
        forces = [r[2] for r in rows]
        summary = {
            "mode": "two arms: one pulls with a force limit, one moves the scale (scripted)",
            "cap_N": args_cli.cap,
            "max_force_N": round(max(forces, default=0.0), 3),
            "max_force_gf": round(max(forces, default=0.0) / 9.80665 * 1000, 1),
            "seconds_above_110pct_cap": round(sum(1 for v in forces if v > args_cli.cap * 1.1) * dt, 3),
            "max_pull_arm_give_m": round(max((r[3] for r in rows), default=0.0), 4),
            "frames": frames,
        }
        with open(f"{OUT}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("[TWO] summary:", json.dumps(summary), flush=True)
        for name, arm in (("pull arm", A), ("move arm", B)):
            travel = [round(float(v), 2) for v in (arm.q_max - arm.q_min)]
            print(f"[TWO] {name}: travel of motors 1..7 during the run [rad]:", travel, flush=True)


if __name__ == "__main__":
    main()
