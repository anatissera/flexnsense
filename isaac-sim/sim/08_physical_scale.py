"""The two-arm bench with a PHYSICAL scale: a tube with mass and collisions, not a drawing.

What is physical here, and what is not:
  * The scale's tube is a rigid body (TUBE_MASS, collisions on). Its swivel end hangs from the pull arm's jaws on a ball
    joint. It rests on the table, and the arms and the table cannot pass through it.
  * The finger holds the loop end with a spring, F = K * distance (this is the scale's own spring, placed at the loop).
    The pull arm now feels that force through the joint, the way the real arm does, and gives way above the limit.
  * The move arm holds the tube with its jaws, by friction: the jaws close on the tube and stop on it, and nothing
    in the script holds the tube while it is carried (--virtual_hold brings back the old spring between jaws and tube).
  * At load, the script corrects the three gripper joints of the converted arm (their two sides disagree by 36 and
    87 degrees, which left the jaws detached from the gripper), and adds a rubber pad to each of the move arm's jaws.
  * The hand, the wire loop, the cap, the band and the indicator are still drawn only.

Runs ON the workstation:

    source ~/isaac-env.sh && cd ~/IsaacLab
    uv run --extra isaacsim python ~/flexnsense_sim/08_physical_scale.py --tag physical
    PUBLIC_IP=<vm-ip> LIVESTREAM=1 uv run --extra isaacsim python ~/flexnsense_sim/08_physical_scale.py --cycles 100000 --no_record --viz kit

Everything is SCRIPTED; no trained policy runs here.

Writes ~/flexnsense-sim/runs/physical/<tag>/ : frame_*.png (bench | overhead camera | wrist camera), log.csv, summary.json.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Two ElRobot followers: one pulls with a force limit, one moves the scale.")
parser.add_argument("--cap", type=float, default=1.96, help="Force limit [N] (1.96 N = 200 gf).")
parser.add_argument("--gain", type=float, default=1.0, help="Admittance gain B [m/s per N].")
parser.add_argument("--pull", type=float, default=0.06, help="How far the finger pulls away [m].")
parser.add_argument("--tag", default="physical", help="Output folder name under runs/physical/.")
parser.add_argument("--cycles", type=int, default=1, help="Cycles to run (index -> middle -> back to index).")
parser.add_argument("--no_record", action="store_true", help="Do not save frames (for the live stream).")
parser.add_argument("--closeup", action="store_true", help="Third picture: look along the scale at the move arm's jaws "
                    "(to check the grip), not the wrist camera view.")
parser.add_argument("--virtual_hold", action="store_true", help="Carry the scale with a virtual spring between the jaw "
                    "tips and the tube (the old way). Without it the jaws pinch the tube and friction carries it.")
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
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import instantiate
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

ROOT = os.path.expanduser("~/flexnsense-sim")
# The arm model. usd_decomposed/ (from 01b_import_decomposed.sh) has collision shapes that follow the parts; usd/ has
# one coarse hull per part, which wedges thin objects between the jaws. ELROBOT_USD overrides the choice.
_USD_DEC = f"{ROOT}/assets/elrobot/usd_decomposed/elrobot_follower.usd/elrobot_follower/elrobot_follower.usda"
_USD_HULL = f"{ROOT}/assets/elrobot/usd/elrobot_follower.usd/elrobot_follower/elrobot_follower.usda"
USD = os.environ.get("ELROBOT_USD") or (_USD_DEC if os.path.exists(_USD_DEC) else _USD_HULL)
OUT = f"{ROOT}/runs/physical/{args_cli.tag}"

FULL_SCALE = 5.886  # N: the PESOLA reads up to 600 gf
TRAVEL = 0.10  # m: ASSUMED marker travel at full scale
K_SPRING = FULL_SCALE / TRAVEL
SCALE_LEN = 0.23  # m: ring (in the pull arm's jaws) to the finger loop, at zero force
# The PESOLA, from the gripper to the finger: swivel and rod (it lengthens under load) - tube - white cap - wire loop.
# The loop is the C-shaped bail at the top of the tube; it goes round the finger. Lengths estimated from the photo.
TUBE_LEN, CAP_LEN, BAIL_LEG, BAIL_R = 0.165, 0.012, 0.032, 0.013  # SCALE_LEN = rod 0.021 + tube + cap + bail leg
ROD = SCALE_LEN - BAIL_LEG - CAP_LEN - TUBE_LEN  # m: swivel and rod between the jaws and the tube (0.021)
TUBE_MASS = 0.05  # kg, ASSUMED
CTRL_EVERY = 4  # physics steps per arm command: 240 Hz physics, 60 Hz control
AXIS_SIGN = 1.0  # +1 if the tube's local +Z points away from the pull arm (checked at start, see [PHYS])
MAX_YIELD_SPEED = 0.08  # m/s: the fastest the pull arm gives way, however large the force
# The jaws. NormaCore's gripper sheet (hardware/pgripper): stroke 27 mm per jaw, closing force at least 4 kg (39 N).
# The squeeze is capped well below that: a thin aluminium tube needs little, and a soft cap is kinder to the solver.
# Told to shut, the jaws stop on the tube about 8 mm short, so they squeeze with about JAW_K * 8 mm = 3 N each.
JAW_K, JAW_D, JAW_FORCE = 400.0, 20.0, 6.0  # N/m, N.s/m, N
JAW_ARMATURE = float(os.environ.get("DBG_ARM", "0.3"))  # kg: apparent mass of a jaw seen through the gearbox, ASSUMED
JAW_OVER = float(os.environ.get("DBG_OVER", "0.0"))  # m: how far past shut the move arm's jaws are told to go while holding
if os.environ.get("DBG_JAW"):  # test hook: "stiffness,damping,force"
    JAW_K, JAW_D, JAW_FORCE = (float(v) for v in os.environ["DBG_JAW"].split(","))
# Measured on NormaCore's jaw meshes (gripper frame): the flat fingers run from 60 to 104 mm along the gripper, their
# inner faces 50 mm apart when open. The tube is taken in the middle of that stretch, 82 mm from the gripper origin.
GRIP_Y = 0.082
SOLVER_POS, SOLVER_VEL = 32, 4  # solver passes per step for the arms and the tube: a pinched light tube needs them
GRASP_K, GRASP_D, LOOP_D = 150.0, 4.0, 2.4  # dampings near critical for the tube's mass  # the move arm's virtual hold [N/m, N.s/m]; damping at the loop [N.s/m]
GRIPPER_PATH = ("Geometry/base_link/ST3215_1_v1_1/Joint_01_1/ST3215_2_v1_1/Joint_02_1/ST3215_3_v1_1/Joint_03_v1_1/"
                "ST3215_4_v1_1/Joint_04_v1_1/ST3215_5_v1_1/Joint_05_v1_1/ST3215_6_v1_1/Joint_06_v1_1/ST3215_7_v1_1/"
                "Gripper_Base_v1_1")  # the gripper link inside the converted model
FINGER_UP, KNUCKLE_Z = 0.062, 0.018  # m: the raised part of each finger, and the height of the flat part
TIP = 0.10  # m: ASSUMED gripper origin -> point between the jaw tips
REACH, HEIGHT = 0.20, 0.05  # m: the pull arm's gripper, from its base and above the table
GRASP_AT = 0.150  # m along the scale from the ring: where the move arm takes hold. The tube itself runs from 0.021
# to 0.186; beyond that are the cap and the wire loop, which are drawn only, so the jaws must close on the tube.
B_BASE = (0.24, 0.40, 0.0)  # m: the move arm's base, beside the top half of the scale (the pull arm's base is the origin)
B_YAW = math.pi / 2  # the move arm faces the scale
SPACING, LIFT, HOVER, FINGER_LEN = 0.024, 0.04, 0.04, 0.075
SETTLE, REACH_T = 1.0, 4.0
PULL_T, HOLD_T, RELAX_T, REST_T = 2.0, 1.5, 1.2, 0.8
DOWN_T, GRIP_T, LIFT_T, SHIFT_T, LOWER_T, OPEN_T, UP_T = 1.3, 0.9, 1.0, 1.2, 1.0, 0.7, 1.2
AIM_WEIGHT = 0.15  # how much one radian of aim error counts against one metre of position error
RETURN_SPEED = 0.02  # m/s: how fast the pull arm returns to its place once the loop is off the finger
FINGERS = ["index", "middle", "ring", "little"]


PULL_SEED = {"rev_motor_01": 0.0, "rev_motor_02": 0.04, "rev_motor_03": 1.057, "rev_motor_04": 0.072, "rev_motor_05": 0.0,
             "rev_motor_06": -1.169, "rev_motor_07": 0.0, "rev_motor_08": 1.98, "rev_motor_08_1": -0.0229,
             "rev_motor_08_2": 0.0229}  # from tools/seed_pose.py: low, level, jaws closed on the ring
MOVE_SEED = {"rev_motor_01": -0.598, "rev_motor_02": -0.069, "rev_motor_03": -0.126, "rev_motor_04": 0.195,
             "rev_motor_05": -0.293, "rev_motor_06": 1.155, "rev_motor_07": -0.905, "rev_motor_08": 0.9,
             "rev_motor_08_1": -0.0104, "rev_motor_08_2": 0.0104}  # over the scale near the loop, tilted, jaws half open


def arm_cfg(path: str, pos=(0.0, 0.0, 0.0), yaw: float = 0.0, joint_pos: dict | None = None) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=path,
        # Clamped to the table. No collisions between the arm's own parts: the converted model's collision shapes are
        # coarse hulls, and the gripper's small gear presses on the forearm's hull, which turned the wrist whenever
        # the gripper motor was driven.
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD, fix_root_link=True, articulation_props=[PhysxArticulationCfg(
                enabled_self_collisions=False, solver_position_iteration_count=SOLVER_POS,
                solver_velocity_iteration_count=SOLVER_VEL)]
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos, rot=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)), joint_pos=joint_pos or {".*": 0.0}
        ),
        actuators={
            "servos": ImplicitActuatorCfg(
                joint_names_expr=["rev_motor_0[1-7]"], joint_effort_limit=2.94, joint_velocity_limit=4.71,
                stiffness=40.0, damping=2.0, armature=float(os.environ.get("DBG_SERVO_ARM", "0.0")),
            ),
            # The gripper. sim/dbg_gripper.py showed that driving motor 8 (the gear) is what turns the wrist in this
            # converted model (3.2 rad in 4 s), while driving the two jaws directly is clean. So the gear is left
            # free and the jaws are driven: stiffness in N/m, force limit in N.
            "gear": ImplicitActuatorCfg(
                joint_names_expr=["rev_motor_08"], joint_effort_limit=2.94, joint_velocity_limit=4.71,
                stiffness=0.0, damping=0.5,
            ),
            "jaws": ImplicitActuatorCfg(
                joint_names_expr=["rev_motor_08_[12]"], joint_effort_limit=JAW_FORCE,
                joint_velocity_limit=float(os.environ.get("DBG_JAWVEL", "0.2")),
                # The jaws are not free 12 g sliders: the servo turns them through a 345:1 gearbox, whose rotor adds a
                # large apparent mass. Without it the light jaws bounce on what they pinch and the scene blows up.
                armature=JAW_ARMATURE,
                stiffness=JAW_K, damping=JAW_D,
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
        self.grip_ids, self.grip_names = robot.find_joints("rev_motor_08_[12]")
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
        self._tick, self._cmd = 0, None
        self.grip(0.9)

    def seed(self, angles: dict):
        """Replace the start posture: {joint name: angle [rad]} for the joints given, the others unchanged."""
        names = list(self.robot.joint_names)
        for n, v in angles.items():
            self.start[0, names.index(n)] = v

    def grip(self, closed: float, over: float = 0.0):
        """closed = 0 open ... 1 shut. Motor 8 up brings the jaws together (jaw 08_1 down, jaw 08_2 up).

        over [m]: ask each jaw to go this far past shut, scaled by `closed`. The jaws stop on what they hold, and the
        soft drive then presses with a steady force."""
        way = {"rev_motor_08_1": 0.0, "rev_motor_08_2": 1.0}
        frac = lambda n: closed if way[n] == 1.0 else 1.0 - closed
        push = lambda n: (over if way[n] == 1.0 else -over) * closed
        self.grip_des = torch.stack(
            [self.limits[j, 0] + (self.limits[j, 1] - self.limits[j, 0]) * frac(n) + push(n)
             for j, n in zip(self.grip_ids, self.grip_names)]
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
        self._tick += 1
        if self._cmd is not None and (self._tick - 1) % CTRL_EVERY != 0:  # between control ticks: keep the last command
            self.robot.set_joint_position_target_index(target=self._cmd)
            return
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
        self._cmd = cmd
        self.robot.set_joint_position_target_index(target=cmd)
        q = self.robot.data.joint_pos.torch[0, self.all_arm]
        self.q_min = q.clone() if self.q_min is None else torch.minimum(self.q_min, q)
        self.q_max = q.clone() if self.q_max is None else torch.maximum(self.q_max, q)


def main():
    # A light tube on springs needs a fine physics step: 240 steps a second, a picture every 4th step. The arms are
    # still commanded 60 times a second (CTRL_EVERY), as before.
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device, dt=1.0 / float(os.environ.get("DBG_HZ", "240")), render_interval=4
    )
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
        wood.func("/World/TableTop", wood, translation=(0.0, 0.38, -0.015 - float(os.environ.get("DBG_TABLE_DROP", "0"))))  # top surface at z = 0
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
        # both arms are created in their working postures, so the tube and the jaws agree from the first instant
        robot_a = Articulation(cfg=arm_cfg("/World/PullArm", joint_pos=PULL_SEED))
        robot_b = Articulation(cfg=arm_cfg("/World/MoveArm", B_BASE, B_YAW, joint_pos=MOVE_SEED))
        leaders = []  # the two leader arms are only used to record demonstrations: they are not on the bench in use
        # --- the scale's tube: a real body. It starts where the pull arm's start posture puts the jaws (0.30 m ahead).
        scale = RigidObject(
            cfg=RigidObjectCfg(
                prim_path="/World/Scale",
                spawn=sim_utils.CylinderCfg(
                    radius=0.0075, height=TUBE_LEN, axis="Z",
                    rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg(), PhysxRigidBodyCfg(
                        max_depenetration_velocity=0.3, solver_position_iteration_count=SOLVER_POS,
                        solver_velocity_iteration_count=SOLVER_VEL)], mass_props=sim_utils.MassCfg(mass=TUBE_MASS),
                    collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.82, 0.85), metallic=0.5),
                ),
                # turned a quarter turn about X so that its own +Z (its length) points along the bench's +Y, away from
                # the pull arm: quaternion (x, y, z, w)
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, REACH + TIP + ROD + TUBE_LEN / 2, HEIGHT), rot=(-0.70710678, 0.0, 0.0, 0.70710678)
                ),
            )
        )
        import omni.usd
        from pxr import Gf, Sdf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        grip_path = f"/World/PullArm/{GRIPPER_PATH}"
        assert stage.GetPrimAtPath(grip_path).IsValid(), f"gripper link not found at {grip_path}"
        # a thin tube has almost no inertia about its own length (1.4e-6 kg.m2), which the solver handles badly when
        # it is pinched: that one value is raised; the other two are the true ones for a 165 mm, 50 g tube
        mass_api = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/World/Scale"))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1.1e-4, 1.1e-4, 2.0e-5))
        ball = UsdPhysics.SphericalJoint.Define(stage, "/World/ScaleSwivel")  # the swivel at the top of the scale
        ball.CreateBody0Rel().SetTargets([Sdf.Path(grip_path)])
        ball.CreateBody1Rel().SetTargets([Sdf.Path("/World/Scale")])
        ball.CreateLocalPos0Attr(Gf.Vec3f(0.0, TIP, 0.0))  # between the jaw tips (the gripper points along its +y)
        # The tube's length is its own Z axis. Its swivel end is a rod's length beyond the end that faces the pull arm.
        ball.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, AXIS_SIGN * -(ROD + TUBE_LEN / 2)))
        ball.CreateLocalRot0Attr(Gf.Quatf(1.0))
        ball.CreateLocalRot1Attr(Gf.Quatf(1.0))
        ball.CreateExcludeFromArticulationAttr(True)
        ball.CreateCollisionEnabledAttr(False)
        # the tube must not collide with the jaws that hold it
        rel = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath("/World/Scale")).CreateFilteredPairsRel()
        from pxr import Usd

        filtered = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(grip_path)) if os.environ.get("DBG_FILTER") else []:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
                rel.AddTarget(prim.GetPath())
                filtered.append(prim.GetName())
        print("[PHYS] the tube ignores collisions with:", filtered, flush=True)

        # --- put the jaws back on the grippers. In the converted model the two sliding joints that carry the jaws have
        # frames on the jaw side that do not agree with where the jaw is drawn, so the physics engine moved both jaws
        # about 10 cm off the gripper and turned them 36 degrees: the arm closed fingers that were not there. Here the
        # jaw-side frame of each joint is recomputed from where the parts are drawn (all joints at zero).
        from pxr import UsdGeom

        xf0 = UsdGeom.XformCache()
        for arm_name in ("PullArm", "MoveArm"):
            for jp in stage.GetPrimAtPath(f"/World/{arm_name}/Physics").GetChildren():
                joint = UsdPhysics.Joint(jp)
                t0, t1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
                if not jp.IsA(UsdPhysics.Joint) or not t0 or not t1:
                    continue
                b0, b1 = stage.GetPrimAtPath(t0[0]), stage.GetPrimAtPath(t1[0])

                def side(rot_attr, pos_attr):
                    q = rot_attr.Get()
                    m = Gf.Matrix4d().SetRotate(Gf.Quatd(q.GetReal(), Gf.Vec3d(*q.GetImaginary())))
                    m.SetTranslateOnly(Gf.Vec3d(*pos_attr.Get()))
                    return m

                w0 = side(joint.GetLocalRot0Attr(), joint.GetLocalPos0Attr()) * xf0.GetLocalToWorldTransform(b0)
                w1 = side(joint.GetLocalRot1Attr(), joint.GetLocalPos1Attr()) * xf0.GetLocalToWorldTransform(b1)
                gap = (w0.ExtractTranslation() - w1.ExtractTranslation()).GetLength()
                dq = w0.RemoveScaleShear().ExtractRotationQuat() * w1.RemoveScaleShear().ExtractRotationQuat().GetInverse()
                turn = math.degrees(2 * math.acos(min(1.0, abs(dq.GetReal()))))
                if gap > 5e-4 or turn > 0.5:
                    in_b1 = w0 * xf0.GetLocalToWorldTransform(b1).GetInverse()
                    q1 = in_b1.RemoveScaleShear().ExtractRotationQuat()
                    joint.GetLocalPos1Attr().Set(Gf.Vec3f(*in_b1.ExtractTranslation()))
                    joint.GetLocalRot1Attr().Set(Gf.Quatf(float(q1.GetReal()), Gf.Vec3f(*q1.GetImaginary())))
                    print(f"[PHYS] {arm_name} joint {jp.GetName()}: its two sides disagreed by {gap * 1000:.1f} mm and {turn:.1f} deg "
                          "with all joints at zero; corrected", flush=True)

        # --- finger pads on the move arm's jaws. The converted meshes do not give the fingers a usable gripping
        # surface, so each finger gets a flat rubber pad where NormaCore's mesh has its inner face (measured from the
        # STL files: fingers from 60 to 104 mm along the gripper, inner faces at +26.5 and -23.7 mm across it).
        # A pad is part of its jaw: it moves with it and collides like it.
        PAD_SIZE = (0.004, 0.040, 0.026)  # m: thickness across the jaws, length along the finger, width
        from pxr import UsdGeom

        xf = UsdGeom.XformCache()
        g_inv = xf.GetLocalToWorldTransform(stage.GetPrimAtPath(f"/World/MoveArm/{GRIPPER_PATH}")).GetInverse()
        for jaw, face_x in (("Gripper_Jaw_01_v1_1", 0.0275), ("Gripper_Jaw_02_v1_1", -0.0247)):
            jaw_path = f"/World/MoveArm/{GRIPPER_PATH}/{jaw}"
            assert stage.GetPrimAtPath(jaw_path).IsValid(), f"jaw link not found at {jaw_path}"
            # the converted model shares its parts as read-only copies; a part must be made its own before adding to it
            for prefix in Sdf.Path(jaw_path).GetPrefixes():
                pr = stage.GetPrimAtPath(prefix)
                if pr.IsInstance():
                    pr.SetInstanceable(False)
            # The jaw's own frame in the converted model is turned and shifted against the gripper's frame, so the
            # pad's place (given in the gripper's frame, where the meshes were measured) is carried into the jaw's.
            jaw_in_gripper = xf.GetLocalToWorldTransform(stage.GetPrimAtPath(jaw_path)) * g_inv
            local = Gf.Matrix4d().SetTranslate(Gf.Vec3d(face_x, GRIP_Y, 0.0)) * jaw_in_gripper.GetInverse()
            tr, qt = local.ExtractTranslation(), local.ExtractRotationQuat()
            print(f"[PHYS] {jaw}: its frame in the gripper's frame: origin {[round(v, 3) for v in jaw_in_gripper.ExtractTranslation()]} | "
                  f"pad placed at {[round(v, 3) for v in tr]} in the jaw's frame", flush=True)
            # Built directly: a unit cube under the jaw, sized in the jaw's own units (its frame is in millimetres).
            unit = jaw_in_gripper.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)).GetLength()
            if os.environ.get("DBG_NOPAD"):
                continue
            cube = UsdGeom.Cube.Define(stage, f"{jaw_path}/Pad")
            cube.CreateSizeAttr(1.0)
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.05, 0.06)])
            cube.AddTranslateOp().Set(Gf.Vec3d(*tr))
            cube.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(qt)
            cube.AddScaleOp().Set(Gf.Vec3f(*(v / unit for v in PAD_SIZE)))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(cube.GetPrim())
            rng = box.ComputeAlignedRange()
            g_l2w = xf.GetLocalToWorldTransform(stage.GetPrimAtPath(f"/World/MoveArm/{GRIPPER_PATH}"))
            c_g = g_l2w.GetInverse().Transform(rng.GetMidpoint())
            print(f"[PHYS] {jaw}: pad size in the world {[round(v * 1000, 1) for v in rng.GetSize()]} mm, centre in the gripper's "
                  f"frame {[round(v * 1000, 1) for v in c_g]} mm (jaw unit {unit:.4f} m)", flush=True)
        for arm_name in ("PullArm", "MoveArm"):
            for prim in Usd.PrimRange(stage.GetPrimAtPath(f"/World/{arm_name}")):
                attr = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
                if attr and attr.HasValue():
                    print(f"[PHYS] {arm_name}: collisions between its own parts = {attr.Get()} (set on {prim.GetName()})", flush=True)

        if os.environ.get("DBG_MEASURE"):
            # where are the gripper's parts, in the frame of the simulated gripper body? (all joints at zero)
            import numpy as np
            from pxr import UsdGeom

            cache = UsdGeom.XformCache()
            g_prim = stage.GetPrimAtPath(f"/World/MoveArm/{GRIPPER_PATH}")
            g_inv = cache.GetLocalToWorldTransform(g_prim).GetInverse()
            out = {}
            for child in g_prim.GetChildren():
                pts = []
                for pr in Usd.PrimRange(child, Usd.TraverseInstanceProxies()):
                    if pr.IsA(UsdGeom.Mesh):
                        m = cache.GetLocalToWorldTransform(pr) * g_inv
                        pts += [m.Transform(Gf.Vec3d(*v)) for v in (UsdGeom.Mesh(pr).GetPointsAttr().Get() or [])]
                if pts:
                    out[child.GetName()] = np.array([[v[0], v[1], v[2]] for v in pts])
                    print(f"[MEASURE] {child.GetName()}: {len(pts)} points", flush=True)
            np.savez(os.path.expanduser("~/gripper_points.npz"), **out)

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
                    "marker": sim_utils.SphereCfg(radius=0.0048, visual_material=mat((0.92, 0.08, 0.06))),  # a red dot
                    "limit": sim_utils.CylinderCfg(radius=0.0080, height=0.002, visual_material=mat((0.05, 0.07, 0.10))),
                },
            )
        )
        N_ROD, N_BAIL, nf = 8, 26, len(FINGERS)
        marker_idx = torch.tensor([0] * nf + [1] * nf + [2] * nf + [3] + [4] * (N_ROD + N_BAIL) + [5, 6, 7, 8, 9, 8, 8], device=sim.device)
        sim.set_camera_view([-0.45, 0.25, 0.5], [0.0, 0.42, 0.05])
        sim.reset()
        os.makedirs(OUT, exist_ok=True)
        dev, dt = sim.device, sim.get_physics_dt()
        ident = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=dev)
        up = torch.tensor([0.0, 0.0, 1.0], device=dev)
        A, B = Arm(robot_a, dev), Arm(robot_b, dev)
        for nm, rb in (("pull arm", robot_a), ("move arm", robot_b), ("tube", scale)):
            for attr in ("root_view", "root_physx_view"):
                view = getattr(rb, attr, None)
                if view is not None and hasattr(view, "max_shapes"):
                    print(f"[PHYS] {nm}: {view.max_shapes} collision shapes in the physics engine", flush=True)
                    break
            else:
                print(f"[PHYS] {nm}: could not count collision shapes; views: {[a for a in dir(rb) if 'view' in a]}", flush=True)
        cam_off = torch.tensor([0.0, -0.10, 0.04], device=dev)   # the camera sits beside the wrist, towards the pull arm
        cam_look = torch.tensor([0.0, 0.05, 0.0], device=dev)    # and looks slightly towards the hand
        B.grip(0.1)  # the move arm starts with its jaws open
        A.seed(PULL_SEED)
        B.seed(MOVE_SEED)
        for lead in leaders:
            lead.set_joint_position_target_index(target=lead.data.default_joint_pos.torch.clone())

        def step_sim():
            for r in (robot_a, robot_b, scale, *leaders):
                r.write_data_to_sim()
            sim.step()
            for r in (robot_a, robot_b, scale, *leaders):
                r.update(dt)
            # the camera on the move arm's wrist rides with its gripper and looks at what the jaws reach for
            pb_w, _ = B.pose()
            back = B.pointing()
            eye_w = pb_w[0] - back * 0.02 + cam_off
            if args_cli.closeup:  # from the pull arm's side, above the tube, looking along it at the fingers
                jaws_at = pb_w[0] + back * GRIP_Y
                cam_wrist.set_world_poses_from_view(
                    (jaws_at + torch.tensor([float(v) for v in os.environ.get("DBG_EYE", "0.0,-0.18,0.006").split(",")], device=jaws_at.device)).unsqueeze(0), jaws_at.unsqueeze(0)
                )
            else:
                cam_wrist.set_world_poses_from_view(eye_w.unsqueeze(0), (pb_w[0] + back * (TIP + 0.03) + cam_look).unsqueeze(0))
            for c in (cam_side, cam_top, cam_wrist):
                c.update(dt=dt)

        z_axis = torch.tensor([0.0, 0.0, AXIS_SIGN], device=dev)
        first_finger = torch.tensor([0.0, REACH + TIP + SCALE_LEN, HEIGHT], device=dev)  # where the index finger stands

        def loop_on(point: torch.Tensor):
            """The loop is on a finger standing at `point`: the scale's spring pulls the loop there."""
            sp_ = scale.data.root_pose_w.torch[0]
            ax_ = matrix_from_quat(sp_[3:7].unsqueeze(0))[0] @ z_axis
            lp_ = sp_[0:3] + ax_ * (TUBE_LEN / 2 + CAP_LEN + BAIL_LEG)
            f_ = K_SPRING * (point - lp_) - LOOP_D * scale.data.root_lin_vel_w.torch[0]
            scale.permanent_wrench_composer.set_forces_and_torques_index(
                forces=f_.reshape(1, 1, 3), positions=lp_.reshape(1, 1, 3), body_ids=[0], is_global=True
            )

        for _ in range(int(1.5 * SETTLE / dt)):
            loop_on(first_finger)
            A.hold_start()
            B.hold_start()
            step_sim()

        r3 = lambda v: [round(float(x), 3) for x in v]
        print("[TWO] move arm root pose (pos, quat):", r3(robot_b.data.root_pose_w.torch[0]), "| gripper at", r3(B.pose()[0][0]),
              "| pointing", r3(B.pointing()), "| joints", r3(robot_b.data.joint_pos.torch[0]), flush=True)
        print("[TWO] pull arm gripper at", r3(A.pose()[0][0]), "| pointing", r3(A.pointing()), flush=True)
        sp0 = scale.data.root_pose_w.torch[0]
        ax0 = matrix_from_quat(sp0[3:7].unsqueeze(0))[0] @ torch.tensor([0.0, 0.0, AXIS_SIGN], device=dev)
        print("[PHYS] after settling: tube centre", r3(sp0[0:3]), "axis", r3(ax0), "| its swivel end", r3(sp0[0:3] - ax0 * (ROD + TUBE_LEN / 2)),
              "| jaw tips", r3(A.tip()), "| gripper origin", r3(A.pose()[0][0]), "pointing", r3(A.pointing()), flush=True)
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
            loop_on(first_finger)
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
        FRAME_EVERY = max(1, round(1.0 / (30.0 * dt)))  # 30 pictures per simulated second
        rows, frames, last_kind = [], 0, None
        centre_fix = torch.zeros(3, device=dev)  # sideways correction that puts the tube midway between the jaws
        n_steps = int(args_cli.cycles * cycle_len / dt)
        for step in range(n_steps):
            t = step * dt
            if t > float(os.environ.get("DBG_STOP_T", "1e9")):
                break
            tc = t % cycle_len
            for kind, dur, f_from, f_to in plan:
                if tc < dur:
                    break
                tc -= dur
            a = ease(tc / dur)
            # --- where the tube really is: its centre, its axis, its two ends, the loop beyond its cap
            sp = scale.data.root_pose_w.torch[0]
            if not bool(torch.isfinite(sp).all()) or float(sp[0:3].norm()) > 5.0:
                print(f"[TWO] STOPPED at t={t:.2f}: the tube left the bench", flush=True)
                break
            centre_s = sp[0:3].clone()
            dirn = matrix_from_quat(sp[3:7].unsqueeze(0))[0] @ torch.tensor([0.0, 0.0, AXIS_SIGN], device=dev)
            tip_a = centre_s - dirn * (ROD + TUBE_LEN / 2)  # the swivel point (in the jaws)
            loop_pt = centre_s + dirn * (TUBE_LEN / 2 + CAP_LEN + BAIL_LEG)  # centre of the wire loop
            vel_s = scale.data.root_lin_vel_w.torch[0]
            dist = SCALE_LEN
            pulls = [0.0] * nf
            force = 0.0
            on_scale, at_point = torch.zeros(3, device=dev), centre_s
            on_b = torch.zeros(3, device=dev)
            pb_now, qb_now = B.pose()
            Rb_now = matrix_from_quat(qb_now)[0]
            loc_now = Rb_now.T @ (centre_s + dirn * torch.dot(pb_now[0] - centre_s, dirn) - pb_now[0])

            def finger_holds(k: int):
                """The finger in the loop: the scale's spring, between the loop and where finger k stands."""
                stand_k = base_tip(k) + u * pulls[k]
                pull_vec = K_SPRING * (stand_k - loop_pt)
                return pull_vec - LOOP_D * vel_s, float(pull_vec.norm())

            if kind == "pull":
                pulls[f_to] = pull_at(tc)
                on_scale, force = finger_holds(f_to)
                at_point = loop_pt
                if force > args_cli.cap and float((target_a - pa_goal).norm()) < 0.08:
                    target_a = target_a + dirn * min(args_cli.gain * (force - args_cli.cap), MAX_YIELD_SPEED) * dt
                b_tip_target = park(f_to)
                B.grip(0.1)
            else:
                # after a pull the arm has given way a few centimetres: it goes back slowly, never in one jump
                back = pa_goal - target_a
                target_a = target_a + back * min(1.0, RETURN_SPEED * dt / max(float(back.norm()), 1e-9))
                g_from, g_to = grasp_on(f_from), grasp_on(f_to)
                if kind == "down":
                    # the real tube, 82 mm along the fingers; then slide sideways until it is midway between the jaws
                    grab = centre_s + dirn * (GRASP_AT - ROD - TUBE_LEN / 2) + b_point * (TIP - GRIP_Y)
                    if tc < 1e-9:
                        centre_fix = torch.zeros(3, device=dev)
                    if a > 0.3:
                        centre_fix = centre_fix + Rb_now[:, 0] * float(loc_now[0]) * 4.0 * dt
                    b_tip_target = park(f_from) + (grab - park(f_from)) * a + centre_fix
                    B.grip(0.1)
                elif kind == "grip":
                    b_tip_target = grab_hold
                    B.grip(0.1 + 0.9 * a, JAW_OVER)
                elif kind == "lift":
                    b_tip_target = grab_hold + up * LIFT * a
                elif kind == "shift":
                    b_tip_target = grab_hold + (g_to - g_from) * a + up * LIFT
                elif kind == "lower":
                    b_tip_target = grab_hold + (g_to - g_from) + up * LIFT * (1 - a)
                elif kind == "open":
                    b_tip_target = grab_hold + (g_to - g_from)
                    B.grip(1.0 - 0.9 * a, JAW_OVER)
                else:  # up
                    b_tip_target = grab_hold + (g_to - g_from) + (park(f_to) - grab_hold - (g_to - g_from)) * a
                    B.grip(0.1)
                if kind in ("lift", "shift", "lower") and args_cli.virtual_hold:
                    # the old way: a stiff virtual hold between the jaw tips and the tube
                    at_point = centre_s + dirn * (GRASP_AT - ROD - TUBE_LEN / 2)
                    on_scale = GRASP_K * (B.tip() - at_point) - GRASP_D * vel_s
                    on_b = -on_scale
                elif kind in ("lift", "shift", "lower"):
                    pass  # the jaws hold the tube by friction: no force from the script at all
                else:  # down, grip: still on the finger it came from; open, up: on the finger it was put on
                    on_scale, _ = finger_holds(f_from if kind in ("down", "grip") else f_to)
                    at_point = loop_pt
            if kind != last_kind or (kind in ("grip", "lift") and step % 48 == 0):
                pb, _ = B.pose()
                print(f"[TWO] t={t:5.2f} {kind:5s} {f_from}->{f_to} | move arm tip error "
                      f"{float((B.tip() - b_tip_target).norm())*1000:5.1f} mm, tilt {deg(B.pointing(), b_point):4.1f} deg | "
                      f"pull arm error {float((A.pose()[0][0] - target_a).norm())*1000:5.1f} mm | "
                      f"tube centre height {float(centre_s[2])*1000:5.1f} mm, loop height {float(loop_pt[2])*1000:5.1f} mm, "
                      f"loop to finger {float((loop_pt - base_tip(f_to)).norm())*1000:5.1f} mm", flush=True)
                # where is the tube, seen from the move arm's gripper? (x across the jaws, y along them, z through them)
                pbq = B.pose()
                Rb = matrix_from_quat(pbq[1])[0]
                near = centre_s + dirn * torch.dot(pbq[0][0] - centre_s, dirn)  # the point of the tube's axis nearest the gripper
                loc = Rb.T @ (near - pbq[0][0])
                for jn in ("Gripper_Jaw_01_v1_1", "Gripper_Jaw_02_v1_1"):
                    jb = robot_b.find_bodies(jn)[0][0]
                    jp = robot_b.data.body_pose_w.torch[0, jb]
                    rel_p = Rb.T @ (jp[0:3] - pbq[0][0])
                    rel_R = Rb.T @ matrix_from_quat(jp[3:7].unsqueeze(0))[0]
                    print(f"[JAWCHK] {kind:5s} {jn}: link origin in the gripper's frame {r3(rel_p)} | its x axis there {r3(rel_R[:, 0])} "
                          f"y axis {r3(rel_R[:, 1])}", flush=True)
                s_grip = float(torch.dot(pbq[0][0] - tip_a, dirn))  # how far along the scale (from the ring) the gripper is
                print(f"[GRIPCHK]       gripper is {s_grip*1000:5.1f} mm along the scale (tube: {ROD*1000:.0f} to {(ROD+TUBE_LEN)*1000:.0f} mm) | "
                      "tube direction in the gripper's frame:", r3(Rb.T @ dirn), flush=True)
                jq = robot_b.data.joint_pos.torch[0, B.grip_ids]
                print(f"[GRIPCHK] {kind:5s} tube axis in the gripper's frame [mm]: across {float(loc[0])*1000:6.1f}, along the jaws "
                      f"{float(loc[1])*1000:6.1f}, through {float(loc[2])*1000:6.1f} | jaws at {[round(float(v)*1000, 1) for v in jq]} mm "
                      f"(open 0, shut -25.5 / +25.5)", flush=True)
            if kind == "down":
                grab_hold = b_tip_target.clone()  # where the jaws stopped: they close here
            last_kind = kind
            # forces: on the tube (from the finger or from the move arm's hold) and, in return, on the move arm.
            # The pull arm gets no force from the script any more: it feels the scale through the swivel joint.
            scale.permanent_wrench_composer.set_forces_and_torques_index(
                forces=on_scale.reshape(1, 1, 3), positions=at_point.reshape(1, 1, 3), body_ids=[0], is_global=True
            )
            robot_b.permanent_wrench_composer.set_forces_and_torques_index(
                forces=on_b.reshape(1, 1, 3), positions=B.tip().reshape(1, 1, 3), body_ids=[B.ee], is_global=True
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
            # Three red dots on top of the tube, like the stickers on the real scale: one fixed near each end, and the
            # indicator between them, running from 0 (by the purple band) to 600 gf (by the swivel).
            window = TUBE_LEN - 0.060
            top = up * 0.0078
            mark_at = lambda f: along(tube_end - 0.036 - window * min(1.0, f / FULL_SCALE)) + top
            qd = quat_z_to(dirn)
            trans = torch.cat([tips, bones_flat, bones_up, palm, rod, bail,
                               torch.tensor([[0.0, 0.0, -5.0]], device=dev), along(tube_end - 0.014).unsqueeze(0),
                               along(tube_end + CAP_LEN / 2).unsqueeze(0), mark_at(force).unsqueeze(0),
                               (mark_at(args_cli.cap) - top).unsqueeze(0),
                               (along(tube_end - 0.026) + top).unsqueeze(0), (along(tube_start + 0.008) + top).unsqueeze(0)])
            orient = ident.repeat(trans.shape[0], 1)
            orient[nf : 2 * nf] = quat_z_to(u)
            orient[3 * nf] = quat_from_matrix(torch.stack([u, h, up], dim=1).unsqueeze(0))[0]
            orient[-7:-2] = qd
            rig.visualize(translations=trans, orientations=orient, marker_indices=marker_idx)

            rows.append((round(t, 4), kind, round(force, 4), round(float(torch.dot(A.pose()[0][0] - pa_goal, u)), 5)))
            if step % FRAME_EVERY == 0 and args_cli.hero and not args_cli.no_record:
                Image.fromarray(cam_side.data.output["rgb"][0, ..., :3].cpu().numpy()).save(f"{OUT}/frame_{frames:04d}.png")
                frames += 1
            elif step % FRAME_EVERY == 0 and not args_cli.no_record:
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
