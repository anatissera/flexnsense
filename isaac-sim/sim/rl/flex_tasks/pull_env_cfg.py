"""Settings of the pull-arm task (Flex-Pull-v0)."""

from __future__ import annotations

import os

from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

USD = os.path.expanduser("~/flexnsense-sim/assets/elrobot/usd/elrobot_follower.usd/elrobot_follower/elrobot_follower.usda")
# the start posture from tools/seed_pose.py: gripper low and level, jaws closed on the scale's ring
SEED = {"rev_motor_01": 0.0, "rev_motor_02": 0.04, "rev_motor_03": 1.057, "rev_motor_04": 0.072, "rev_motor_05": 0.0,
        "rev_motor_06": -1.169, "rev_motor_07": 0.0, "rev_motor_08": 1.98, "rev_motor_08_1": -0.0229, "rev_motor_08_2": 0.0229}

ELROBOT = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD, fix_root_link=True, articulation_props=[PhysxArticulationCfg(enabled_self_collisions=False)]
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos=SEED),
    actuators={
        # ASSUMED servo stiffness and damping (not measured on the ST3215); effort and speed limits from the URDF
        "servos": ImplicitActuatorCfg(joint_names_expr=["rev_motor_0[1-7]"], joint_effort_limit=2.94,
                                      joint_velocity_limit=4.71, stiffness=40.0, damping=2.0),
        # the gear motor is left free and the two jaws are driven directly (see sim/dbg_gripper.py: driving the gear
        # in this converted model turns the wrist)
        "gear": ImplicitActuatorCfg(joint_names_expr=["rev_motor_08"], joint_effort_limit=2.94,
                                    joint_velocity_limit=4.71, stiffness=0.0, damping=0.5),
        "jaws": ImplicitActuatorCfg(joint_names_expr=["rev_motor_08_[12]"], joint_effort_limit=6.0,
                                    joint_velocity_limit=0.2, stiffness=400.0, damping=20.0, armature=0.3),
    },
)


@configclass
class PullSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())  # the table top, z = 0
    robot: ArticulationCfg = ELROBOT
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1500.0))


@configclass
class PullEnvCfg(DirectRLEnvCfg):
    decimation = 2  # 120 Hz physics, the policy acts 60 times a second
    episode_length_s = 10.0
    action_space = 5  # motors 1, 2, 3, 4 and 6; the two roll motors (5, 7) and the gripper are held
    observation_space = 29
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=PhysxCfg())
    scene: PullSceneCfg = PullSceneCfg(num_envs=4096, env_spacing=1.2)

    action_scale = 0.012  # rad: the largest change of a joint target in one step
    tip = 0.10  # m: gripper origin -> the point between the jaws that holds the ring
    cap = 1.96  # N (200 gf): the force limit
    # the scale's spring: 5.886 N (600 gf) over an ASSUMED 0.10 m, varied +-30 % per episode
    spring = 58.86
    spring_range = (0.7, 1.3)
    rest = 0.23  # m: ring to finger at zero force
    pull_range = (0.03, 0.10)  # m: how far the finger pulls away
    ramp_range = (0.8, 3.0)  # s: how long the pull takes to build up
    hold_range = (1.0, 3.0)  # s
    delay_range = (0.3, 1.5)  # s before the pull starts
    yaw_range = 0.17  # rad: the pull direction, off the arm's line
    force_noise = 0.06  # N: noise on the force reading (the camera reading of the scale)
    joint_noise = 0.15  # rad: spread of the start posture's joints 2, 3, 4, 6
    # rewards
    rew_over = -6.0  # per N above the limit, plus the same again squared
    rew_hold = -8.0  # per m away from the starting place while the force is low
    rew_side = -8.0  # per m off the scale's line, or up/down
    rew_aim = -1.0  # for pointing away from the scale's line
    rew_rate = -0.02  # per squared change of action
    rew_alive = 1.0
    max_drift = 0.22  # m: further from the start than this ends the episode
    # Joint speed in the observation: the simulator's own value, or the change in angle since the last decision
    # divided by the time between decisions (which is all a real control loop has).
    finite_diff_velocity = False
    # The force reading as the camera gives it: steps of this size [N] (0 = exact) and this share of frames lost
    # (the last reading is then repeated).
    force_step = 0.0
    force_dropout = 0.0
    # Robustness to a real arm that does not behave like the simulated one:
    reward_cap_share = 1.0  # the reward treats this share of the limit as the limit (0.8 = aim 20 % under it)
    stiffness_range = None  # (low, high) N.m/rad: servo stiffness drawn per arm and episode; None = the fixed 40
    damping_range = None  # (low, high) N.m.s/rad
    follow_range = None  # (low, high): share of the way to its new goal a servo is sent each decision (1 = at once)
    bench = False  # True: the scene has the table, the hand and the scale as bodies (PullBenchEnvCfg)


@configclass
class Pull10EnvCfg(PullEnvCfg):
    """The same task at the pace of the team's real control loop: one decision every 100 ms (Station publishes a
    frame 10 times a second), joint speed taken from two consecutive frames, a noisier force reading."""

    decimation = 6  # 60 Hz physics, 10 decisions a second
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation, physics=PhysxCfg())
    action_scale = 0.06  # rad per decision: the same top speed as 0.012 rad at 60 decisions a second (0.6 rad/s)
    force_noise = 0.10  # N
    finite_diff_velocity = True


HAND_USD = os.path.expanduser("~/flexnsense-sim/assets/bench/hand.usda")  # written by sim/10_make_hand_usd.py
_SEEN = sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)  # a body the task places; forces do not move it
_look = lambda c, m=0.0: sim_utils.PreviewSurfaceCfg(diffuse_color=c, metallic=m, roughness=0.5)


@configclass
class PullBenchSceneCfg(InteractiveSceneCfg):
    """The bench: NVIDIA's lab table, the arm clamped to it, the hand, the pulling finger, the scale's tube."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)))
    # the table from NVIDIA's Isaac asset library (the one Isaac Lab's own manipulation tasks use); its top is at z = 0
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.40, 0.0)),
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )
    robot: ArticulationCfg = ELROBOT
    hand: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Hand",
        spawn=sim_utils.UsdFileCfg(usd_path=HAND_USD),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.53, 0.0)),
    )
    finger: RigidObjectCfg = RigidObjectCfg(  # the index finger's upright part: it goes through the scale's loop and pulls
        prim_path="{ENV_REGEX_NS}/PullFinger",
        spawn=sim_utils.CylinderCfg(radius=0.0095, height=0.08, axis="Z", rigid_props=_SEEN,
                                    visual_material=_look((0.86, 0.60, 0.47))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.53, 0.04)),
    )
    tube: RigidObjectCfg = RigidObjectCfg(  # the scale's tube: 165 mm, between the arm's jaws and the finger
        prim_path="{ENV_REGEX_NS}/ScaleTube",
        spawn=sim_utils.CylinderCfg(radius=0.0075, height=0.165, axis="Z", rigid_props=_SEEN,
                                    visual_material=_look((0.78, 0.80, 0.84), 0.6)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.42, 0.05), rot=(-0.70710678, 0.0, 0.0, 0.70710678)),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.95, 0.95, 1.0)))


@configclass
class PullBenchEnvCfg(Pull10EnvCfg):
    """The pull task on the bench as built in sim/08: table, hand, scale, 10 decisions a second, and a force reading
    that behaves like the camera's (steps of 5 gf, some frames lost)."""

    scene: PullBenchSceneCfg = PullBenchSceneCfg(num_envs=4096, env_spacing=2.2)
    bench = True
    force_step = 0.049  # N: 5 gf, about what the camera resolves on the scale
    force_dropout = 0.03
    # The first version trained here kept the force at the limit in the simulator but went over it in the no-robot
    # self-test, whose stand-in motors lag differently. So: aim 20 % under the limit, and do not let the policy
    # count on one exact servo behaviour.
    reward_cap_share = 0.8
    stiffness_range = (15.0, 80.0)
    damping_range = (1.0, 4.0)
    follow_range = (0.35, 1.0)
