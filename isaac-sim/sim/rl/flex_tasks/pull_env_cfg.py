"""Settings of the pull-arm task (Flex-Pull-v0)."""

from __future__ import annotations

import os

from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

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
