"""Settings of the move-arm task (Flex-Move-v0)."""

from __future__ import annotations

import math

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace

from .pull_env_cfg import ELROBOT

# start posture from tools/seed_pose.py: over the scale near its loop, tilted 30 degrees, jaws open
SEED = {"rev_motor_01": -0.598, "rev_motor_02": -0.069, "rev_motor_03": -0.126, "rev_motor_04": 0.195,
        "rev_motor_05": -0.293, "rev_motor_06": 1.155, "rev_motor_07": -0.905, "rev_motor_08": 0.2,
        "rev_motor_08_1": -0.0025, "rev_motor_08_2": 0.0025}
BASE, YAW = (0.24, 0.40, 0.0), math.pi / 2  # beside the top half of the scale, facing it (as on the bench)


@configclass
class MoveSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())  # the table top, z = 0
    robot: ArticulationCfg = replace(
        ELROBOT,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=BASE, rot=(0.0, 0.0, math.sin(YAW / 2), math.cos(YAW / 2)), joint_pos=SEED
        ),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1500.0))


@configclass
class MoveEnvCfg(DirectRLEnvCfg):
    decimation = 2  # 120 Hz physics, the policy acts 60 times a second
    episode_length_s = 8.0
    action_space = 6  # motors 1, 2, 3, 4, 6 and the gripper (open/close)
    observation_space = 44
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=PhysxCfg())
    scene: MoveSceneCfg = MoveSceneCfg(num_envs=4096, env_spacing=1.4)

    action_scale = 0.02  # rad per step
    grip_rate = 0.12  # how much of the gripper's travel one full action moves per step
    grip_y = 0.082  # m: gripper origin -> middle of the fingers (measured on NormaCore's meshes)
    # the bench, in each environment's own frame (the pull arm's base would be the origin)
    ring = (0.0, 0.30, 0.05)  # where the pull arm's jaws hold the scale's ring
    scale_len = 0.23  # m: ring to loop
    grasp_at = 0.15  # m from the ring: where the move arm takes the tube
    finger_gap = 0.024  # m between neighbouring fingers
    bench_noise = 0.01  # m: the hand is not always in the same place
    # rules
    take_within = 0.015  # m: jaws this close to the grasp point, and closing, take the tube
    take_tilt = 0.7  # rad: and the gripper within this angle of its working tilt
    place_within = 0.008  # m: letting go this close to the target finger places the loop
    place_within_start = 0.012  # m: the tolerance at the start of training; it tightens to place_within
    place_anneal_steps = 24000  # policy steps (of each arm) over which it tightens: 1000 iterations of 24 steps
    clear_height = 0.02  # m: the loop must be this high above its resting height to pass over a finger
    # rewards (per second unless noted)
    rew_reach, rew_carry, rew_done = 1.0, 2.0, 6.0
    rew_take_once, rew_place_once, rew_drop_once = 4.0, 15.0, -2.0
    rew_drag = -240.0  # per second, but it lasts one step (1/60 s): -4 for hitting a finger, which ends the episode
    rew_aim = -0.5
    rew_rate = -0.02
    rew_table = -3.0  # the gripper point below 15 mm
