"""Settings of the two-arm task (Flex-Duo-v0): both followers on one wooden table, each with its own policy."""

from __future__ import annotations

import math
import os

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace

from .pull_env_cfg import ELROBOT, SEED as PULL_SEED

# Measured on the real bench (16 November 2026): the two bases are 28 cm apart, and the move arm's gripper works
# about 7 cm from the pull arm's gripper, at the top of the scale. The move arm faces the scale's top from the side.
MOVE_BASE, MOVE_YAW = (0.20, 0.20, 0.0), 0.86  # 0.283 m from the pull arm's base at the origin
# Asleep: upright column, forearm level, turned away from the scale and the pull arm (motor 1 at -1.45 rad), jaws
# open. With motor 1 at zero the forearm would hang over the scale and hide it from the overhead camera. The wrist
# roll (motor 7) is preset so that the jaws close across the tube when the arm reaches it; the policy does not move it.
SLEEP = {"rev_motor_01": -1.45, "rev_motor_0[2-6]": 0.0, "rev_motor_07": -0.96, "rev_motor_08": 0.2,
         "rev_motor_08_1": -0.0025, "rev_motor_08_2": 0.0025}
_wood = lambda c, r=0.8: sim_utils.PreviewSurfaceCfg(diffuse_color=c, roughness=r)
_still = AssetBaseCfg.InitialStateCfg


@configclass
class DuoSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(), init_state=_still(pos=(0.0, 0.0, -0.74)))
    # a plain wooden table, top surface at z = 0, with a sheet of cardboard under the hand as on the real bench
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table", init_state=_still(pos=(0.0, 0.38, -0.015)),
        spawn=sim_utils.CuboidCfg(size=(1.05, 1.20, 0.03), visual_material=_wood((0.72, 0.56, 0.36)),
                                  collision_props=sim_utils.UsdPhysicsCollisionCfg()),
    )
    leg0 = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Leg0", init_state=_still(pos=(-0.45, -0.15, -0.385)),
                        spawn=sim_utils.CuboidCfg(size=(0.05, 0.05, 0.71), visual_material=_wood((0.55, 0.42, 0.27))))
    leg1 = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Leg1", init_state=_still(pos=(0.45, -0.15, -0.385)),
                        spawn=sim_utils.CuboidCfg(size=(0.05, 0.05, 0.71), visual_material=_wood((0.55, 0.42, 0.27))))
    leg2 = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Leg2", init_state=_still(pos=(-0.45, 0.91, -0.385)),
                        spawn=sim_utils.CuboidCfg(size=(0.05, 0.05, 0.71), visual_material=_wood((0.55, 0.42, 0.27))))
    leg3 = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Leg3", init_state=_still(pos=(0.45, 0.91, -0.385)),
                        spawn=sim_utils.CuboidCfg(size=(0.05, 0.05, 0.71), visual_material=_wood((0.55, 0.42, 0.27))))
    card = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Cardboard", init_state=_still(pos=(-0.02, 0.45, 0.0015)),
                        spawn=sim_utils.CuboidCfg(size=(0.62, 0.56, 0.003), visual_material=_wood((0.66, 0.52, 0.37), 0.95)))
    pull_arm: ArticulationCfg = replace(ELROBOT, prim_path="{ENV_REGEX_NS}/PullArm")
    move_arm: ArticulationCfg = replace(
        ELROBOT, prim_path="{ENV_REGEX_NS}/MoveArm",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=MOVE_BASE, rot=(0.0, 0.0, math.sin(MOVE_YAW / 2), math.cos(MOVE_YAW / 2)), joint_pos=SLEEP),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1300.0, color=(0.95, 0.96, 1.0)))


@configclass
class DuoEnvCfg(DirectRLEnvCfg):
    """One episode: the finger pulls (the pull arm limits the force), the pull ends, the move arm wakes up and moves
    the scale's loop from that finger to the other one and goes back to rest, then the other finger pulls."""

    decimation = 6  # 60 Hz physics, 10 decisions a second (Station's pace) for both arms
    episode_length_s = 24.0
    action_space = 6  # the MOVE arm learns here: motors 1, 2, 3, 4, 6 and open/close
    observation_space = 51
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation, physics=PhysxCfg())
    scene: DuoSceneCfg = DuoSceneCfg(num_envs=4096, env_spacing=2.4)

    # the pull arm's own policy, trained separately on the same task (Flex-PullBench-v0) and run here as it is
    pull_policy = os.path.expanduser("~/flexnsense-sim/stage/isaac-sim/policies/pull_bench_v1/policy.pt")
    pull_action_scale = 0.06
    cap = 1.96  # N
    tip = 0.10  # m: pull arm's gripper origin -> the point between its jaws
    spring, spring_range = 58.86, (0.7, 1.3)
    rest = 0.225  # m: ring to loop at zero force (the datasheet's length without load)
    pull_range, ramp_range, hold_range, delay_range = (0.03, 0.10), (0.8, 2.0), (1.0, 2.0), (0.3, 1.0)
    force_noise, force_step, force_dropout = 0.10, 0.049, 0.03
    move_time = 10.0  # s the move arm has between the two pulls
    finger_gap = 0.024  # m between index and middle finger

    # the move arm
    action_scale = 0.03  # rad per decision: at most 0.3 rad/s. Slow on purpose: it works 7 cm from the other gripper,
    # and so near the ring a 1 mm move of the jaws moves the loop 3 mm
    grip_rate = 0.34  # of the gripper's travel per decision at full action
    grip_y = 0.082  # m: gripper origin -> middle of the fingers
    grasp_at = 0.07  # m from the ring: where it takes the tube (measured: 7 cm from the pull arm's gripper)
    take_within, take_tilt = 0.015, 0.7
    place_within, place_within_start, place_anneal_steps = 0.008, 0.016, 30000
    clear_height = 0.02
    keep_away = 0.07  # m: during a pull the move arm's fingers stay at least this far from the scale
    # rewards (per second unless "once")
    rew_sleep, rew_reach, rew_carry, rew_done = 1.5, 1.0, 2.0, 5.0
    rew_take_once, rew_place_once, rew_drop_once, rew_snag_once, rew_late_once = 4.0, 15.0, -2.0, -4.0, -3.0
    rew_intrude, rew_aim, rew_rate, rew_table = -3.0, -0.5, -0.02, -3.0
