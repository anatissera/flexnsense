"""Settings of the physical-contact task (Flex-Phys-v0).

Nothing here is moved by a rule. The scale is a chain of real bodies on the pull arm's gripper (swivel, tube, sliding
rod on the datasheet's spring, wire loop), the fingers are solid posts, and the move arm has to pinch the tube between
its jaw pads and carry it by friction. The loop is on a finger only when the ring really is around the post.

The hand lies palm up, fingers bent upwards, with the thumb on the right (seen from the pull arm) pointing at the move
arm: the index finger is the post nearest the thumb, the middle finger the next one to its left.
"""

from __future__ import annotations

import math
import os

from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .duo_env_cfg import MOVE_BASE, MOVE_YAW, SLEEP, DuoSceneCfg
from .pull_env_cfg import SEED as PULL_SEED

_A = os.path.expanduser("~/flexnsense-sim/assets/elrobot")
PULL_USD = f"{_A}/usd_pull_scale/elrobot_follower.usd/elrobot_pull_scale/elrobot_pull_scale_fixed.usda"
MOVE_USD = f"{_A}/usd_move_pads/elrobot_follower.usd/elrobot_move_pads/elrobot_move_pads_fixed.usda"
# the pull arm's posture with its gripper 6 cm further back (tools/seed_pose.py), motors 1..7; the pull goes towards it
PULL_BACK = [-0.0039, -0.4145, 1.2172, 0.5527, 0.0, -1.3554, 0.0]
SWIVEL = (0.0, 0.30, 0.05)  # m: where the pull arm's jaws hold the scale's swivel in its start posture
LOOP_AT, GRASP_AT = 0.225, 0.07  # m from the swivel: centre of the wire loop; where the move arm takes the tube
FINGER_R, FINGER_TOP = 0.008, 0.054  # m: a finger post's radius and the height of its tip
ARC = 0.221  # m from the swivel to the finger posts
# rad round the swivel, 4 cm apart; positive is to the right (towards the thumb and the move arm)
FINGER_ANGLE = {"index": 0.09, "middle": -0.09, "ring": -0.27, "little": -0.45}
finger_xy = lambda a: (SWIVEL[0] + ARC * math.sin(a), SWIVEL[1] + ARC * math.cos(a))

_props = lambda: [PhysxArticulationCfg(enabled_self_collisions=False, solver_position_iteration_count=16,
                                       solver_velocity_iteration_count=2)]
_servo = lambda: {
    "servos": ImplicitActuatorCfg(joint_names_expr=["rev_motor_0[1-7]"], joint_effort_limit=2.94,
                                  joint_velocity_limit=4.71, stiffness=40.0, damping=2.0),
    "gear": ImplicitActuatorCfg(joint_names_expr=["rev_motor_08"], joint_effort_limit=2.94, joint_velocity_limit=4.71,
                                stiffness=0.0, damping=0.5),
    "jaws": ImplicitActuatorCfg(joint_names_expr=["rev_motor_08_[12]"], joint_effort_limit=6.0,
                                joint_velocity_limit=0.2, stiffness=400.0, damping=20.0, armature=0.3),
}
PULL_ARM = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/PullArm",
    spawn=sim_utils.UsdFileCfg(usd_path=PULL_USD, fix_root_link=True, articulation_props=_props()),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), joint_pos={**PULL_SEED, "scale_yaw": -0.4, "scale_pitch": -0.2, "scale_slide": 0.0}),
    actuators={
        **_servo(),
        # the swivel turns freely; the sliding rod is on the scale's own spring (600 g over 100 mm)
        "swivel": ImplicitActuatorCfg(joint_names_expr=["scale_yaw", "scale_pitch"], joint_effort_limit=1.0,
                                      joint_velocity_limit=30.0, stiffness=0.0, damping=0.004),
        "spring": ImplicitActuatorCfg(joint_names_expr=["scale_slide"], joint_effort_limit=10.0, joint_velocity_limit=3.0,
                                      stiffness=58.86, damping=1.5, armature=0.02),
    },
)
MOVE_ARM = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/MoveArm",
    spawn=sim_utils.UsdFileCfg(usd_path=MOVE_USD, fix_root_link=True, articulation_props=_props()),
    init_state=ArticulationCfg.InitialStateCfg(pos=MOVE_BASE, rot=(0.0, 0.0, math.sin(MOVE_YAW / 2), math.cos(MOVE_YAW / 2)),
                                               joint_pos=SLEEP),
    actuators=_servo(),
)
_skin = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.93, 0.72, 0.60), roughness=0.7)


def _finger(name):
    x, y = finger_xy(FINGER_ANGLE[name])
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Finger_" + name, init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, FINGER_TOP / 2)),
        spawn=sim_utils.CapsuleCfg(radius=FINGER_R, height=FINGER_TOP - 2 * FINGER_R, axis="Z", visual_material=_skin,
                                   collision_props=sim_utils.UsdPhysicsCollisionCfg()))


@configclass
class PhysSceneCfg(DuoSceneCfg):
    pull_arm: ArticulationCfg = PULL_ARM
    move_arm: ArticulationCfg = MOVE_ARM
    finger_index = _finger("index")
    finger_middle = _finger("middle")
    finger_ring = _finger("ring")
    finger_little = _finger("little")
    # drawn only: the palm (facing up) beyond the posts, and the thumb on the right, pointing at the move arm
    palm = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Palm", init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.035, 0.585, 0.011)),
                        spawn=sim_utils.CuboidCfg(size=(0.15, 0.10, 0.022), visual_material=_skin))
    thumb = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Thumb", init_state=AssetBaseCfg.InitialStateCfg(pos=(0.075, 0.60, 0.011)),
                         spawn=sim_utils.CuboidCfg(size=(0.07, 0.022, 0.02), visual_material=_skin))


@configclass
class PhysEnvCfg(DirectRLEnvCfg):
    """One episode: the loop lies on the table; the move arm wakes from rest, pinches the tube, puts the loop on the
    index finger and lets go; the pull arm pulls back until the scale reads 200 g and returns to zero; the move arm
    moves the loop to the middle finger; the pull arm pulls again; the move arm is back at rest."""

    decimation = 12  # 120 Hz physics, 10 decisions a second
    episode_length_s = 40.0
    action_space = 6  # the move arm: motors 1, 2, 3, 4, 6 and open/close
    observation_space = 52
    state_space = 0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation, physics=PhysxCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9))
    scene: PhysSceneCfg = PhysSceneCfg(num_envs=4096, env_spacing=2.4)

    spring = 58.86  # N/m
    pull_to = 1.96  # N (200 gf): the pull arm pulls until the scale shows this, then goes back to zero
    pull_rate, pull_hold = 0.22, 1.0  # share of the 6 cm pull-back per second; seconds held at the top
    start_on_index_share = 0.4  # episodes that start with the loop already on the index finger (the second half only)
    table_spot = (0.20, 0.42)  # rad to the right of the index finger: where the loop lies at the start
    stage_time = 14.0  # s the move arm has for each of its two jobs
    action_scale, max_lead = 0.03, 0.15
    grip_rate = 0.34
    grip_y = 0.094  # m: gripper origin -> the front part of the jaw pads (their middle, at 82 mm, cannot get down
    # to a tube lying low: the fingertips touch the table first)
    max_tip_speed = 0.25
    keep_away = 0.06
    # rewards per second unless "once"
    rew_stage = 5.0  # for every stage already finished: being further is always worth more than lingering
    rew_reach, rew_carry, rew_set, rew_sleep = 0.8, 2.0, 1.0, 1.5
    rew_place_once, rew_all_done_once, rew_late_once = 10.0, 5.0, -3.0
    rew_intrude, rew_rate, rew_speed, rew_table, rew_yank = -3.0, -0.02, -40.0, -3.0, -1.0
