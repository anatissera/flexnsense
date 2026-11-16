"""Settings of the joint task (Flex-Team-v0): both arms on one wooden table, both learning at the same time."""

from __future__ import annotations

from isaaclab_physx.physics import PhysxCfg

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .duo_env_cfg import DuoSceneCfg


@configclass
class TeamEnvCfg(DirectMARLEnvCfg):
    """Two agents, two policies, one episode: the finger pulls, the pull arm limits the force; the pull ends, the
    move arm wakes up and moves the scale's loop to the other finger; the other finger pulls."""

    decimation = 6  # 60 Hz physics, 10 decisions a second for both arms
    episode_length_s = 24.0
    possible_agents = ["pull", "move"]
    action_spaces = {"pull": 5, "move": 6}
    observation_spaces = {"pull": 34, "move": 51}
    state_space = -1
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation, physics=PhysxCfg())
    scene: DuoSceneCfg = DuoSceneCfg(num_envs=4096, env_spacing=2.4)

    cap = 1.96  # N: the force limit
    tip = 0.10
    spring, spring_range = 58.86, (0.7, 1.3)
    rest = 0.225  # the datasheet's length without load
    pull_range, ramp_range, hold_range, delay_range = (0.03, 0.10), (0.8, 2.0), (1.0, 2.0), (0.3, 1.0)
    move_time = 10.0
    finger_gap = 0.024
    table_start_share = 0.4  # share of episodes that start with the loop lying on the table

    # The force reading, as the overhead camera reads the scale's three points: the indicator's place between the two
    # fixed points, in whole pixels, and no reading while the move arm's gripper covers the indicator.
    camera_height = 0.50  # m above the table (measured on the bench)
    camera_px_per_m = 1467.0  # 640 px across a 0.44 m wide view at that height
    scale_travel = 0.10  # m of marker travel at full scale (5.886 N): the datasheet's 100 mm scale
    full_scale = 5.886  # N (600 gf)
    force_noise = 0.05  # N
    cover_radius = 0.035  # m: the gripper covers the indicator when it is this close to it, seen from above

    # the pull arm
    pull_action_scale = 0.06
    reward_cap_share = 0.8  # aim 20 % under the limit
    rew_p_alive, rew_p_over, rew_p_hold, rew_p_side, rew_p_aim, rew_p_rate = 1.0, -6.0, -8.0, -8.0, -1.0, -0.02
    # the move arm (as in Flex-Duo-v0)
    action_scale = 0.03  # slow on purpose, see duo_env_cfg.py
    grip_rate = 0.34
    grip_y = 0.082
    grasp_at = 0.07  # measured: 7 cm from the pull arm's gripper
    take_within, take_tilt = 0.015, 0.7
    place_within, place_within_start, place_anneal_steps = 0.008, 0.016, 30000
    clear_height = 0.04  # m: the fingers stand 8 cm tall and the loop sits at 5 cm, so it must rise 3 cm to come off;
    # 4 cm leaves a margin (the first joint run was trained with 2 cm, which a real finger would not allow)
    keep_away = 0.07
    max_tip_speed = 0.25  # m/s: above this the move arm's gripper is penalised (it works next to a hand)
    rew_speed = -20.0  # per m/s above that
    # Robustness to real arms that do not behave like the simulated one, for BOTH arms: a different servo for every
    # table and episode, and a lag in following goals. Without this the pull policy of the first joint run kept the
    # force in the simulator but went far over the limit in the no-robot test of the real-robot runner.
    stiffness_range = (30.0, 70.0)  # N.m/rad
    damping_range = (1.0, 4.0)  # N.m.s/rad
    follow_range = (0.4, 1.0)  # share of the way to its new goal a servo is sent each decision
    # A goal may never run further than this ahead of where the joint is [rad]. Without it both policies learned to
    # park a goal at the joint's limit and let the servo push with constant torque: in the simulator that limits the
    # force by itself, but it is not position control, it made the move arm lunge, and the pull policy trained that way
    # failed the test of the real-robot runner (3.5 N against a 1.96 N limit).
    # (First tried 0.12 and 0.15: still enough for full torque, and the pull arm then braced its gripper on the table
    # with it. 0.06 rad a decision is 0.6 rad/s, which also keeps the move arm's gripper near 0.25 m/s.)
    pull_max_lead = 0.08
    move_max_lead = 0.06
    # holding a goal away from where the joint is means pushing on something: it costs, per radian and second
    rew_p_lead = -6.0
    rew_m_lead = -2.0
    rew_sleep, rew_reach, rew_carry, rew_done = 1.5, 1.0, 2.0, 5.0
    rew_take_once, rew_place_once, rew_drop_once, rew_snag_once, rew_late_once = 4.0, 15.0, -2.0, -4.0, -3.0
    rew_intrude, rew_aim, rew_rate, rew_table = -3.0, -0.5, -0.02, -3.0
    # what both arms share: the whole sequence is one job
    rew_team_place_once = 5.0  # to the pull arm too, when the loop is placed
    rew_team_fail_once = -2.0  # to the pull arm too, when the move fails
