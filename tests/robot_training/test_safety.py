"""Goal limits applied before any command reaches the follower arm. No robot needed."""

import numpy as np

from safety import GoalSafety, goal_norm_to_ticks

RANGES = [(1000, 3000)] * 7 + [(1200, 2800)]


def test_goal_norm_maps_onto_the_calibration_range():
    ticks = goal_norm_to_ticks(np.array([0.0, 1.0, 0.5, 0.25, 0, 0, 0, 0.5]), RANGES)
    assert ticks[:4].tolist() == [1000, 3000, 2000, 1500]
    assert ticks[7] == 2000


def test_goal_norm_outside_unit_interval_is_clipped_to_the_range():
    ticks = goal_norm_to_ticks(np.array([-0.5, 1.7, 0, 0, 0, 0, 0, 2.0]), RANGES)
    assert ticks[0] == 1000 and ticks[1] == 3000 and ticks[7] == 2800


def test_first_goal_is_rate_limited_from_the_current_position():
    safety = GoalSafety(max_step=150, gripper_max_step=400, max_delta=600)
    pos = np.full(8, 2000)
    goal, msg = safety.apply(pos + 500, pos)
    assert goal[:7].tolist() == [2150] * 7      # arm: 150 ticks per tick
    assert goal[7] == 2400                      # gripper: 400 ticks per tick
    assert "limitados" in msg


def test_following_goals_are_rate_limited_from_the_previous_goal():
    safety = GoalSafety(max_step=150, max_delta=0)
    pos = np.full(8, 2000)
    safety.apply(pos + 500, pos)
    goal, _ = safety.apply(pos + 500, pos)      # arm did not move, goal keeps advancing
    assert goal[0] == 2300


def test_small_moves_pass_through_untouched():
    safety = GoalSafety()
    pos = np.full(8, 2000)
    target = pos + np.array([10, -20, 30, 0, 5, -5, 1, 100])
    goal, msg = safety.apply(target, pos)
    assert goal.tolist() == target.tolist() and msg == ""


def test_arm_goal_too_far_from_the_real_position_aborts_the_tick():
    safety = GoalSafety(max_step=150, max_delta=600)
    pos = np.full(8, 2000)
    for _ in range(4):                          # goal walks to +600 while the arm is stuck
        goal, _ = safety.apply(pos + 2000, pos)
        assert goal is not None
    before = safety.prev.copy()
    goal, msg = safety.apply(pos + 2000, pos)
    assert goal is None and "ABORT" in msg
    assert safety.prev.tolist() == before.tolist()   # an aborted tick does not advance the goal


def test_gripper_is_exempt_from_the_abort_rule():
    safety = GoalSafety(max_step=150, gripper_max_step=400, max_delta=600)
    pos = np.full(8, 2000)
    target = pos.copy()
    target[7] -= 1000                           # squeezing an object: goal far ahead of position
    for _ in range(3):
        goal, _ = safety.apply(target, pos)
        assert goal is not None
    assert goal[7] == 1000


def test_stalled_gripper_goal_is_held_even_if_the_lead_wobbles_around_the_limit():
    # Seen on the arm: the gripper opens against the ring and stops, with the goal 50 to 75 ticks
    # ahead. The servo trips its overload protection unless the goal is pulled back.
    safety = GoalSafety()
    pos = np.array([2432, 2642, 1000, 2688, 1723, 2565, 3076, 1561])
    leads = []
    for gap in [75, 52, 70, 50, 65, 52, 75, 55, 52, 70, 60, 52, 75, 52, 70]:
        target = pos.copy()
        target[7] = pos[7] - gap
        goal, msg = safety.apply(target, pos, np.zeros(8, dtype=np.int64))
        leads.append(int(pos[7] - goal[7]))
    assert leads[:12] == [75, 52, 70, 50, 65, 52, 75, 55, 52, 70, 60, 52]
    assert leads[12:] == [45, 45, 45] and "motor 8 retenido" in msg


def test_a_held_motor_is_released_when_the_policy_pulls_back():
    safety = GoalSafety()
    pos = np.array([2000] * 7 + [1561])
    for _ in range(14):
        target = pos.copy()
        target[7] = 1490
        safety.apply(target, pos, None)
    target[7] = 1700
    goal, msg = safety.apply(target, pos, None)
    assert goal[7] == 1700 and "retenido" not in msg
