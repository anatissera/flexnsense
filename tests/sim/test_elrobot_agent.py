"""The tick-to-qpos conversion, checked without ManiSkill.

`sim/elrobot_agent.py` keeps its ManiSkill imports inside `build_agent()` precisely so this
arithmetic can be tested in the repo's normal environment: it is ordinary unit conversion and
should not need a GPU, a simulator or several gigabytes of install to verify.
"""
import numpy as np
import pytest

from lib.kinematics import CENTRE_TICKS, ticks_to_radians
from sim.elrobot_agent import ARM_MOTORS, parse_ticks, qpos_from_ticks


def test_the_centred_pose_is_all_zeros():
    """Tick 2048 is the URDF's zero, so a centred arm is the zero configuration."""
    q = qpos_from_ticks({})
    assert np.allclose(q[:len(ARM_MOTORS)], 0.0)


def test_the_vector_has_one_entry_per_joint_including_both_jaws():
    assert len(qpos_from_ticks({})) == len(ARM_MOTORS) + 2


def test_each_motor_lands_in_its_own_slot_in_urdf_order():
    """A transposition here would put a shoulder angle on a wrist joint, which looks plausible
    in a viewer and is completely wrong."""
    q = qpos_from_ticks({m: CENTRE_TICKS + 100 * m for m in ARM_MOTORS})
    for i, m in enumerate(ARM_MOTORS):
        assert q[i] == pytest.approx(ticks_to_radians(CENTRE_TICKS + 100 * m), abs=1e-6)


def test_a_missing_motor_is_centred_rather_than_refused():
    q = qpos_from_ticks({1: CENTRE_TICKS + 512})
    assert q[0] == pytest.approx(np.pi / 4, abs=1e-6)
    assert np.allclose(q[1:len(ARM_MOTORS)], 0.0)


def test_the_jaws_travel_in_opposite_senses():
    """The URDF gives one jaw a negative limit and the other a positive one, so a shared
    positive opening would drive one of them the wrong way."""
    q = qpos_from_ticks({}, jaw_m=0.02)
    assert q[-2] == pytest.approx(-0.02)
    assert q[-1] == pytest.approx(+0.02)


def test_parse_ticks_reads_the_command_line_form():
    assert parse_ticks("1:1976,2:1719") == {1: 1976.0, 2: 1719.0}


def test_the_module_imports_without_maniskill():
    """The whole reason build_agent() defers its imports: this file must be importable, and
    testable, in an environment that has no simulator at all."""
    import sim.elrobot_agent as mod

    assert hasattr(mod, "qpos_from_ticks")
