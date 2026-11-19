import struct

import pytest

from arm1.lib.station import (
    _CURRENT_ADDR,
    _POSITION_ADDR,
    _TEMPERATURE_ADDR,
    _VOLTAGE_ADDR,
    build_freeze_arcs,
    parse_motor_state,
)


def _blob(position=None, current=None, temperature=None, voltage_tenths=None, length=0x47):
    """A synthetic state blob with fields placed at the real ST3215 offsets."""
    b = bytearray(length)
    if position is not None:
        b[_POSITION_ADDR:_POSITION_ADDR + 2] = struct.pack("<H", position)
    if current is not None:
        b[_CURRENT_ADDR:_CURRENT_ADDR + 2] = struct.pack("<H", current)
    if temperature is not None:
        b[_TEMPERATURE_ADDR] = temperature
    if voltage_tenths is not None:
        b[_VOLTAGE_ADDR] = voltage_tenths
    return bytes(b)


def test_parses_all_fields_at_their_real_offsets():
    state = parse_motor_state(_blob(position=2048, current=120, temperature=27, voltage_tenths=74))
    assert state.position == 2048
    assert state.current_ma == 120
    assert state.temperature_c == 27
    assert state.voltage_v == 7.4


def test_negative_position_sign_bit():
    """STS3215 flags direction in bit 15; parse_motor_state must unwrap it, not pass it through."""
    raw = 0x8000 | 100  # sign bit set, magnitude 100
    state = parse_motor_state(_blob(position=raw))
    assert state.position == 4095 + 1 - 100


def test_zero_position_round_trips():
    state = parse_motor_state(_blob(position=0))
    assert state.position == 0


def test_missing_fields_are_none_not_zero():
    """A truncated blob must report a field as missing, never as a false zero reading."""
    state = parse_motor_state(bytes(0x38))  # ends right before the position field starts
    assert state.position is None
    assert state.current_ma is None
    assert state.temperature_c is None
    assert state.voltage_v is None


def test_range_passed_through_unchanged():
    state = parse_motor_state(_blob(position=10), range_min=133, range_max=3963)
    assert state.range_min == 133
    assert state.range_max == 3963


def test_full_sized_blob_with_no_fields_set_reads_as_zero_not_none():
    """A present-but-empty blob (all zero bytes) is a real reading of zero, distinct from a
    too-short blob (tested above), which is a missing reading."""
    state = parse_motor_state(bytes(0x47))
    assert state.position == 0
    assert state.current_ma == 0
    assert state.temperature_c == 0
    assert state.voltage_v == 0.0


def test_build_freeze_arcs_shape():
    arcs = build_freeze_arcs({1: (1085, 3011), 8: (1242, 2854)})
    assert set(arcs) == {(1, 1085, 3011, (1085 + 3011) // 2), (8, 1242, 2854, (1242 + 2854) // 2)}


def test_build_freeze_arcs_midpoint_is_mean_of_range():
    (motor_id, lo, hi, midpoint), = build_freeze_arcs({5: (139, 3957)})
    assert motor_id == 5 and lo == 139 and hi == 3957
    assert midpoint == (139 + 3957) // 2


def test_build_freeze_arcs_empty_input():
    assert build_freeze_arcs({}) == []


def test_build_freeze_arcs_preserves_all_motors():
    ranges = {i: (i * 100, i * 100 + 50) for i in range(1, 9)}
    arcs = build_freeze_arcs(ranges)
    assert len(arcs) == 8
    assert {a[0] for a in arcs} == set(range(1, 9))


def test_register_addresses_match_normacore_examples():
    """RAM_TORQUE_ENABLE, RAM_GOAL_POSITION, RAM_ACC and RAM_GOAL_SPEED are copied from
    NormaCore's own examples (st3215-remote-teleop-py/state.py, so101-autocalibration-py/
    st3215.py) - this guards against an accidental edit drifting from the real register map,
    which would silently write the wrong thing to a servo."""
    from arm1.lib.station import RAM_ACC, RAM_GOAL_POSITION, RAM_GOAL_SPEED, RAM_TORQUE_ENABLE

    assert RAM_TORQUE_ENABLE == 0x28
    assert RAM_ACC == 0x29
    assert RAM_GOAL_POSITION == 0x2A
    assert RAM_GOAL_SPEED == 0x2E


# ---------------------------------------------------------------- per-joint speed scaling
def test_the_joint_travelling_furthest_gets_the_requested_speed():
    from arm1.lib.station import scaled_speeds

    speeds = scaled_speeds({1: -2.0, 2: 10.0, 4: -6.0}, base_speed=400)
    assert speeds[2] == 400


def test_a_joint_travelling_less_gets_proportionally_less_speed():
    """The whole point: motor 4 here travels 60% of motor 2's distance, so it should arrive
    at about the same time by moving at about 60% of the speed, not the same speed motor 2
    gets (which is what made a coordinated pull wobble on the bench even after sync_write)."""
    from arm1.lib.station import scaled_speeds

    speeds = scaled_speeds({2: 10.0, 4: -6.0}, base_speed=400)
    assert speeds[4] == pytest.approx(240, abs=1)


def test_speed_is_floored_not_zero_for_a_tiny_delta():
    """0 is ambiguous on this servo family (some firmware reads it as 'no limit' rather than
    'stop'), so a near-stationary joint still gets a safe, small, nonzero speed rather than
    whatever 0 happens to mean on this particular servo."""
    from arm1.lib.station import scaled_speeds

    speeds = scaled_speeds({2: 10.0, 7: 0.001}, base_speed=400, min_speed=30)
    assert speeds[7] == 30


def test_every_motor_at_zero_delta_gets_the_floor():
    """No motion this step at all: nothing to scale against, so every joint just gets the
    floor rather than a division by zero."""
    from arm1.lib.station import scaled_speeds

    speeds = scaled_speeds({1: 0.0, 2: 0.0, 4: 0.0}, base_speed=400, min_speed=30)
    assert speeds == {1: 30, 2: 30, 4: 30}


def test_the_sign_of_the_delta_does_not_affect_the_speed():
    """Speed is a magnitude on this register; direction is carried entirely by the goal
    position, not by the speed value."""
    from arm1.lib.station import scaled_speeds

    speeds = scaled_speeds({1: -10.0, 2: 10.0}, base_speed=400)
    assert speeds[1] == speeds[2]
