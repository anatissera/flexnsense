import struct

from lib.station import (
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
    """RAM_TORQUE_ENABLE and RAM_GOAL_POSITION are copied from NormaCore's own examples
    (st3215-remote-teleop-py/state.py) - this guards against an accidental edit drifting
    from the real register map, which would silently write the wrong thing to a servo."""
    from lib.station import RAM_GOAL_POSITION, RAM_TORQUE_ENABLE

    assert RAM_TORQUE_ENABLE == 0x28
    assert RAM_GOAL_POSITION == 0x2A
