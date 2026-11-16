"""The two-machine protocol: the lease, and refusing anything whose meaning is uncertain.

The group that matters most is "the lease". Everything else here is encode/decode hygiene; the
lease is the mechanism that stops a remote arm moving a patient's finger after the machine that
was supposed to say "stop" has gone quiet.
"""
from __future__ import annotations

import pytest

from session.arm2 import HookResult
from session.wire import (
    SCHEMA_VERSION,
    Lease,
    MessageKind,
    WireError,
    decode,
    encode_abort,
    encode_heartbeat,
    encode_hook_report,
    encode_hook_request,
    outcome_from_report,
)


def lease(move_id=1, issued_at=100.0, period_s=0.5) -> Lease:
    return Lease(move_id=move_id, issued_at=issued_at, period_s=period_s)


# --------------------------------------------------------------------------------- the lease
def test_a_fresh_lease_authorises_motion():
    assert lease(issued_at=100.0, period_s=0.5).fresh_at(100.2)


def test_an_expired_lease_does_not():
    assert not lease(issued_at=100.0, period_s=0.5).fresh_at(100.6)


def test_a_lease_is_dead_at_its_deadline_not_after_it():
    """One cycle of doubt resolves toward stopping, which is the same way
    `control.safety.SafetyGate` resolves a reading it is unsure about."""
    assert not lease(issued_at=100.0, period_s=0.5).fresh_at(100.5)


def test_silence_expires_a_lease_rather_than_extending_it():
    """The whole reason this is a lease and not a flag. A flag set by 'you may move' stays set
    when the link dies, and the far arm keeps going with nobody able to stop it. A lease has to
    be renewed to stay alive, so silence means stop."""
    held = lease(issued_at=100.0, period_s=0.5)
    # No renewal arrives. Time passes. Permission evaporates on its own.
    assert held.fresh_at(100.4)
    assert not held.fresh_at(101.0)
    assert not held.fresh_at(500.0)


def test_renewing_builds_a_new_lease_rather_than_mutating_the_old_one():
    """There must be no code path that extends a lease which has already expired: whoever
    renews has to decide, with the clock in hand, that renewing is still right."""
    original = lease(issued_at=100.0, period_s=0.5)
    renewed = original.renewed(100.3)
    assert original.expires_at == 100.5, "the original must be untouched"
    assert renewed.expires_at == 100.8
    assert renewed.move_id == original.move_id


def test_a_renewal_keeps_the_move_it_belongs_to():
    """So a renewal cannot revive a move that already finished."""
    assert lease(move_id=7).renewed(200.0).move_id == 7


def test_a_lease_valid_for_no_time_is_refused():
    with pytest.raises(ValueError, match="period_s must be positive"):
        Lease(move_id=1, issued_at=0.0, period_s=0.0)
    with pytest.raises(ValueError, match="period_s must be positive"):
        Lease(move_id=1, issued_at=0.0, period_s=-1.0)


def test_a_negative_move_id_is_refused():
    with pytest.raises(ValueError, match="move_id cannot be negative"):
        Lease(move_id=-1, issued_at=0.0, period_s=0.5)


def test_the_lease_period_travels_with_the_lease():
    """So the actuator enforces the brain's chosen period without being configured to match
    it, and the brain can tighten the rein on one stroke without redeploying anything."""
    kind, fields = decode(encode_hook_request("index", lease(period_s=0.25)))
    assert kind is MessageKind.HOOK_REQUEST
    assert fields["lease"].period_s == 0.25


# ------------------------------------------------------------------------- round trips
def test_a_hook_request_round_trips():
    kind, fields = decode(encode_hook_request("index", lease(move_id=3)))
    assert kind is MessageKind.HOOK_REQUEST
    assert fields["finger"] == "index"
    assert fields["lease"].move_id == 3


def test_a_hook_report_round_trips():
    kind, fields = decode(
        encode_hook_report(3, HookResult.CONFIRMED, "middle", "confirmed by operator")
    )
    assert kind is MessageKind.HOOK_REPORT
    assert fields["move_id"] == 3
    assert fields["result"] is HookResult.CONFIRMED
    assert fields["detail"] == "confirmed by operator"


def test_a_heartbeat_round_trips():
    kind, fields = decode(encode_heartbeat(lease(move_id=4, issued_at=50.0)))
    assert kind is MessageKind.HEARTBEAT
    assert fields["lease"].move_id == 4
    assert fields["lease"].issued_at == 50.0


def test_an_abort_round_trips():
    kind, fields = decode(encode_abort(5, "operator stopped the session"))
    assert kind is MessageKind.ABORT
    assert fields["move_id"] == 5
    assert fields["reason"] == "operator stopped the session"


def test_a_report_becomes_the_outcome_the_orchestrator_already_understands():
    """Nothing above the wire should need to know a network was involved."""
    _, fields = decode(encode_hook_report(1, HookResult.FAILED, "ring", "hook slipped"))
    outcome = outcome_from_report(fields)
    assert outcome.result is HookResult.FAILED
    assert outcome.finger == "ring"
    assert not outcome.confirmed


def test_finger_names_are_normalised_on_the_wire():
    """So a brain and an actuator cannot disagree about whether 'pinky' and 'little' are the
    same finger: there is one vocabulary and it is session.fingers."""
    _, fields = decode(encode_hook_request("pinky", lease()))
    assert fields["finger"] == "little"


# ------------------------------------------------------------- what it refuses, and why
def test_a_wrong_schema_version_is_refused():
    """Two machines on different protocol versions is a configuration error. It must surface at
    the handshake, not as a hook move that silently means something else."""
    message = encode_hook_request("index", lease())
    message["schema"] = SCHEMA_VERSION + 1
    with pytest.raises(WireError, match="schema"):
        decode(message)


def test_a_missing_schema_version_is_refused():
    message = encode_hook_request("index", lease())
    del message["schema"]
    with pytest.raises(WireError, match="schema"):
        decode(message)


def test_an_unknown_message_kind_is_refused():
    """It could be a newer protocol asking for something this end does not implement."""
    with pytest.raises(WireError, match="unknown message kind"):
        decode({"schema": SCHEMA_VERSION, "kind": "grant_custody"})


def test_a_non_object_message_is_refused():
    with pytest.raises(WireError, match="must be an object"):
        decode([1, 2, 3])


def test_a_hook_request_without_a_lease_is_refused():
    """An unauthorised move request is not a move request. Tolerating a missing lease would
    make the lease optional, and an optional lease protects nobody."""
    message = encode_hook_request("index", lease())
    del message["lease"]
    with pytest.raises(WireError, match="missing"):
        decode(message)


def test_a_lease_with_missing_fields_is_refused():
    message = encode_hook_request("index", lease())
    del message["lease"]["period_s"]
    with pytest.raises(WireError, match="lease is missing"):
        decode(message)


def test_a_lease_with_unusable_fields_is_refused():
    message = encode_hook_request("index", lease())
    message["lease"]["issued_at"] = "soon"
    with pytest.raises(WireError, match="unusable"):
        decode(message)


def test_a_report_about_an_unidentifiable_move_is_refused():
    """A report that cannot be tied to a request is the mistake that would have arm 1 pull on a
    finger the hook has left."""
    message = encode_hook_report(1, HookResult.CONFIRMED, "index")
    del message["move_id"]
    with pytest.raises(WireError, match="missing"):
        decode(message)


def test_a_report_with_an_unknown_result_is_refused():
    message = encode_hook_report(1, HookResult.CONFIRMED, "index")
    message["result"] = "probably_fine"
    with pytest.raises(WireError, match="unknown hook result"):
        decode(message)


def test_an_unknown_finger_is_refused():
    message = encode_hook_request("index", lease())
    message["finger"] = "thumbnail"
    with pytest.raises(WireError, match="unknown finger"):
        decode(message)


def test_a_non_integer_move_id_is_refused():
    message = encode_hook_report(1, HookResult.CONFIRMED, "index")
    message["move_id"] = "one"
    with pytest.raises(WireError, match="move_id must be an integer"):
        decode(message)


# --------------------------------------------------------- what is deliberately absent
def test_there_is_no_message_that_grants_custody_outright():
    """Custody travels as a lease on every request, so there is nothing to hand over once and
    then forget about. A 'you have custody' message would be the flag this design rejects."""
    kinds = {k.value for k in MessageKind}
    assert not any("custody" in k or "grant" in k for k in kinds), kinds


def test_no_message_carries_a_force_reading():
    """The force never leaves the machine that measures it: that is what keeps every safety
    decision off the network (docs/two-machine-session.md)."""
    messages = [
        encode_hook_request("index", lease()),
        encode_hook_report(1, HookResult.CONFIRMED, "index"),
        encode_heartbeat(lease()),
        encode_abort(1, "stop"),
    ]
    for message in messages:
        flat = str(message)
        assert "force" not in flat and "newton" not in flat.lower(), message
