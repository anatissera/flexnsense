"""Arm 2 over a link, both ends, with a fake link and a fake clock.

The groups that carry weight are the last two: "silence is never success" and "the actuator
stops when the brain goes quiet". Everything above them is the happy path, which is worth
pinning but is not what a distributed version of this machine gets wrong.
"""
from __future__ import annotations

from typing import Any

import pytest

from session.arm2 import HookMover, HookResult
from session.remote_arm2 import ActuatorSide, RemoteHookMover
from session.wire import MessageKind, decode


class FakeLink:
    """Two queues and nothing else. Pair two of these to make a loopback."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.inbox: list[dict[str, Any]] = []

    def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    def drain(self) -> list[dict[str, Any]]:
        got, self.inbox = self.inbox, []
        return got

    def kinds(self) -> list[str]:
        return [m["kind"] for m in self.sent]


class Loopback:
    """A brain link and an actuator link wired to each other, delivered on demand.

    Delivery is explicit rather than automatic so a test can hold a message in flight, which is
    how the crossing cases (an abort passing a report) get written.
    """

    def __init__(self) -> None:
        self.brain = FakeLink()
        self.actuator = FakeLink()

    def deliver(self) -> None:
        self.actuator.inbox.extend(self.brain.sent)
        self.brain.sent = []
        self.brain.inbox.extend(self.actuator.sent)
        self.actuator.sent = []


# --------------------------------------------------------------------------- the seam held
def test_the_remote_mover_is_a_hook_mover():
    """The claim the whole design rests on: `session.orchestrator` cannot tell a network is
    involved, so its own tests keep covering the sequencing unchanged."""
    assert isinstance(RemoteHookMover(FakeLink()), HookMover)


def test_requesting_sends_nothing_until_a_poll_with_a_clock():
    """`HookMover.request` has no clock in its signature, and the lease needs one. Staging the
    send until the first poll is what keeps the protocol's signature unchanged."""
    link = FakeLink()
    mover = RemoteHookMover(link)
    mover.request("index")
    assert link.sent == []
    mover.poll(now=100.0)
    assert link.kinds() == ["hook_request"]


def test_the_request_carries_the_lease_that_authorises_it():
    """So the actuator never sees a request it is not yet allowed to act on."""
    link = FakeLink()
    mover = RemoteHookMover(link, lease_period_s=0.4)
    mover.request("index")
    mover.poll(now=100.0)
    kind, fields = decode(link.sent[0])
    assert kind is MessageKind.HOOK_REQUEST
    assert fields["finger"] == "index"
    assert fields["lease"].period_s == 0.4
    assert fields["lease"].issued_at == 100.0


# ------------------------------------------------------------------------- the happy path
def test_a_confirmed_move_round_trips_end_to_end():
    wire = Loopback()
    mover = RemoteHookMover(wire.brain)
    actuator = ActuatorSide(wire.actuator)

    mover.request("middle")
    assert mover.poll(now=100.0) is None
    wire.deliver()

    decision = actuator.step(now=100.0)
    assert decision.move_to == "middle", "the actuator should be authorised to move"
    actuator.report(HookResult.CONFIRMED, "confirmed by operator")
    wire.deliver()

    outcome = mover.poll(now=100.1)
    assert outcome is not None and outcome.confirmed
    assert outcome.finger == "middle"


def test_a_failed_move_round_trips_too():
    wire = Loopback()
    mover = RemoteHookMover(wire.brain)
    actuator = ActuatorSide(wire.actuator)
    mover.request("ring")
    mover.poll(now=100.0)
    wire.deliver()
    actuator.step(now=100.0)
    actuator.report(HookResult.FAILED, "hook would not seat")
    wire.deliver()

    outcome = mover.poll(now=100.1)
    assert outcome is not None and not outcome.confirmed
    assert outcome.detail == "hook would not seat"


def test_an_outcome_is_delivered_once_and_then_cleared():
    """`poll` is a one-shot read, matching `ManualHookMover`, so the orchestrator cannot act on
    the same outcome twice."""
    wire = Loopback()
    mover = RemoteHookMover(wire.brain)
    actuator = ActuatorSide(wire.actuator)
    mover.request("index")
    mover.poll(now=100.0)
    wire.deliver()
    actuator.step(now=100.0)
    actuator.report(HookResult.CONFIRMED)
    wire.deliver()

    assert mover.poll(now=100.1) is not None
    assert mover.poll(now=100.2) is None


def test_the_brain_renews_the_lease_while_a_move_is_in_progress():
    """The far arm stops when these stop, so they have to keep coming."""
    link = FakeLink()
    mover = RemoteHookMover(link)
    mover.request("index")
    mover.poll(now=100.0)
    link.sent.clear()
    mover.poll(now=100.2)
    mover.poll(now=100.4)
    assert link.kinds() == ["heartbeat", "heartbeat"]


# ----------------------------------------------------------- silence is never success
def test_an_unanswered_move_fails_rather_than_hanging():
    """A move nobody answers has to resolve, or the session waits for ever on a machine that
    is not there."""
    mover = RemoteHookMover(FakeLink(), move_timeout_s=10.0)
    mover.request("index")
    mover.poll(now=100.0)
    assert mover.poll(now=105.0) is None, "still inside the deadline"
    outcome = mover.poll(now=111.0)
    assert outcome is not None
    assert outcome.result is HookResult.FAILED


def test_there_is_no_path_from_silence_to_confirmed():
    """The rule `session/arm2.py` states and this has to honour: success is reported, never
    assumed. A hook reported as moved when it has not moved means arm 1 starts pulling on a
    finger the hook is not attached to."""
    mover = RemoteHookMover(FakeLink(), move_timeout_s=1.0)
    mover.request("index")
    mover.poll(now=100.0)
    outcomes = [mover.poll(now=100.0 + i) for i in range(1, 40)]
    reported = [o for o in outcomes if o is not None]
    assert reported, "it must resolve rather than hang"
    assert all(o.result is HookResult.FAILED for o in reported)


def test_a_timeout_says_the_hook_cannot_be_accounted_for():
    """The detail reaches the session log, and what it has to convey is not "the network was
    slow" but "nobody knows where the hook is", which is why the session stops."""
    mover = RemoteHookMover(FakeLink(), move_timeout_s=1.0)
    mover.request("index")
    mover.poll(now=100.0)
    outcome = mover.poll(now=102.0)
    assert "cannot be accounted for" in outcome.detail


def test_a_timeout_also_tells_the_far_end_to_stop():
    """The brain gave up, so the far arm must not still be trying."""
    link = FakeLink()
    mover = RemoteHookMover(link, move_timeout_s=1.0)
    mover.request("index")
    mover.poll(now=100.0)
    link.sent.clear()
    mover.poll(now=102.0)
    assert "abort" in link.kinds()


def test_a_late_report_cannot_answer_the_current_move():
    """The mistake this prevents: a CONFIRMED from a move that already timed out arriving while
    a different move is in flight, and being read as that move succeeding."""
    wire = Loopback()
    mover = RemoteHookMover(wire.brain, move_timeout_s=1.0)
    actuator = ActuatorSide(wire.actuator)

    mover.request("index")
    mover.poll(now=100.0)
    wire.deliver()
    actuator.step(now=100.0)
    assert mover.poll(now=102.0).result is HookResult.FAILED   # gave up on move 1

    mover.request("middle")                                    # move 2 starts
    mover.poll(now=103.0)
    actuator.report(HookResult.CONFIRMED, "late answer to move 1")
    wire.deliver()

    assert mover.poll(now=103.1) is None, "move 1's report must not complete move 2"
    assert any("not the move in progress" in r for r in mover.rejected)


def test_a_report_with_nothing_in_flight_is_dropped():
    link = FakeLink()
    mover = RemoteHookMover(link)
    from session.wire import encode_hook_report

    link.inbox.append(encode_hook_report(1, HookResult.CONFIRMED, "index"))
    assert mover.poll(now=100.0) is None


def test_a_malformed_message_is_logged_and_dropped_not_acted_on():
    link = FakeLink()
    mover = RemoteHookMover(link)
    link.inbox.append({"schema": 999, "kind": "hook_report"})
    assert mover.poll(now=100.0) is None
    assert mover.rejected


def test_poll_needs_a_clock_to_time_out():
    """Documented rather than desirable: without a clock the timeout cannot fire, so every
    real caller passes one. The optional argument exists only to satisfy `HookMover`."""
    mover = RemoteHookMover(FakeLink(), move_timeout_s=0.001)
    mover.request("index")
    assert mover.poll() is None
    assert mover.poll() is None


def test_aborting_tells_the_far_end_and_clears_the_move():
    link = FakeLink()
    mover = RemoteHookMover(link)
    mover.request("index")
    mover.poll(now=100.0)
    link.sent.clear()
    mover.abort()
    assert "abort" in link.kinds()
    assert mover.awaiting_report is None


def test_aborting_with_nothing_in_progress_is_safe():
    """`HookMover.abort` must be safe to call at any time."""
    mover = RemoteHookMover(FakeLink())
    mover.abort()
    mover.abort()


def test_a_nonsense_timeout_is_refused():
    with pytest.raises(ValueError, match="move_timeout_s must be positive"):
        RemoteHookMover(FakeLink(), move_timeout_s=0.0)
    with pytest.raises(ValueError, match="lease_period_s must be positive"):
        RemoteHookMover(FakeLink(), lease_period_s=0.0)


# ------------------------------------------- the actuator stops when the brain goes quiet
def test_the_actuator_will_not_move_without_an_authorised_request():
    actuator = ActuatorSide(FakeLink())
    decision = actuator.step(now=100.0)
    assert not decision.may_move
    assert decision.reason == "no authorised move"


def test_the_actuator_moves_while_the_lease_is_fresh():
    wire = Loopback()
    mover = RemoteHookMover(wire.brain, lease_period_s=0.5)
    actuator = ActuatorSide(wire.actuator)
    mover.request("index")
    mover.poll(now=100.0)
    wire.deliver()
    assert actuator.step(now=100.2).move_to == "index"


def test_the_actuator_halts_when_the_lease_expires():
    """The branch a dead link takes, and the reason this is a lease and not a flag. A flag set
    by "you may move" stays set when the brain dies and the far arm keeps going with nobody
    able to stop it."""
    wire = Loopback()
    mover = RemoteHookMover(wire.brain, lease_period_s=0.5)
    actuator = ActuatorSide(wire.actuator)
    mover.request("index")
    mover.poll(now=100.0)
    wire.deliver()
    assert actuator.step(now=100.2).may_move

    # The brain goes quiet. No abort arrives, because it cannot.
    stopped = actuator.step(now=101.0)
    assert not stopped.may_move
    assert stopped.halt, "it was moving, so it has to be brought to a stop"
    assert "gone quiet" in stopped.reason


def test_the_actuator_stays_stopped_once_the_lease_has_gone():
    """No path revives an expired lease: permission has to be re-granted by a new request."""
    wire = Loopback()
    mover = RemoteHookMover(wire.brain, lease_period_s=0.5)
    actuator = ActuatorSide(wire.actuator)
    mover.request("index")
    mover.poll(now=100.0)
    wire.deliver()
    actuator.step(now=100.1)
    actuator.step(now=101.0)
    for t in (101.1, 102.0, 200.0):
        assert not actuator.step(now=t).may_move


def test_heartbeats_keep_the_actuator_moving_past_one_lease_period():
    wire = Loopback()
    mover = RemoteHookMover(wire.brain, lease_period_s=0.5)
    actuator = ActuatorSide(wire.actuator)
    mover.request("index")
    mover.poll(now=100.0)
    wire.deliver()
    for t in (100.2, 100.4, 100.6, 100.8, 101.0, 101.2):
        mover.poll(now=t)
        wire.deliver()
        assert actuator.step(now=t).may_move, f"should still be authorised at {t}"


def test_a_heartbeat_for_another_move_authorises_nothing():
    """How a brain that restarted and reused an id would otherwise keep an abandoned move
    alive."""
    wire = Loopback()
    actuator = ActuatorSide(wire.actuator)
    mover = RemoteHookMover(wire.brain, lease_period_s=0.5)
    mover.request("index")
    mover.poll(now=100.0)
    wire.deliver()
    actuator.step(now=100.1)

    from session.wire import Lease, encode_heartbeat

    wire.actuator.inbox.append(encode_heartbeat(Lease(move_id=99, issued_at=100.9,
                                                      period_s=0.5)))
    assert not actuator.step(now=101.0).may_move
    assert any("not the move in hand" in r for r in actuator.rejected)


def test_an_abort_stops_the_actuator_immediately():
    wire = Loopback()
    mover = RemoteHookMover(wire.brain)
    actuator = ActuatorSide(wire.actuator)
    mover.request("index")
    mover.poll(now=100.0)
    wire.deliver()
    actuator.step(now=100.0)

    mover.abort()
    wire.deliver()
    stopped = actuator.step(now=100.1)
    assert not stopped.may_move
    assert stopped.halt
    assert "aborted" in stopped.reason


def test_the_actuator_cannot_report_without_a_move_in_hand():
    """An unattributable report is one the brain is guaranteed to drop, so the mistake is worth
    catching on the sending side."""
    actuator = ActuatorSide(FakeLink())
    with pytest.raises(RuntimeError, match="nothing to report"):
        actuator.report(HookResult.CONFIRMED)


def test_the_actuator_cannot_report_twice_for_one_move():
    wire = Loopback()
    mover = RemoteHookMover(wire.brain)
    actuator = ActuatorSide(wire.actuator)
    mover.request("index")
    mover.poll(now=100.0)
    wire.deliver()
    actuator.step(now=100.0)
    actuator.report(HookResult.CONFIRMED)
    with pytest.raises(RuntimeError, match="nothing to report"):
        actuator.report(HookResult.CONFIRMED)


def test_the_actuator_holds_no_session_state():
    """Rule 4 of docs/two-machine-session.md: exactly one orchestrator exists, and it is not
    this. The actuator cannot form an opinion about the session because it is never given the
    information to have one."""
    actuator = ActuatorSide(FakeLink())
    forbidden = ("phase", "rep", "protocol", "block", "cap", "mode", "custody")
    present = [n for n in vars(actuator) if any(f in n.lower() for f in forbidden)]
    assert present == [], f"the actuator should not track {present}"


def test_a_malformed_message_does_not_authorise_the_actuator():
    actuator = ActuatorSide(FakeLink())
    actuator.link.inbox.append({"schema": 1, "kind": "hook_request"})   # no finger, no lease
    assert not actuator.step(now=100.0).may_move
    assert actuator.rejected


def test_the_orchestrator_drives_a_remote_arm_with_no_changes_to_itself():
    """The design claim, checked rather than asserted in a doc: a `RemoteHookMover` drops into
    the orchestrator exactly where a `ManualHookMover` goes, and the sequencing is unchanged.

    If this ever needs the orchestrator to know about a link, the seam has moved and
    docs/two-machine-session.md is wrong.
    """
    from lib.units import grams_force_to_newtons
    from session.interlock import ArmOneStatus
    from session.orchestrator import Command, Orchestrator
    from session.protocol import Block, Protocol

    wire = Loopback()
    mover = RemoteHookMover(wire.brain)
    actuator = ActuatorSide(wire.actuator)
    protocol = Protocol(
        blocks=[Block.from_grams("index", "active", reps=1, cap_gf=100)],
        ceiling_n=grams_force_to_newtons(300),
    )
    session = Orchestrator(protocol, mover)

    slack = ArmOneStatus(released=True, force_n=0.05, reading_age_s=0.01)
    now = 100.0
    saw_move_hook = False
    for _ in range(50):
        now += 0.1
        step = session.step(slack)
        if step.command is Command.MOVE_HOOK:
            saw_move_hook = True
        # The brain's side needs a clock, and the orchestrator has none to give it: the loop
        # driving the session is what calls poll with one. Standing in for that here.
        mover.poll(now=now)
        wire.deliver()
        if actuator.step(now=now).may_move:
            actuator.report(HookResult.CONFIRMED, "confirmed on the far machine")
        wire.deliver()
        if step.command is Command.RUN_BLOCK:
            break
    assert saw_move_hook, "the session should have asked for a hook move"
    assert session.hooked_finger == "index", (
        "a confirmed remote move should advance the session exactly as a local one does"
    )
