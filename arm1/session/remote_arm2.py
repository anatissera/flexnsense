"""Arm 2 over a link: the brain's side, and the actuator's side, both pure.

Two classes, one per machine, and neither of them owns a socket. They take a `Link`, which is
anything with `send(message)` and `drain() -> list[message]`, so every behaviour that matters
here is tested against a list in memory with a fake clock (`tests/session/test_remote_arm2.py`)
rather than against a server that has to be started and raced with.

That is not a testing convenience, it is the only way the interesting cases get covered at all.
What can go wrong in a distributed session is almost entirely timing: a report that never
comes, a report that arrives after everyone gave up, a lease that went stale while the far arm
was mid-move, an abort that crosses a report in flight. None of those are reproducible against
a real socket on demand, and all of them are three lines against a fake one.

The design, and why arm 1's machine is the brain, is in docs/two-machine-session.md.

## The brain's side is a `HookMover`

`RemoteHookMover` satisfies `session.arm2.HookMover` exactly as `ManualHookMover` does, so
`session.orchestrator` cannot tell that a network is involved and its own tests keep covering
the sequencing unchanged. That the seam held is checkable: `tests/session/test_orchestrator.py`
needed no edit for this.

## Silence never becomes success

`session/arm2.py` already requires that success is reported and never assumed, which is why
`PolicyHookMover` raises rather than returning a plausible outcome. Over a link, the same rule
has teeth: a move that goes unanswered past its deadline resolves to `HookResult.FAILED`, and
there is no path in this module from "heard nothing" to `CONFIRMED`. A hook reported as moved
when it has not moved means arm 1 starts pulling on a finger the hook is not attached to.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from arm1.session.arm2 import HookMover, HookOutcome, HookResult
from arm1.session.fingers import normalise
from arm1.session.wire import (
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

__all__ = ["Link", "RemoteHookMover", "ActuatorSide", "ActuatorDecision"]

#: How long a move may go unanswered before it is failed, in seconds. Generous compared with
#: anything in `control/`, and deliberately so: this is a person or a policy physically moving
#: a hook between fingers, not a control cycle, so tens of seconds is normal and a timeout
#: here means the far machine is gone rather than slow.
DEFAULT_MOVE_TIMEOUT_S = 90.0

#: How long a lease is good for without renewal. Short, because this is the number that decides
#: how long a remote arm keeps moving after the brain goes quiet.
DEFAULT_LEASE_PERIOD_S = 0.5


class Link(Protocol):
    """A message channel. Both ends of this module take one, and neither implements one.

    Deliberately smaller than a socket: no connect, no close, no addresses. Everything in this
    module is then testable against a list, and the real transport (FastAPI WebSocket, matching
    `gauge_reader/server.py` and `hand_tracking/server.py`) is a separate, thin adapter whose
    own bugs cannot hide inside the protocol logic.
    """

    def send(self, message: dict[str, Any]) -> None:
        """Queue a message for the other end. Must not block."""
        ...

    def drain(self) -> list[dict[str, Any]]:
        """Every message that has arrived since the last call, oldest first."""
        ...


# ------------------------------------------------------------------------------- brain's side
class RemoteHookMover:
    """`HookMover` backed by a second machine. Lives on arm 1's machine.

    Poll-based, which is what `session.arm2.HookMover` already is, and the reason that shape
    was chosen pays off here: the orchestrator stays a pure state machine with no clock and no
    I/O, and all the network's waiting happens inside `poll`.

    The clock is an argument to `poll` rather than read from `time`, for the same reason it is
    an argument everywhere else in `session/`: the timeout is the behaviour most worth testing
    and it is untestable against a real clock.
    """

    def __init__(self, link: Link, move_timeout_s: float = DEFAULT_MOVE_TIMEOUT_S,
                 lease_period_s: float = DEFAULT_LEASE_PERIOD_S):
        if move_timeout_s <= 0:
            raise ValueError(f"move_timeout_s must be positive, got {move_timeout_s}")
        if lease_period_s <= 0:
            raise ValueError(f"lease_period_s must be positive, got {lease_period_s}")
        self.link = link
        self.move_timeout_s = move_timeout_s
        self.lease_period_s = lease_period_s
        self._move_id = 0
        self._pending: str | None = None
        self._requested_at: float | None = None
        self._lease: Lease | None = None
        self._outcome: HookOutcome | None = None
        #: Messages that could not be trusted, for the session log. A link delivering these is
        #: broken or mismatched, and that is worth seeing rather than silently dropping.
        self.rejected: list[str] = []

    # `HookMover` requires `request(finger)` with no clock, so the request is staged here and
    # actually sent on the next `poll`, which is the call that has a clock. Keeping the
    # protocol's signature unchanged is what lets the orchestrator stay unaware of all this.
    def request(self, finger: str) -> None:
        self._move_id += 1
        self._pending = normalise(finger)
        self._requested_at = None
        self._lease = None
        self._outcome = None

    def poll(self, now: float | None = None) -> HookOutcome | None:
        """The outcome, or None while the move is still in progress.

        `now` is optional only so this still satisfies `HookMover` for a caller that does not
        have a clock to hand; every real caller passes one, and without it the timeout cannot
        fire, which `poll_needs_a_clock_to_time_out` pins down.
        """
        self._receive(now)
        if self._outcome is not None:
            outcome, self._outcome = self._outcome, None
            return outcome
        if self._pending is None or now is None:
            return None

        if self._requested_at is None:
            # First poll after a request: issue the lease and send the request together, so the
            # actuator never sees a request it is not yet authorised to act on.
            self._requested_at = now
            self._lease = Lease(move_id=self._move_id, issued_at=now,
                                period_s=self.lease_period_s)
            self.link.send(encode_hook_request(self._pending, self._lease))
            return None

        if now - self._requested_at > self.move_timeout_s:
            # Heard nothing in time. This resolves to FAILED and never to CONFIRMED: the
            # orchestrator stops the session on a failed hook move, which is the correct
            # response to a hook whose position nobody can account for.
            finger, self._pending = self._pending, None
            self._requested_at = None
            self._lease = None
            self.link.send(encode_abort(self._move_id, "no report before the deadline"))
            return HookOutcome(
                HookResult.FAILED, finger,
                f"the far machine sent no report within {self.move_timeout_s:g} s, so where the "
                "hook is cannot be accounted for"
            )

        # Renew while the move is in progress. The far arm stops when these stop arriving,
        # which is the whole point of a lease rather than a flag.
        assert self._lease is not None
        self._lease = self._lease.renewed(now)
        self.link.send(encode_heartbeat(self._lease))
        return None

    def abort(self) -> None:
        """Stop the current move. Safe at any time, including with nothing in progress."""
        if self._pending is not None:
            self.link.send(encode_abort(self._move_id, "aborted by the session"))
        self._pending = None
        self._requested_at = None
        self._lease = None
        self._outcome = None

    @property
    def awaiting_report(self) -> str | None:
        """The finger a move is in flight for, or None. For the dashboard and the log."""
        return self._pending

    def _receive(self, now: float | None) -> None:
        for raw in self.link.drain():
            try:
                kind, fields = decode(raw)
            except WireError as exc:
                self.rejected.append(str(exc))
                continue
            if kind is not MessageKind.HOOK_REPORT:
                # The brain issues the other three kinds and never receives them. An actuator
                # sending one is confused about which end it is, which is worth logging.
                self.rejected.append(f"unexpected {kind.value} from the actuator")
                continue
            if fields["move_id"] != self._move_id or self._pending is None:
                # A report for a move that is over, or that was never made. Dropping it is the
                # point: a late CONFIRMED from a previous move must never be read as the answer
                # to the current one.
                self.rejected.append(
                    f"report for move {fields['move_id']}, which is not the move in progress"
                )
                continue
            self._outcome = outcome_from_report(fields)
            self._pending = None
            self._requested_at = None
            self._lease = None


# ---------------------------------------------------------------------------- actuator's side
@dataclass(frozen=True)
class ActuatorDecision:
    """What the actuator should do this cycle. One per step, like `orchestrator.Step`."""

    #: The finger to move the hook to while this is set. `None` means hold still.
    move_to: str | None = None
    #: Why the actuator is holding still, when it is. Goes in its log.
    reason: str = ""
    #: Set when motion has just been halted and the arm should be brought to a stop, as
    #: opposed to simply never having been started.
    halt: bool = False

    @property
    def may_move(self) -> bool:
        return self.move_to is not None


@dataclass
class ActuatorSide:
    """The far machine's whole decision logic. Holds no session state, and decides nothing.

    It executes and reports, which is rule 4 in docs/two-machine-session.md: exactly one
    orchestrator exists, and it is not this. There is no phase here, no rep count, no protocol;
    the actuator cannot form an opinion about the session because it is not given the
    information to have one.

    What it does own is **stopping**. `step` returns `move_to=None` the moment the lease
    authorising a move has expired, which is the rule that makes a link failure safe: the arm
    halts because permission ran out, not because anyone managed to send a stop.
    """

    link: Link
    _move_to: str | None = None
    _lease: Lease | None = None
    #: This machine's own clock when `_lease` arrived. The lease is timed from here, never
    #: from the brain's `issued_at`: two machines' monotonic clocks share no origin, so
    #: comparing them asks which booted first rather than whether permission has run out.
    _lease_at: float | None = None
    _moving: bool = False
    #: Messages that could not be trusted, for this machine's log.
    rejected: list[str] = field(default_factory=list)

    def step(self, now: float) -> ActuatorDecision:
        """Consume what arrived, then say whether motion is authorised right now."""
        for raw in self.link.drain():
            try:
                kind, fields = decode(raw)
            except WireError as exc:
                self.rejected.append(str(exc))
                continue
            if kind is MessageKind.HOOK_REQUEST:
                self._move_to = fields["finger"]
                self._lease = fields["lease"]
                self._lease_at = now
            elif kind is MessageKind.HEARTBEAT:
                renewal = fields["lease"]
                if self._lease is not None and renewal.move_id == self._lease.move_id:
                    self._lease = renewal
                    # The renewal's clock starts now, on this machine, which is what makes
                    # "the brain has gone quiet" the thing actually being measured.
                    self._lease_at = now
                else:
                    # A renewal for a move this end is not doing cannot authorise anything. It
                    # is how a stale brain, or one that restarted and reused an id, would
                    # otherwise keep an abandoned move alive.
                    self.rejected.append(
                        f"heartbeat for move {renewal.move_id}, which is not the move in hand"
                    )
            elif kind is MessageKind.ABORT:
                return self._halt("aborted by the brain")
            else:
                self.rejected.append(f"unexpected {kind.value} from the brain")

        if self._move_to is None or self._lease is None or self._lease_at is None:
            return ActuatorDecision(reason="no authorised move")
        if not self._lease.fresh_at(now, received_at=self._lease_at):
            # Permission expired. This is the branch a dead link takes, and it is the same
            # shape as `control.safety.SafetyGate` refusing to move on a stale force reading:
            # the absence of fresh authorisation is itself the instruction to stop.
            return self._halt(
                f"the lease on move {self._lease.move_id} expired; the brain has gone quiet"
            )
        self._moving = True
        return ActuatorDecision(move_to=self._move_to)

    def report(self, result: HookResult, detail: str = "") -> None:
        """Send the outcome of the move in hand. The only message that may complete a move.

        Refuses when there is no move in hand, rather than inventing a `move_id`: an
        unattributable report is exactly what `RemoteHookMover` drops, so sending one would be
        a message guaranteed to be ignored, and the mistake worth catching here instead.
        """
        if self._lease is None or self._move_to is None:
            raise RuntimeError(
                "nothing to report: no authorised move is in hand. A report has to carry the "
                "move_id it answers, or the brain cannot tell which request it belongs to"
            )
        self.link.send(
            encode_hook_report(self._lease.move_id, result, self._move_to, detail)
        )
        self._move_to = None
        self._lease = None
        self._lease_at = None
        self._moving = False

    def _halt(self, reason: str) -> ActuatorDecision:
        was_moving, self._moving = self._moving, False
        self._move_to = None
        self._lease = None
        self._lease_at = None
        return ActuatorDecision(reason=reason, halt=was_moving)
