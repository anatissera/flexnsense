"""The messages two machines send each other, and the lease that keeps the far one honest.

Pure: no sockets, no clock, no I/O. Every function here turns data into data, and the clock
arrives as a number, for the same reason `session.orchestrator` and `control.safety` are built
that way. The failures worth testing in a distributed session are all timing failures (a
message that never comes, an outcome that arrives after everyone gave up, a lease that went
stale while the far arm was mid-move), and timing failures are only testable when the clock is
an argument.

The design this implements, including why arm 1's machine is the brain and arm 2's is a remote
actuator that holds no session state, is in docs/two-machine-session.md. The part of it that
lives in this file is the wire vocabulary and the lease.

## The lease, which is the whole safety idea

Custody of motion is not a flag that gets set. It is a **lease with a deadline**: arm 2 may
move only while it holds one that has not expired, and it halts the moment it has.

A flag would be wrong in exactly one way, and it is the way that matters. If the brain grants
motion and then the link dies, a flag stays set and the far arm keeps going with nobody able to
stop it. That is the networked form of the failure `control.safety.SafetyGate` and
`session.reps.RepMachine._deadman` both exist to prevent: a mechanism that keeps moving a
patient because the thing that was supposed to say "stop" cannot be heard. The brain renews the
lease while it can see that renewing is safe; silence therefore means stop, which is the only
safe meaning silence can have.

So the lease has to be renewed to stay alive, and nothing in this module can extend one that
has already expired (`Lease.renewed` builds a new lease; it does not mutate an old one).

## Versioning

`SCHEMA_VERSION` is checked on decode and a mismatch is refused rather than tolerated. Two
machines running different versions of this protocol is a configuration error that should
surface at the handshake, not as a hook move that silently means something else.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from session.arm2 import HookOutcome, HookResult
from session.fingers import FINGERS, normalise

__all__ = [
    "SCHEMA_VERSION",
    "MessageKind",
    "Lease",
    "WireError",
    "encode_hook_request",
    "encode_hook_report",
    "encode_heartbeat",
    "encode_abort",
    "decode",
]

#: Bumped whenever a message's meaning changes. Checked on decode; a mismatch is refused.
SCHEMA_VERSION = 1


class WireError(ValueError):
    """A message that cannot be trusted to mean what it says.

    Its own class because the right response differs from a normal `ValueError`: a malformed or
    wrong-version message must not be retried or guessed at, it must drop the link and surface
    at the handshake.
    """


class MessageKind(Enum):
    """Every message either direction may send. Deliberately few.

    What is *not* here is as much of the design as what is. There is no "custody granted"
    message, because custody is carried on every request as a lease rather than handed over
    once; there is no "status" message from arm 2, because it holds no session state to report;
    and there is no force reading anywhere, because the force never leaves the machine that
    measures it (docs/two-machine-session.md).
    """

    #: Brain to actuator: move the hook to a finger. Carries the lease that authorises it.
    HOOK_REQUEST = "hook_request"
    #: Actuator to brain: what happened. The only message that may complete a move.
    HOOK_REPORT = "hook_report"
    #: Brain to actuator: the lease is still good. Keeps a move alive past one lease period.
    HEARTBEAT = "heartbeat"
    #: Brain to actuator: stop, whatever you were doing. Never refused, never acknowledged as
    #: success.
    ABORT = "abort"


@dataclass(frozen=True)
class Lease:
    """Permission for the far arm to move, good for `period_s` after it is **received**.

    The one thing this type must not do is compare the two machines' clocks. `issued_at` is
    the brain's `time.monotonic()`, and two monotonic clocks have no common origin: one counts
    from its own boot and the other from its. Comparing them does not ask "has the lease
    expired", it asks "which machine booted first", and the answer is constant for a whole
    session. That was a real bug here, found by running the protocol against the actuator: in
    one direction every lease looked fresh for ever, which is precisely the flag-not-a-lease
    failure this design exists to prevent, and in the other the arm refused to move at all.

    So expiry is measured by the receiver, against its own clock, from when the message
    arrived: `expires_once_received_at(received_at)`. `issued_at` is only an identifier and a
    log line, never a deadline, and `period_s` travels with the lease so the actuator enforces
    the brain's chosen period without being configured to match it.

    Monotonic seconds at each end, never wall clock: a session that straddles a clock
    adjustment must not become a session where a lease never expires.
    """

    #: Identifies the move this lease authorises, so a renewal cannot revive a finished one.
    move_id: int
    #: The brain's monotonic clock when this lease was issued. **Not comparable with the
    #: actuator's clock**, and never used as a deadline; see the class docstring.
    issued_at: float
    #: How long a lease is good for, counted from when the receiver got it.
    period_s: float

    def __post_init__(self) -> None:
        if self.move_id < 0:
            raise ValueError(f"move_id cannot be negative, got {self.move_id}")
        if self.period_s <= 0:
            raise ValueError(
                f"period_s must be positive, got {self.period_s}: a lease that is valid for no "
                "time cannot authorise anything, and one valid for negative time is a bug "
                "dressed as permission"
            )

    @property
    def expires_at(self) -> float:
        """When this lease expires **on the clock that issued it**.

        Only meaningful to the brain, for its own bookkeeping. The actuator must not use it:
        see `expires_once_received_at`.
        """
        return self.issued_at + self.period_s

    def expires_once_received_at(self, received_at: float) -> float:
        """When this lease expires, on the clock of whoever received it at `received_at`."""
        return received_at + self.period_s

    def fresh_at(self, now: float, received_at: float | None = None) -> bool:
        """Whether this lease still authorises motion at `now`.

        `received_at` is `now` on the **receiver's** clock when this lease arrived, and is what
        a receiver on another machine must pass: without it this compares the issuer's clock
        with the caller's, which across two machines is not a comparison at all (see the class
        docstring). It defaults to `issued_at` so the issuer can ask about its own lease.

        Expiry is exclusive: a lease is dead *at* its deadline, not after it. One cycle of
        doubt resolves toward stopping.
        """
        start = self.issued_at if received_at is None else received_at
        return now < start + self.period_s

    def renewed(self, now: float) -> "Lease":
        """A new lease for the same move, issued now.

        Returns a new object rather than mutating, so there is no code path that extends a
        lease that has already expired: whoever renews has to decide, with the clock in hand,
        that renewing is still the right thing.
        """
        return Lease(move_id=self.move_id, issued_at=now, period_s=self.period_s)

    def to_dict(self) -> dict[str, Any]:
        return {"move_id": self.move_id, "issued_at": self.issued_at,
                "period_s": self.period_s}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Lease":
        missing = {"move_id", "issued_at", "period_s"} - set(raw)
        if missing:
            raise WireError(f"lease is missing {sorted(missing)}")
        try:
            return cls(move_id=int(raw["move_id"]), issued_at=float(raw["issued_at"]),
                       period_s=float(raw["period_s"]))
        except (TypeError, ValueError) as exc:
            raise WireError(f"lease has unusable fields: {exc}") from exc


# ----------------------------------------------------------------------------------- encoding
def _envelope(kind: MessageKind, **fields: Any) -> dict[str, Any]:
    return {"schema": SCHEMA_VERSION, "kind": kind.value, **fields}


def encode_hook_request(finger: str, lease: Lease) -> dict[str, Any]:
    """Brain to actuator: move the hook to `finger`, authorised by `lease`.

    The finger is normalised here so a brain and an actuator cannot disagree about whether
    "pinky" and "little" are the same finger. `session.fingers` is the single vocabulary, and
    an unknown name is refused rather than forwarded.
    """
    return _envelope(MessageKind.HOOK_REQUEST, finger=normalise(finger),
                     lease=lease.to_dict())


def encode_hook_report(move_id: int, result: HookResult, finger: str,
                       detail: str = "") -> dict[str, Any]:
    """Actuator to brain: this is what happened to move `move_id`.

    Carries `move_id` so a report cannot be mistaken for the answer to a different request,
    which is the mistake that would have arm 1 pull on a finger the hook left.
    """
    return _envelope(MessageKind.HOOK_REPORT, move_id=int(move_id), result=result.value,
                     finger=normalise(finger), detail=detail)


def encode_heartbeat(lease: Lease) -> dict[str, Any]:
    """Brain to actuator: the lease on `lease.move_id` is renewed as of now."""
    return _envelope(MessageKind.HEARTBEAT, lease=lease.to_dict())


def encode_abort(move_id: int, reason: str) -> dict[str, Any]:
    """Brain to actuator: stop. Carries a reason so the far end can log why it stopped."""
    return _envelope(MessageKind.ABORT, move_id=int(move_id), reason=reason)


# ----------------------------------------------------------------------------------- decoding
def decode(raw: Any) -> tuple[MessageKind, dict[str, Any]]:
    """`(kind, fields)` for a received message, or `WireError`.

    Strict on purpose, and the strictness is the feature. Every rejection here is a message
    whose meaning is not certain, and the one thing a distributed version of this machine must
    not do is act on a message it half understood: an unknown `kind` could be a newer protocol
    asking for something this end does not implement, and a missing field could be a hook
    report about a move nobody can identify.
    """
    if not isinstance(raw, Mapping):
        raise WireError(f"message must be an object, got {type(raw).__name__}")
    version = raw.get("schema")
    if version != SCHEMA_VERSION:
        raise WireError(
            f"message speaks schema {version!r}, this end speaks {SCHEMA_VERSION}. Two machines "
            "on different protocol versions is a configuration error, so this is refused at the "
            "handshake rather than guessed at"
        )
    try:
        kind = MessageKind(raw.get("kind"))
    except ValueError:
        raise WireError(
            f"unknown message kind {raw.get('kind')!r}; this end knows "
            f"{sorted(k.value for k in MessageKind)}"
        ) from None

    fields = {k: v for k, v in raw.items() if k not in ("schema", "kind")}
    if kind is MessageKind.HOOK_REQUEST:
        _require(fields, "finger", "lease")
        fields["finger"] = _finger(fields["finger"])
        fields["lease"] = Lease.from_dict(fields["lease"])
    elif kind is MessageKind.HEARTBEAT:
        _require(fields, "lease")
        fields["lease"] = Lease.from_dict(fields["lease"])
    elif kind is MessageKind.HOOK_REPORT:
        _require(fields, "move_id", "result", "finger")
        fields["move_id"] = _int(fields["move_id"], "move_id")
        fields["finger"] = _finger(fields["finger"])
        try:
            fields["result"] = HookResult(fields["result"])
        except ValueError:
            raise WireError(
                f"unknown hook result {fields['result']!r}; known results are "
                f"{sorted(r.value for r in HookResult)}"
            ) from None
        fields.setdefault("detail", "")
    elif kind is MessageKind.ABORT:
        _require(fields, "move_id")
        fields["move_id"] = _int(fields["move_id"], "move_id")
        fields.setdefault("reason", "")
    return kind, fields


def outcome_from_report(fields: Mapping[str, Any]) -> HookOutcome:
    """A decoded `HOOK_REPORT` as the `HookOutcome` the orchestrator already understands.

    The point of the conversion living here is that nothing above this module needs to know a
    network was involved: `session.orchestrator` consumes `HookOutcome` whether a person
    confirmed the move on a dashboard or a second machine reported it.
    """
    return HookOutcome(fields["result"], fields["finger"], fields.get("detail", ""))


def _require(fields: Mapping[str, Any], *names: str) -> None:
    missing = [n for n in names if n not in fields]
    if missing:
        raise WireError(f"message is missing {missing}")


def _int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise WireError(f"{name} must be an integer, got {value!r}") from exc


def _finger(value: Any) -> str:
    try:
        return normalise(str(value))
    except ValueError:
        raise WireError(
            f"unknown finger {value!r}; known fingers are {list(FINGERS)}"
        ) from None
