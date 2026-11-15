"""The two-arm interlock: only one arm may be able to move, ever.

This is the safety core of the session layer, kept in its own pure module for the same reason
`control/safety.py` is: it is small, it is the part that can hurt someone, and it has to stay
exhaustively testable.

The hazard it exists for: arm 2 moves the dynamometer's hook from one finger to the next
between blocks. If arm 1 is still holding tension when that happens, the hook is being pulled
off a finger that is still loaded, or dragged across a hand that is still tied to a spring.
That is the worst state this machine has.

So custody of motion is a token. Exactly one holder at a time, `NEITHER` is a legal and common
state, and the transitions are the only way it changes.

The rule that carries the weight:

    **A missing force reading is not evidence of zero force.**

Handing custody to arm 2 requires positive evidence that arm 1 is unloaded: a fresh, confident
reading at or below the slack threshold. A dropout, a stale frame or a low-confidence frame all
refuse the handover, because "we cannot see the force" is exactly when the hook must not move.
The same principle as `control.safety.SafetyGate` refusing to guess, applied one layer up.

Nothing above this may talk it out of refusing. `Interlock` has no method that forces a
handover and none that clears a refusal; a refusal is resolved by making the world safe (arm 1
releases, the reading returns) or by a human stopping the session.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from lib.units import newtons_to_grams_force


class Custody(Enum):
    """Which arm is currently allowed to move. Exactly one value at a time."""

    #: Nothing may move. The state between blocks, and the only state a handover passes
    #: through: custody never goes from one arm to the other directly.
    NEITHER = "neither"
    #: Arm 1, the admittance controller on the patient's finger.
    ARM1 = "arm1"
    #: Arm 2, moving the hook between fingers.
    ARM2 = "arm2"


@dataclass(frozen=True)
class ArmOneStatus:
    """What is known about arm 1 right now, as evidence rather than as a claim.

    `force_n` is `None` when the gauge reader did not see the marker, which is a different
    thing from zero and must never be collapsed into it.
    """

    #: True once arm 1's control loop has unloaded the spring and stopped. A latched RELEASE in
    #: `control.safety` or a completed `ControlLoop.release()` both produce this.
    released: bool
    #: Newtons, or None when the force cannot be seen this cycle.
    force_n: float | None
    #: Seconds since the frame this reading came from. Old evidence is not evidence.
    reading_age_s: float

    @property
    def readable(self) -> bool:
        return self.force_n is not None


@dataclass(frozen=True)
class Verdict:
    """Whether a handover is allowed, and if not, the reason a human can act on."""

    allowed: bool
    reason: str

    def __bool__(self) -> bool:
        return self.allowed


@dataclass(frozen=True)
class InterlockLimits:
    """What counts as unloaded, and how fresh the evidence has to be."""

    #: At or below this, the tether is considered slack enough to move the hook. Not zero: a
    #: real tether has a little residual tension from its own weight and from friction, and
    #: demanding exactly zero would make the handover impossible rather than safe. Keep it well
    #: under any therapeutic force, since the point is that nothing is pulling on the finger.
    slack_n: float = 0.20
    #: Readings older than this are refused outright. Shorter than the control loop's own
    #: staleness limit would be pointless; this is the same idea at the handover.
    max_age_s: float = 0.2

    def __post_init__(self) -> None:
        if self.slack_n <= 0:
            raise ValueError(f"slack_n must be positive, got {self.slack_n}")
        if self.max_age_s <= 0:
            raise ValueError(f"max_age_s must be positive, got {self.max_age_s}")


class InterlockViolation(RuntimeError):
    """Raised when something tries a transition the interlock does not permit.

    An exception rather than a return value: every caller of a `take_*` method is asserting it
    has already checked, so reaching here is a bug in the caller, and the only safe response is
    to stop rather than to continue with a different plan.
    """


class Interlock:
    """Holds custody and decides handovers. Pure: no I/O, no clock, no hardware.

    Usage is always ask then act, and the asking is not optional:

        if interlock.may_hand_to_arm2(status):
            interlock.release_all()
            interlock.take_arm2()
    """

    def __init__(self, limits: InterlockLimits | None = None):
        self.limits = limits if limits is not None else InterlockLimits()
        self._custody = Custody.NEITHER
        self._stopped_reason: str | None = None

    # ------------------------------------------------------------------ state
    @property
    def custody(self) -> Custody:
        return self._custody

    @property
    def stopped(self) -> bool:
        """True once the session has been stopped. Terminal: no further custody is granted."""
        return self._stopped_reason is not None

    @property
    def stopped_reason(self) -> str | None:
        return self._stopped_reason

    # ------------------------------------------------------------------ the decision
    def may_hand_to_arm2(self, arm1: ArmOneStatus) -> Verdict:
        """Whether the hook may move now. The one decision this module exists for.

        Every refusal names what is wrong, because the operator reading it has to be able to
        fix it: release arm 1, clear the camera's view, or stop the session.
        """
        if self.stopped:
            return Verdict(False, f"session stopped: {self._stopped_reason}")
        if self._custody is Custody.ARM2:
            return Verdict(True, "arm 2 already holds custody")
        if self._custody is Custody.ARM1:
            return Verdict(
                False,
                "arm 1 still holds custody; it has to give it up before the hook can move",
            )
        if not arm1.released:
            return Verdict(
                False,
                "arm 1 has not released: moving the hook while the tether is still under "
                "control would drag it off a loaded finger",
            )
        if not arm1.readable:
            # The rule this module exists for. Refusing here is the whole point: a dropout is
            # when we are least able to know whether a finger is loaded.
            return Verdict(
                False,
                "no force reading: a missing reading is not evidence of zero force, and the "
                "hook must not move while the tension cannot be seen",
            )
        if arm1.reading_age_s > self.limits.max_age_s:
            return Verdict(
                False,
                f"force reading is {arm1.reading_age_s * 1000:.0f} ms old, over the "
                f"{self.limits.max_age_s * 1000:.0f} ms limit: stale evidence is not evidence",
            )
        assert arm1.force_n is not None  # readable, checked above
        if arm1.force_n > self.limits.slack_n:
            return Verdict(
                False,
                f"tether still under {newtons_to_grams_force(arm1.force_n):.0f} gf, above the "
                f"{newtons_to_grams_force(self.limits.slack_n):.0f} gf slack threshold: "
                "something is still pulling on the finger",
            )
        return Verdict(True, "arm 1 released and the tether is slack")

    def may_hand_to_arm1(self) -> Verdict:
        """Whether arm 1 may start exercising.

        Deliberately simpler, and deliberately not symmetric. Arm 2 holding the hook is not a
        force on the patient, so the evidence needed is only that arm 2 has finished and given
        custody up, which is what `release_all` records. The asymmetry is the point: the
        dangerous direction is the one where a finger might still be loaded.
        """
        if self.stopped:
            return Verdict(False, f"session stopped: {self._stopped_reason}")
        if self._custody is Custody.ARM1:
            return Verdict(True, "arm 1 already holds custody")
        if self._custody is Custody.ARM2:
            return Verdict(
                False, "arm 2 still holds custody; the hook move has to finish or fail first"
            )
        return Verdict(True, "nothing holds custody")

    # ------------------------------------------------------------------ transitions
    def release_all(self) -> None:
        """Give up custody. Always allowed, and idempotent.

        The only transition with no precondition, because every path to safety goes through it
        and a safety action must never be refused.
        """
        self._custody = Custody.NEITHER

    def take_arm1(self) -> None:
        verdict = self.may_hand_to_arm1()
        if not verdict:
            raise InterlockViolation(f"arm 1 cannot take custody: {verdict.reason}")
        self._custody = Custody.ARM1

    def take_arm2(self, arm1: ArmOneStatus) -> None:
        """Grant custody to arm 2, re-checking the evidence rather than trusting the caller.

        The check is repeated here on purpose. A caller that asked `may_hand_to_arm2` and then
        did something else before acting would otherwise hand over on evidence that has since
        gone stale, and this is the transition where that matters.
        """
        verdict = self.may_hand_to_arm2(arm1)
        if not verdict:
            raise InterlockViolation(f"arm 2 cannot take custody: {verdict.reason}")
        self._custody = Custody.ARM2

    def stop(self, reason: str) -> None:
        """Stop the session. Terminal, and it drops custody on the way.

        No method undoes this. Resuming after a stop means constructing a new session, which
        forces the pre-tension and direction checks in `Phase.ARMED` to run again rather than
        letting a session pick up where an unexplained stop left off.
        """
        if self._stopped_reason is None:
            self._stopped_reason = reason
        self._custody = Custody.NEITHER
