"""The safety gate: given one force reading, decide whether the arm may move at all.

Pure and side-effect free, so every branch is unit-tested without a camera or a robot (see
tests/control/test_safety.py). Nothing here commands anything; it returns a decision and the
caller acts on it. That split is deliberate: the rule "a missing or low-confidence reading
leads to release and stop, never to a guess" (docs/contributing.md) is the one thing in this project that
must be verifiable by reading a single function.

Design background: docs/admittance-design.md. The camera is the only force sensor on this arm
(docs/force-calibration-result.md), so this gate has no second signal to fall back on.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    """What the caller must do with the arm this cycle."""

    HOLD = "hold"        # under the cap: the arm is the anchor, do not move
    YIELD = "yield"      # over the cap: move to reduce tension, by the control law
    RELEASE = "release"  # unload the spring and stop commanding; recoverable only by a human


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str
    force_n: float | None = None   # the reading this decision was made on, when there was one

    @property
    def may_move(self) -> bool:
        return self.action is Action.YIELD


@dataclass(frozen=True)
class SafetyLimits:
    """Everything that can stop the arm. Units are newtons, seconds and metres (docs/contributing.md)."""

    f_threshold_n: float      # the clinician's cap; the plateau the exercise holds at
    f_max_n: float            # hard ceiling; above this it releases, no grace
    max_age_s: float = 0.1    # a reading older than this is stale, not a reading
    max_dropouts: int = 2     # consecutive frames without a reading before releasing
    max_travel_m: float = 0.05  # total displacement from the session start
    #: How far below `f_threshold_n` the arm starts yielding, so it is already moving by the
    #: time the reading reaches the cap rather than starting from a standstill there. Zero
    #: (the default) is the literal cap, exactly as before this field existed: no motion
    #: until the reading is actually over it.
    #:
    #: Why this needs to exist at all: the loop from camera frame to commanded motion has real
    #: latency (grab, detect, decide, write to the servo), and the admittance law's own
    #: velocity is zero right at the cap by construction (`yield_velocity`'s overshoot is zero
    #: there). A finger that keeps closing in that gap can push the reading well past the cap
    #: before the arm has any speed at all, which on this bench showed up as reaching the
    #: ceiling almost immediately after the cap in a fast rep. Starting the ramp a margin
    #: early means there is already nonzero velocity by the time the cap is reached, at the
    #: cost of yielding a little below the nominal cap, which is a session-level,
    #: clinician-level tradeoff, hence a field here and not a hardcoded constant.
    anticipation_n: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.f_threshold_n < self.f_max_n:
            raise ValueError(
                f"need 0 < f_threshold_n ({self.f_threshold_n}) < f_max_n ({self.f_max_n})"
            )
        if self.max_age_s <= 0:
            raise ValueError(f"max_age_s must be positive, got {self.max_age_s}")
        if self.max_dropouts < 0:
            raise ValueError(f"max_dropouts cannot be negative, got {self.max_dropouts}")
        if self.max_travel_m <= 0:
            raise ValueError(f"max_travel_m must be positive, got {self.max_travel_m}")
        if not 0 <= self.anticipation_n < self.f_threshold_n:
            raise ValueError(
                f"need 0 <= anticipation_n ({self.anticipation_n}) < f_threshold_n "
                f"({self.f_threshold_n}): it is a margin below the cap, not a new cap"
            )

    @property
    def yield_from_n(self) -> float:
        """The force above which the arm actually starts to yield: the cap, moved earlier by
        `anticipation_n`. What `SafetyGate.check` and `ControlLoop` both act on; `f_threshold_n`
        stays the number everything else (the printed cap, the exercise's target plateau) is
        described in terms of."""
        return self.f_threshold_n - self.anticipation_n


class SafetyGate:
    """Stateful across frames, because tolerating a dropout at all requires counting them.

    Once it releases it stays released: `latched` is only cleared by `reset()`, which a human
    has to trigger. A gate that could un-release itself would let a flickering reading restart
    a pull on a finger nobody is watching.
    """

    def __init__(self, limits: SafetyLimits):
        self.limits = limits
        self.consecutive_dropouts = 0
        self.latched: Decision | None = None

    def reset(self) -> None:
        """Clear a latched release. Only call this when a person has decided it is safe."""
        self.consecutive_dropouts = 0
        self.latched = None

    def check(
        self,
        force_n: float | None,
        age_s: float,
        travel_m: float = 0.0,
    ) -> Decision:
        """One frame in, one decision out.

        `force_n`: the reading, or None when the gauge reader did not see the marker this
        frame. `age_s`: how long ago that frame was captured. `travel_m`: absolute
        displacement from where the session started.
        """
        if self.latched is not None:
            return self.latched

        if age_s > self.limits.max_age_s:
            return self._latch("stale reading", force_n)

        if abs(travel_m) > self.limits.max_travel_m:
            return self._latch(
                f"travel {travel_m:+.3f} m beyond limit {self.limits.max_travel_m:.3f} m", force_n
            )

        if force_n is None:
            self.consecutive_dropouts += 1
            if self.consecutive_dropouts > self.limits.max_dropouts:
                return self._latch(
                    f"{self.consecutive_dropouts} consecutive frames without a reading", None
                )
            # Within tolerance: hold position, do not move on a force we cannot see.
            return Decision(
                Action.HOLD,
                f"no reading ({self.consecutive_dropouts}/{self.limits.max_dropouts})",
                None,
            )

        self.consecutive_dropouts = 0

        if force_n > self.limits.f_max_n:
            return self._latch(
                f"force {force_n:.2f} N over hard ceiling {self.limits.f_max_n:.2f} N", force_n
            )

        if force_n > self.limits.yield_from_n:
            note = (f"over cap {self.limits.f_threshold_n:.2f} N" if self.limits.anticipation_n == 0
                   else f"over yield start {self.limits.yield_from_n:.2f} N, anticipating "
                        f"cap {self.limits.f_threshold_n:.2f} N")
            return Decision(Action.YIELD, f"force {force_n:.2f} N {note}", force_n)

        return Decision(Action.HOLD, f"force {force_n:.2f} N under cap", force_n)

    def _latch(self, reason: str, force_n: float | None) -> Decision:
        self.latched = Decision(Action.RELEASE, reason, force_n)
        return self.latched
