"""The two exercise modes, as data rather than as branches.

Both modes are the same two-beat repetition and differ only in **who is the source of
motion** in each beat:

    active    work: the patient flexes, the arm yields at constant force
              reset: the arm walks back to where the rep began, patient relaxed
    passive   work: the arm pulls the finger toward extension at constant force
              reset: the patient flexes back, the arm releases while they do

So the driver flips within a rep, and flips the other way between modes. That one fact is
why there is a single rep skeleton (`session.reps`) rather than two state machines: the mode
fills in which law runs and who drives, and nothing else. Keeping that in one table here is
what stops a third mode from being a refactor, and it is what `tests/session/test_modes.py`
checks against the instruction table.

Names are `active` and `passive` after active and passive range of motion (AROM and PROM),
the vocabulary a hand therapist already uses, so the clinician-facing labels need no
translation. "tension" would be a bad name because both modes apply tension and share the
same force cap; "accompany" would be a bad name because the arm accompanies in both, leading
the finger in one and following it in the other. `YIELD` and `PULL` name the *laws*, not the
modes, which is why a passive rep uses both of them.

Full reasoning: docs/session-orchestration.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Driver(Enum):
    """Who is the source of motion in a phase. The dashboard keys its whole display off this."""

    PATIENT = "patient"
    DEVICE = "device"


class Law(Enum):
    """What the arm does during a phase.

    `YIELD` and `PULL` are mirror images and are *not* interchangeable from a safety point of
    view, see `PhasePlan.device_driven`.
    """

    #: Move only once the force is over the cap, in the direction that reduces it.
    #: `control.admittance.yield_velocity` and the existing safety gate implement this.
    YIELD = "yield"
    #: Advance while the force is *under* the cap, toward a position target, and stop on
    #: reaching the cap. The arm is the source of motion here.
    PULL = "pull"
    #: Hold position. Used for cueing, holding at end of range, and every terminal phase.
    STILL = "still"
    #: Unload the spring and stop. Only ever reached through `Phase.ABORTED`.
    RELEASE = "release"


class Target(Enum):
    """What bounds how far the arm travels in a phase, when the arm is the one moving."""

    #: No position target: the patient's own range of motion is the limit.
    NONE = "none"
    #: Back to the position this repetition started from.
    REP_START = "rep_start"
    #: The stretch distance the prescription asked for, measured from the rep start.
    ROM = "rom"


class Phase(Enum):
    """One repetition, start to finish. `session.reps` owns the transitions between these."""

    IDLE = "idle"
    #: Take the slack up to just under the cap. Not a formality: on a ~54 N/m spring, reaching
    #: 1.96 N takes ~36 mm of stretch, and any of that left undone is range the patient spends
    #: on a rising ramp instead of on the exercise.
    ARMED = "armed"
    #: The instruction is shown and spoken, and nothing moves until the phase's driver acts.
    CUEING = "cueing"
    #: Force rising toward the cap. Same plan as `WORK`; separate because the force against
    #: displacement curve and the session log want the run-up apart from the exercise.
    RAMP = "ramp"
    #: At the cap, the work phase's law running. This is the exercise.
    WORK = "work"
    #: At end of range, counting out the prescribed hold time.
    HOLD = "hold"
    #: Back to the rep's start position, under the reset phase's law.
    RESET = "reset"
    REP_DONE = "rep_done"
    SET_DONE = "set_done"
    #: The gate latched RELEASE, or a human stopped the session. Terminal until a human resets.
    ABORTED = "aborted"


#: Phases that end a repetition or a set. The rep machine does not command motion in these.
TERMINAL_PHASES = frozenset({Phase.REP_DONE, Phase.SET_DONE, Phase.ABORTED})


@dataclass(frozen=True)
class PhasePlan:
    """Who drives a phase, what the arm does in it, and what bounds the arm's travel."""

    driver: Driver
    law: Law
    target: Target = Target.NONE

    @property
    def device_driven(self) -> bool:
        """True when the arm is the source of motion in this phase.

        **`control.safety.Decision.may_move` must not be used to gate motion in a
        device-driven phase.** It is true only on `Action.YIELD`, which is correct when the
        patient moves and the arm gives way, and exactly backwards here: a device-driven phase
        has to advance while the force is *under* the cap and stop on reaching it. Reusing
        `may_move` would produce an arm that sits still until the finger is already at the
        limit and only then starts pulling.

        Two further rules follow for these phases, both spelled out in
        docs/session-orchestration.md:

        - advancing requires a fresh, confident reading on the current cycle, so a single
          missing frame stops the advance. This is a deadman, not a dropout tolerance: a lost
          reading in a patient-driven phase leaves the arm static, while here it would leave
          the arm pulling with no idea how hard.
        - `control.loop.ControlLoop._check_yield_direction` does not apply. That guard assumes
          the arm is giving way and the force should therefore fall; while the arm pulls, a
          rising force is the expected thing. The position target plus the cap replace it.
        """
        return self.driver is Driver.DEVICE and self.law is Law.PULL


@dataclass(frozen=True)
class Mode:
    """One exercise kind: the work beat and the reset beat, with who drives each."""

    name: str
    work: PhasePlan
    reset: PhasePlan

    @property
    def needs_rom_target(self) -> bool:
        """Whether a prescription for this mode has to carry a stretch distance.

        Only true when some phase pulls toward `Target.ROM`: a device-driven pull is not
        bounded by force alone, because under the cap nothing tells it to stop. A
        patient-driven phase needs no target, the patient's own range being the limit.
        """
        return any(p.target is Target.ROM for p in (self.work, self.reset))


ACTIVE = Mode(
    name="active",
    work=PhasePlan(Driver.PATIENT, Law.YIELD, Target.NONE),
    reset=PhasePlan(Driver.DEVICE, Law.PULL, Target.REP_START),
)

PASSIVE = Mode(
    name="passive",
    work=PhasePlan(Driver.DEVICE, Law.PULL, Target.ROM),
    reset=PhasePlan(Driver.PATIENT, Law.YIELD, Target.NONE),
)

CPM = Mode(
    name="cpm",
    work=PhasePlan(Driver.DEVICE, Law.PULL, Target.ROM),
    reset=PhasePlan(Driver.DEVICE, Law.PULL, Target.REP_START),
)
"""Continuous passive motion: the arm takes the finger out to the stretch target and brings it
back, and the patient does nothing in either beat.

The name is the one a therapist already uses for a machine that does exactly this, which is
also what keeps it honestly distinct from `PASSIVE`. Both are passive in the sense that the
patient is not working, but they are different exercises:

    passive   the arm pulls toward extension, and the *patient* flexes back
    cpm       the arm pulls toward extension, and the *arm* brings the finger back

So `passive` still asks something of the patient once per repetition, and is the mode to
prescribe when that is the point. `cpm` asks nothing, which makes it the mode for maintaining
range in a hand that cannot yet contribute, and for strengthening work where the useful thing
is many identical repetitions rather than the patient's own effort.

Two consequences follow from both beats being device-driven, and they are why this mode is
worth having as its own entry rather than as a flag on `passive`:

- **The repetition is the same motion every time**, because nothing in it waits on a person.
  The path is decided before the stroke begins, which is what lets it run faster and more
  smoothly than a yielding arm chasing a signal (`control.loop.PullPlan`).
- **Both beats are governed by the deadman**, not just one. `PhasePlan.device_driven` is true
  for the work beat *and* the reset beat here, where in the other two modes exactly one beat
  is. A lost force reading stops either half of this stroke.

The force cap is a ceiling in this mode rather than the thing the exercise works against: the
stroke ends at the position target, and the cap only ends it early, when a finger turns out to
be stiffer than the prescription assumed. That is a clinically meaningful early stop and is
reported as one, not treated as a failed repetition.
"""

#: Every mode a prescription may name. `session.protocol` validates against these keys.
MODES: dict[str, Mode] = {m.name: m for m in (ACTIVE, PASSIVE, CPM)}


def mode_by_name(name: str) -> Mode:
    """The mode called `name`, or a `ValueError` listing what is available."""
    try:
        return MODES[name]
    except KeyError:
        raise ValueError(
            f"unknown exercise mode {name!r}; known modes are {sorted(MODES)}"
        ) from None


def plan_for(mode: Mode, phase: Phase) -> PhasePlan:
    """What happens in `phase` under `mode`.

    `ARMED` and `CUEING` are mode-independent in what the arm does, but their driver is not
    arbitrary: `ARMED` is always the device taking up slack, and `CUEING` reports the driver of
    the work phase that is about to start, because that is who is being asked to act. `RAMP`
    deliberately returns the same plan as `WORK`: below the cap the yield law yields nothing
    and the pull law advances, which is already the correct behaviour for each mode, so the
    two phases differ only in what they mean to the log.
    """
    if phase is Phase.ARMED:
        return PhasePlan(Driver.DEVICE, Law.PULL, Target.NONE)
    if phase is Phase.CUEING:
        return PhasePlan(mode.work.driver, Law.STILL, Target.NONE)
    if phase in (Phase.RAMP, Phase.WORK):
        return mode.work
    if phase is Phase.HOLD:
        return PhasePlan(mode.work.driver, Law.STILL, Target.NONE)
    if phase is Phase.RESET:
        return mode.reset
    if phase is Phase.ABORTED:
        return PhasePlan(Driver.DEVICE, Law.RELEASE, Target.NONE)
    return PhasePlan(Driver.DEVICE, Law.STILL, Target.NONE)
