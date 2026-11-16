"""The per-repetition machine: cue the patient, and count what they actually did.

One skeleton for both modes. The mode fills in which law runs in the work and reset beats and
who drives each (`session.modes`), and nothing else differs, because the two exercises are the
same two-beat repetition with the beats in opposite order.

This is where the repo's two disagreeing models of a repetition are reconciled. There were two:
a commanded one, `GuideHand.tsx` counting reps off a `performance.now()` timeline against
hardcoded constants, and an observed one, `LiveAnalyzer` in `curlsim/live.ts` detecting phases
from the measured signal. They were rivals. Here the timeline becomes **the cue** and the
detector becomes **the counter**, which is what "command and confirm" means:

- the transition out of `CUEING` is the driver actually starting, not a timer expiring;
- `REP_DONE` means the movement was measured to completion, so a patient who did not really
  do it is visible instead of being counted;
- `reps_completed` is therefore what happened, which is the number the session log and the
  clinical deliverable want.

Pure: no I/O, no hardware, and the clock arrives as a number. It reports which `Law` the arm
should be running and leaves commanding it to the caller, which is also what lets it be tested
without `control/` present at all.

**The safety contribution of this module is the deadman on a device-driven phase.** While the
arm is pulling the finger, advancing requires a fresh, confident reading on the current cycle;
a single missing frame stops the advance. That is the opposite of the dropout tolerance that is
correct for a patient-driven phase, where a lost reading leaves the arm static and harmless.
See `session.modes.PhasePlan.device_driven` for why, and `docs/session-orchestration.md` for
the whole design.
"""
from __future__ import annotations

from dataclasses import dataclass

from session.instructions import instruction_for
from session.modes import Driver, Law, Phase, Target, plan_for
from session.protocol import PRETENSION_FRACTION, Block


@dataclass(frozen=True)
class RepLimits:
    """What counts as started, finished, and still. Timings a clinician may want to change.

    The defaults are starting points chosen to be safe rather than brisk; none of them is a
    measured number, and the distances in metres all want checking against a real rig.
    """

    #: Where `ARMED` leaves the tether between repetitions, as a fraction of the block's cap.
    #:
    #: A fraction rather than an absolute force, because the early post-surgical caps are the
    #: clinically interesting ones and an absolute default breaks them. Two things have to be
    #: true at once and they pull in opposite directions: the resting tension has to be high
    #: enough that the slack is out, so the patient's range goes into the exercise rather than
    #: into stretching a loose tether, and low enough that a healing finger is not left holding
    #: a therapeutic force while being told to relax and get ready. A quarter of the cap
    #: satisfies both at every cap in the 0-to-450 gf band dynamic finger splints work in,
    #: which a fixed 0.5 N did not: it rejected a 60 gf prescription outright.
    #:
    #: Whatever it resolves to has to stay more than `start_delta_n` below the cap, or `RAMP`
    #: has no duration: starting and reaching the cap become the same event. `RepMachine`
    #: checks that against the block's actual cap rather than leaving it to be discovered on a
    #: finger.
    pretension_fraction: float = PRETENSION_FRACTION
    #: How much the force has to rise above the rep's baseline before the patient counts as
    #: having started. A few times the gauge reader's noise, so a jittery reading does not
    #: start a repetition on its own.
    start_delta_n: float = 0.15
    #: Minimum time an instruction is shown and spoken before a **device-driven** phase starts
    #: moving. Not cosmetic: the arm must not start pulling the instant the text appears, or
    #: the patient is being moved before they have read why.
    cue_dwell_s: float = 1.5
    #: How long to wait for a patient-driven phase to start before giving up on the repetition.
    #: Reported as an incomplete attempt, not as an abort: a patient who cannot move the finger
    #: is a clinical finding, not a fault.
    cue_timeout_s: float = 20.0
    #: Movement smaller than this does not count as advancing, for deciding the patient has
    #: reached their end of range.
    still_m: float = 0.002
    #: How long travel has to stay within `still_m` before a patient-driven work phase is
    #: treated as having reached the end of range.
    settle_s: float = 0.7
    #: How close to the rep's start position counts as back.
    home_m: float = 0.004
    #: How long a device-driven phase may go without a usable reading before the repetition is
    #: abandoned. Deliberately short, and the advance stops on the very first missing frame
    #: regardless; this is only how long to keep waiting for the reading to come back.
    blind_grace_s: float = 0.3
    #: The same, for a device-driven phase that can only ever *reduce* tension, which
    #: `RESET` in active mode is: the arm walks back toward the rep's start, bounded so it
    #: can never end up tighter than the operator rigged it.
    #:
    #: Longer than `blind_grace_s` on purpose. The reason the short grace exists is that a
    #: blind pull could keep loading a finger with no idea how hard; a motion that is
    #: structurally incapable of adding tension does not carry that risk, so the same
    #: strictness buys nothing. It costs something real, though, and this is measured rather
    #: than assumed: on the bench, the *successful* end of a reset is exactly when the tether
    #: goes slack, and a slack tether is when the camera loses the PESOLA's markers. Every
    #: repetition therefore ended by aborting a few frames after the arm had done its job,
    #: and the session never reached a second rep. See `_deadman`.
    slack_blind_grace_s: float = 3.0
    #: Longest a single phase may last before the repetition is abandoned, so nothing waits for
    #: ever on a signal that is never coming.
    phase_timeout_s: float = 60.0

    def __post_init__(self) -> None:
        for name in ("pretension_fraction", "start_delta_n", "cue_dwell_s", "cue_timeout_s",
                     "still_m", "settle_s", "home_m", "blind_grace_s",
                     "slack_blind_grace_s", "phase_timeout_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.pretension_fraction >= 1.0:
            raise ValueError(
                f"pretension_fraction must be below 1.0, got {self.pretension_fraction}: "
                "resting at or above the cap would hold the finger at the therapeutic force "
                "between repetitions"
            )


@dataclass(frozen=True)
class RepInput:
    """One cycle's worth of what the machine can observe.

    `force_n` is `None` when the gauge reader did not see the marker. It is never zero in that
    case, and this module never treats it as zero.
    """

    #: Newtons, or None when the force cannot be seen this cycle.
    force_n: float | None
    #: Seconds since the frame this reading came from.
    reading_age_s: float
    #: Arm displacement since the repetition began, in metres. Signed, positive in the slack
    #: direction, which is the direction a yielding arm moves.
    travel_m: float
    #: Monotonic seconds. Passed in rather than read, so the machine stays pure.
    now: float
    #: Whether the reading is trustworthy, if the caller knows. A `False` here is treated
    #: exactly like a missing reading, which is the point of the gauge reader reporting
    #: confidence separately from a number.
    confident: bool = True

    @property
    def usable(self) -> bool:
        return self.force_n is not None and self.confident


@dataclass(frozen=True)
class RepStep:
    """What the machine decided this cycle."""

    phase: Phase
    #: What the arm should be doing now. The caller turns this into motion through `control/`.
    law: Law
    #: What bounds the arm's travel this cycle, when the arm is the one moving. Needed
    #: alongside `law` and not derivable from it: `Law.PULL` appears in a mode's work beat
    #: aiming at `Target.ROM` and in its reset beat aiming at `Target.REP_START`, which are
    #: opposite directions, so a caller holding only the law would know the arm should pull
    #: and not which way. `control.loop.PullPlan` is what consumes this.
    target: Target
    driver: Driver
    instruction: str
    #: 1-based index of the repetition in progress, or the total once the set is done.
    rep_index: int
    reps_completed: int
    #: Why this cycle looks like it does. Goes into the session log.
    note: str
    #: Set once the block is over, whether every repetition was completed or not.
    finished: bool = False
    #: Attempts that started but did not complete, so a short set is visible rather than
    #: looking like a set that was never attempted.
    attempts_abandoned: int = 0

    @property
    def may_advance(self) -> bool:
        """Whether the arm should be moving at all this cycle."""
        return self.law in (Law.YIELD, Law.PULL)


@dataclass
class _PhaseState:
    """Bookkeeping for the phase in progress."""

    entered_at: float = 0.0
    baseline_n: float | None = None
    #: Furthest travel reached in this phase, for deciding the patient has stopped advancing.
    peak_travel_m: float = 0.0
    last_moved_at: float = 0.0
    blind_since: float | None = None


class RepMachine:
    """Runs one block's repetitions. Feed it a `RepInput` per cycle, act on the `RepStep`.

    It does not own safety: `control.safety.SafetyGate` decides HOLD / YIELD / RELEASE from the
    force, and a latched RELEASE is reported in by calling `abandon`. What this adds on top is
    the deadman on a device-driven phase and the honest counting.
    """

    def __init__(self, block: Block, limits: RepLimits | None = None):
        self.block = block
        self.limits = limits if limits is not None else RepLimits()
        self.pretension_n = block.cap_n * self.limits.pretension_fraction
        headroom = block.cap_n - self.pretension_n
        if headroom <= self.limits.start_delta_n:
            raise ValueError(
                f"a {block.cap_gf:.0f} gf cap rests at {self.pretension_n:.2f} N and leaves "
                f"only {headroom:.2f} N below the cap, which is not more than the "
                f"{self.limits.start_delta_n:.2f} N rise that counts as the patient starting: "
                "the repetition would reach the cap at the same instant it began, so RAMP "
                "would have no duration and the patient would be at the therapeutic force "
                "before being cued. The cap is too close to the gauge reader's noise floor "
                "for this exercise to be measurable; raise the cap or lower start_delta_n "
                "only if the reader is actually quieter than it assumes."
            )
        self.phase = Phase.IDLE
        self.reps_completed = 0
        self.attempts_abandoned = 0
        self._st = _PhaseState()
        self._stopped_reason: str | None = None
        #: Whether the work beat of the current attempt actually reached the cap. A repetition
        #: that never got there is not a repetition, however far the patient moved.
        self._reached_work = False

    # ------------------------------------------------------------------ state
    @property
    def rep_index(self) -> int:
        """1-based index of the repetition in progress, capped at the block's total."""
        return min(self.reps_completed + 1, self.block.reps)

    @property
    def finished(self) -> bool:
        return self.phase in (Phase.SET_DONE, Phase.ABORTED)

    @property
    def plan(self):
        return plan_for(self.block.mode_spec, self.phase)

    # ------------------------------------------------------------------ the one entry point
    def step(self, inp: RepInput) -> RepStep:
        """Advance one cycle."""
        if self.finished:
            return self._emit("the block is over", finished=True)

        if self.phase is Phase.IDLE:
            return self._enter(Phase.ARMED, inp, "taking up the slack")

        # A device-driven phase may not advance on absent evidence. Checked before anything
        # else, because every other decision below could otherwise command a pull.
        deadman = self._deadman(inp)
        if deadman is not None:
            return deadman

        if self.phase is Phase.ARMED:
            return self._armed(inp)
        if self.phase is Phase.CUEING:
            return self._cueing(inp)
        if self.phase is Phase.RAMP:
            return self._ramp(inp)
        if self.phase is Phase.WORK:
            return self._work(inp)
        if self.phase is Phase.HOLD:
            return self._hold(inp)
        if self.phase is Phase.RESET:
            return self._reset(inp)
        return self._emit("waiting")

    def abandon(self, reason: str) -> RepStep:
        """Stop the block. Terminal, and never refused.

        A latched RELEASE from the safety gate, a human stopping, and the orchestrator stopping
        the session all arrive here. The repetition in progress is counted as an abandoned
        attempt rather than silently dropped.
        """
        if not self.finished and self.phase not in (Phase.IDLE, Phase.ARMED):
            self.attempts_abandoned += 1
        self._stopped_reason = reason
        self.phase = Phase.ABORTED
        return self._emit(reason, finished=True)

    # ------------------------------------------------------------------ the deadman
    def _deadman(self, inp: RepInput) -> RepStep | None:
        """Stop a device-driven advance the moment the force stops being visible.

        A patient-driven phase tolerates a dropout, because a lost reading there leaves the arm
        static, which is harmless. In a device-driven phase the arm is pulling the finger, so a
        lost reading would leave it pulling with no idea how hard. Returns a `RepStep` that
        holds still, and abandons the attempt if the reading does not come back.

        `RESET` gets `slack_blind_grace_s` instead, which is longer. The hazard the short
        grace guards against is a blind motion that keeps *loading* a finger; a reset only
        ever walks back toward the position the repetition began at, so the tension it can
        reach is bounded by where the operator rigged it rather than by a reading nobody can
        see. The strictness is not free, and the cost showed up on the bench as a session
        that could never finish one repetition: a reset ends by letting the tether go slack,
        a slack tether is precisely when the camera loses the PESOLA's markers, so the short
        grace fired every time the reset *succeeded*.

        Note this is keyed on the phase, not on `plan.law`: active mode's reset is declared
        `Law.PULL` in `session/modes.py`, the same law as a passive-mode work pull that
        genuinely does load the finger, so the law alone cannot tell the two apart.
        """
        if not self.plan.device_driven:
            self._st.blind_since = None
            return None
        grace = (self.limits.slack_blind_grace_s if self.phase is Phase.RESET
                 else self.limits.blind_grace_s)
        stale = inp.reading_age_s > grace
        if inp.usable and not stale:
            self._st.blind_since = None
            return None

        why = "no usable force reading" if not inp.usable else (
            f"force reading {inp.reading_age_s * 1000:.0f} ms old"
        )
        if self._st.blind_since is None:
            self._st.blind_since = inp.now
        elif inp.now - self._st.blind_since > grace:
            return self.abandon(
                f"{why} for more than {grace * 1000:.0f} ms while the arm "
                "was moving the finger"
            )
        # Hold still, not release: the gate owns releasing, and this is a pause of the advance.
        return self._emit(f"{why}; holding still rather than pulling blind", law=Law.STILL)

    # ------------------------------------------------------------------ phases
    def _armed(self, inp: RepInput) -> RepStep:
        """Take the slack up to just under the cap.

        Not a formality. On a soft tether, reaching the cap takes tens of millimetres of
        stretch, and any of it left undone is range the patient spends on a rising ramp rather
        than on the exercise (`session.calibration.stretch_to_reach_m`).
        """
        if self._timed_out(inp):
            return self.abandon("could not take up the slack in time")
        if not inp.usable:
            # Arming is device-driven, so the deadman above already handled a dropout; reaching
            # here means the plan said otherwise, and holding is still the safe reading.
            return self._emit("waiting for a force reading before taking up slack", law=Law.STILL)
        assert inp.force_n is not None
        if inp.force_n >= self.pretension_n:
            return self._enter(Phase.CUEING, inp, "pre-tensioned, cueing the patient")
        return self._emit(f"taking up slack, {inp.force_n:.2f} of {self.pretension_n:.2f} N")

    def _cueing(self, inp: RepInput) -> RepStep:
        """Show the instruction and wait for the phase's driver to act.

        The two drivers wait for different things, and that is the whole of "command and
        confirm": a patient has to actually start, and a device has to wait for the patient to
        have read why it is about to move.
        """
        elapsed = inp.now - self._st.entered_at
        if self.block.mode_spec.work.driver is Driver.DEVICE:
            if elapsed < self.limits.cue_dwell_s:
                return self._emit(
                    f"instruction shown for {elapsed:.1f}s of {self.limits.cue_dwell_s:.1f}s "
                    "before the arm moves"
                )
            return self._enter(Phase.RAMP, inp, "the arm begins to move the finger")

        if elapsed > self.limits.cue_timeout_s:
            # Not an abort. A patient who cannot move the finger is a clinical finding.
            return self._abandon_attempt(
                inp, f"the patient did not start within {self.limits.cue_timeout_s:.0f}s"
            )
        if inp.usable and self._st.baseline_n is not None:
            assert inp.force_n is not None
            if inp.force_n - self._st.baseline_n >= self.limits.start_delta_n:
                return self._enter(Phase.RAMP, inp, "the patient started moving")
        return self._emit("waiting for the patient to start")

    def _ramp(self, inp: RepInput) -> RepStep:
        """Force rising toward the cap. The run-up, kept separate from the exercise.

        A ROM-targeted phase has a **second** way out, and leaving it out was a real bug: the
        prescribed stretch distance can be reached before the cap is, and then the cap is never
        reached at all, because the thing bounding the motion is the position target and the
        arm has stopped at it. Checking the cap alone, the machine sat in RAMP against a
        stationary arm until the phase timed out and the repetition was abandoned, with the
        arm having performed the stretch correctly.

        The two exits are not the same clinical event, and the note says which happened:

        - **the target**: the stroke the prescription asked for, done in full;
        - **the cap**: the finger was stiffer than the prescription assumed, so it ends short
          of the target. A meaningful finding, and the reason `rom_target_m` is a target rather
          than a guarantee.

        Both count as having reached the work phase, because in a mode where the arm does the
        moving the exercise is the stretch itself, not the act of reaching a force. The honest
        counting in `_complete_rep` rests on `_reached_work` being set only here, which both
        of these do, so a stroke that never started still cannot be counted.
        """
        if self._timed_out(inp):
            return self._abandon_attempt(inp, self._ramp_timeout_reason())
        if inp.usable:
            assert inp.force_n is not None
            if inp.force_n >= self.block.cap_n:
                self._reached_work = True
                return self._enter(Phase.WORK, inp, "at the cap")
        target = self.block.rom_target_m if self.plan.target is Target.ROM else None
        if target is not None and abs(inp.travel_m) >= target:
            self._reached_work = True
            return self._enter(
                Phase.WORK, inp, f"reached the {target * 1000:.0f} mm target under the cap"
            )
        return self._emit("force rising toward the cap")

    def _ramp_timeout_reason(self) -> str:
        """Why a ramp ran out, in the terms of whichever thing was supposed to end it.

        "The force never reached the cap" is the right sentence for a patient-driven ramp and
        the wrong one for a device-driven stretch, where the arm not having travelled is the
        actual finding and the force is a consequence.
        """
        if self.plan.target is Target.ROM:
            return "the arm did not reach the stretch target or the cap in time"
        return "the force never reached the cap"

    def _work(self, inp: RepInput) -> RepStep:
        """The exercise. Ends when the mover has gone as far as it is going to."""
        if self._timed_out(inp):
            return self._abandon_attempt(inp, "the work phase did not end in time")
        self._track_movement(inp)

        if self.plan.target is Target.ROM:
            target = self.block.rom_target_m
            assert target is not None, "a ROM-targeted phase requires rom_target_m"
            if abs(inp.travel_m) >= target:
                return self._enter(Phase.HOLD, inp, f"reached the {target * 1000:.0f} mm target")
            # The stretch can also be over because the cap stopped it, which is the other half
            # of the same deadlock `_ramp` describes: `control.loop.ControlLoop._pull_step`
            # halts a tension-raising step at the cap, and the only thing that was raising the
            # force was that step, so waiting for more travel waits forever. Ending here is
            # what turns "the finger is stiffer than the prescription assumed" into a short
            # stroke that still counts, rather than a phase timeout and a lost repetition.
            if inp.usable:
                assert inp.force_n is not None
                if inp.force_n >= self.block.cap_n:
                    return self._enter(
                        Phase.HOLD, inp,
                        f"stopped at the cap, {abs(inp.travel_m) * 1000:.0f} of "
                        f"{target * 1000:.0f} mm"
                    )
            return self._emit(
                f"stretching, {abs(inp.travel_m) * 1000:.0f} of {target * 1000:.0f} mm"
            )

        # Patient-driven: they are done when they stop advancing. Mirrors how LiveAnalyzer
        # detects a hold, which is velocity falling away rather than a clock running out.
        if inp.now - self._st.last_moved_at >= self.limits.settle_s:
            return self._enter(Phase.HOLD, inp, "the patient stopped advancing")
        return self._emit(f"following the patient, {abs(inp.travel_m) * 1000:.0f} mm")

    def _hold(self, inp: RepInput) -> RepStep:
        if inp.now - self._st.entered_at >= self.block.hold_s:
            return self._enter(Phase.RESET, inp, "hold done, returning")
        held = inp.now - self._st.entered_at
        return self._emit(f"holding, {held:.1f}s of {self.block.hold_s:.1f}s")

    def _reset(self, inp: RepInput) -> RepStep:
        """Back to where the repetition started, under whichever law the mode's reset beat uses."""
        if self._timed_out(inp):
            return self._abandon_attempt(inp, "did not return to the start position in time")
        if abs(inp.travel_m) <= self.limits.home_m:
            return self._complete_rep(inp)
        return self._emit(f"returning, {abs(inp.travel_m) * 1000:.0f} mm from the start")

    # ------------------------------------------------------------------ counting
    def _complete_rep(self, inp: RepInput) -> RepStep:
        """Count a repetition. Only ever called once the cap was actually reached.

        The honest counting the design asks for, and it is a property of the state graph rather
        than of a runtime check: `RESET` is reachable only through `HOLD`, which is reachable
        only through `WORK`, and `WORK` is entered only when the force reaches the cap. So a
        patient who moved a little and came back never arrives here at all; they time out in
        `RAMP` and become an abandoned attempt.

        An earlier version guarded this with `if not self._reached_work`, which a test showed
        could never fire. A dead safety check is worse than none, because it advertises a
        protection that is not operating, so the guarantee is asserted and documented instead.
        """
        assert self._reached_work, (
            "reached REP_DONE without passing through WORK; the phase graph is supposed to "
            "make this unreachable, so a new transition has broken the counting guarantee"
        )
        self.reps_completed += 1
        self._reached_work = False
        if self.reps_completed >= self.block.reps:
            self.phase = Phase.SET_DONE
            return self._emit(
                f"{self.reps_completed} of {self.block.reps} repetitions done", finished=True
            )
        return self._enter(
            Phase.CUEING, inp, f"repetition {self.reps_completed} of {self.block.reps} done"
        )

    def _abandon_attempt(self, inp: RepInput, why: str) -> RepStep:
        """Give up on this repetition and move on, without stopping the block.

        Distinct from `abandon`: nothing unsafe happened, the patient just did not complete the
        movement. The set still ends after the prescribed number of attempts, so a patient who
        cannot do the exercise does not loop for ever.
        """
        self.attempts_abandoned += 1
        self._reached_work = False
        attempted = self.reps_completed + self.attempts_abandoned
        if attempted >= self.block.reps:
            self.phase = Phase.SET_DONE
            return self._emit(f"{why}; ending the set", finished=True)
        return self._enter(Phase.CUEING, inp, f"{why}; moving to the next attempt")

    # ------------------------------------------------------------------ helpers
    def _enter(self, phase: Phase, inp: RepInput, note: str) -> RepStep:
        self.phase = phase
        self._st = _PhaseState(
            entered_at=inp.now,
            baseline_n=inp.force_n if inp.usable else None,
            peak_travel_m=abs(inp.travel_m),
            last_moved_at=inp.now,
        )
        return self._emit(note)

    def _track_movement(self, inp: RepInput) -> None:
        travel = abs(inp.travel_m)
        if travel > self._st.peak_travel_m + self.limits.still_m:
            self._st.peak_travel_m = travel
            self._st.last_moved_at = inp.now

    def _timed_out(self, inp: RepInput) -> bool:
        return inp.now - self._st.entered_at > self.limits.phase_timeout_s

    def _emit(self, note: str, law: Law | None = None, finished: bool = False) -> RepStep:
        plan = self.plan
        emitted = law if law is not None else plan.law
        return RepStep(
            phase=self.phase,
            law=emitted,
            # A phase held still (a dropout, say) has no travel bound to report, because
            # nothing is moving: reporting the plan's target there would describe a motion the
            # machine has just decided not to make.
            target=plan.target if emitted is Law.PULL else Target.NONE,
            driver=plan.driver,
            instruction=instruction_for(self.block.mode, self.phase, self.block.finger),
            rep_index=self.rep_index,
            reps_completed=self.reps_completed,
            note=note,
            finished=finished or self.finished,
            attempts_abandoned=self.attempts_abandoned,
        )
