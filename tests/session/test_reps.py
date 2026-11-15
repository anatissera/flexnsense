"""The rep machine: honest counting, and the deadman on a device-driven phase.

Two groups matter more than the rest. "counting is honest" is the clinical deliverable: a
repetition that was not really done must not be counted. "the deadman" is the safety one: while
the arm is pulling a finger, a lost reading stops the advance on the very first frame.
"""
import pytest

from lib.units import grams_force_to_newtons
from session.modes import Driver, Law, Phase
from session.protocol import Block
from session.reps import RepInput, RepLimits, RepMachine

CAP_N = grams_force_to_newtons(200)
#: What ARMED drives towards: the resting tension between repetitions, a fraction of the cap so
#: RAMP is a real phase and the finger is not held at a therapeutic force while being cued.
PRETENSIONED = CAP_N * RepLimits().pretension_fraction
#: Started moving, but still below the cap. This is the RAMP band.
RAMPING = PRETENSIONED + RepLimits().start_delta_n + 0.01


def active(reps=2, **kw) -> Block:
    return Block(finger="index", mode="active", reps=reps, cap_n=CAP_N, hold_s=0.5, **kw)


def passive(reps=2, rom_target_m=0.025, **kw) -> Block:
    return Block(finger="index", mode="passive", reps=reps, cap_n=CAP_N, hold_s=0.5,
                 rom_target_m=rom_target_m, **kw)


def inp(force_n=PRETENSIONED, now=0.0, travel_m=0.0, age=0.01, confident=True) -> RepInput:
    return RepInput(force_n=force_n, reading_age_s=age, travel_m=travel_m, now=now,
                    confident=confident)


class Clock:
    """Advances time so a test reads as a sequence of events rather than of timestamps."""

    def __init__(self, machine: RepMachine):
        self.m = machine
        self.t = 0.0

    def step(self, dt=0.033, **kw):
        self.t += dt
        return self.m.step(inp(now=self.t, **kw))

    def until(self, predicate, dt=0.033, limit=2000, **kw):
        """Step until `predicate(step)`, so a test does not hardcode a cycle count."""
        for _ in range(limit):
            s = self.step(dt=dt, **kw)
            if predicate(s):
                return s
        raise AssertionError(f"never satisfied; stuck in {self.m.phase.value}")

    def arm_and_cue(self):
        """Through ARMED into CUEING, which every repetition starts from."""
        return self.until(lambda s: s.phase is Phase.CUEING)


# ---------------------------------------------------------------- the skeleton
def test_a_block_starts_by_taking_up_the_slack():
    """Not a formality: on a soft tether, reaching the cap costs tens of millimetres of range,
    and leaving it undone spends the patient's range on a ramp instead of on the exercise."""
    m = RepMachine(active())
    step = m.step(inp(force_n=0.0))
    assert step.phase is Phase.ARMED
    assert step.law is Law.PULL and step.driver is Driver.DEVICE


def test_arming_ends_at_the_resting_tension_not_at_the_cap():
    """Pre-tensioning all the way to the cap would leave a healing finger holding a therapeutic
    force while being told to relax and get ready."""
    m = RepMachine(active())
    c = Clock(m)
    c.step(force_n=0.1)
    assert m.phase is Phase.ARMED
    step = c.step(force_n=PRETENSIONED)
    assert step.phase is Phase.CUEING
    assert PRETENSIONED < CAP_N


def test_an_active_rep_runs_cue_ramp_work_hold_reset():
    m = RepMachine(active(reps=1))
    c = Clock(m)
    c.arm_and_cue()
    # The patient starts: force rises past the start threshold.
    seen = [c.step(force_n=RAMPING).phase]
    seen.append(c.step(force_n=CAP_N).phase)                      # at the cap -> WORK
    seen.append(c.until(lambda s: s.phase is Phase.HOLD,
                        force_n=CAP_N, travel_m=0.01).phase)       # stops advancing
    seen.append(c.until(lambda s: s.phase is not Phase.HOLD,
                        force_n=CAP_N, travel_m=0.01).phase)       # hold elapses -> RESET
    assert seen == [Phase.RAMP, Phase.WORK, Phase.HOLD, Phase.RESET]


def test_a_passive_rep_reaches_the_rom_target_then_holds():
    m = RepMachine(passive(reps=1, rom_target_m=0.02))
    c = Clock(m)
    c.arm_and_cue()
    # Device-driven: the cue dwells, then the arm moves without the patient doing anything.
    c.until(lambda s: s.phase is Phase.RAMP, force_n=PRETENSIONED)
    c.step(force_n=CAP_N)
    assert m.phase is Phase.WORK
    step = c.until(lambda s: s.phase is Phase.HOLD, force_n=CAP_N, travel_m=0.02)
    assert "target" in step.note


# ---------------------------------------------------------------- command and confirm
def test_a_patient_driven_cue_waits_for_the_patient_not_for_a_clock():
    """The resolution of the two rep models: the timeline is a cue, the signal is the counter."""
    m = RepMachine(active())
    c = Clock(m)
    c.arm_and_cue()
    for _ in range(100):                       # plenty of time passes
        step = c.step(force_n=PRETENSIONED)
        assert step.phase is Phase.CUEING
    assert c.step(force_n=RAMPING).phase is Phase.RAMP


def test_a_device_driven_cue_waits_before_moving_the_patient():
    """The arm must not start pulling the instant the text appears, or the patient is being
    moved before they have read why."""
    m = RepMachine(passive(), RepLimits(cue_dwell_s=1.0))
    c = Clock(m)
    c.arm_and_cue()
    step = c.step(force_n=PRETENSIONED)
    assert step.phase is Phase.CUEING and step.law is Law.STILL
    step = c.until(lambda s: s.phase is Phase.RAMP, force_n=PRETENSIONED)
    assert c.t >= 1.0


def test_noise_does_not_start_a_repetition():
    """start_delta_n exists so a jittery reading does not count as the patient moving."""
    m = RepMachine(active(), RepLimits(start_delta_n=0.15))
    c = Clock(m)
    c.arm_and_cue()
    for jitter in (0.05, -0.04, 0.09, -0.02, 0.14):
        assert c.step(force_n=PRETENSIONED + jitter).phase is Phase.CUEING


# ---------------------------------------------------------------- counting is honest
def test_a_completed_rep_is_counted():
    m = RepMachine(active(reps=2))
    c = Clock(m)
    _run_active_rep(c)
    assert m.reps_completed == 1
    assert m.phase is Phase.CUEING          # straight on to the next


def test_a_rep_that_never_reached_the_cap_is_not_counted():
    """The honest counting the clinical deliverable rests on: a patient who moved a little and
    never got to the cap has not done a repetition, and counting it would make the session look
    better than it was."""
    m = RepMachine(active(reps=2), RepLimits(phase_timeout_s=3.0))
    c = Clock(m)
    c.arm_and_cue()
    c.step(force_n=RAMPING)                                 # started
    assert m.phase is Phase.RAMP
    # Stays in the ramp band and never reaches the cap.
    step = c.until(lambda s: s.phase is Phase.CUEING, dt=0.2, force_n=RAMPING, travel_m=0.0)
    assert m.reps_completed == 0
    assert m.attempts_abandoned == 1
    assert "never reached the cap" in step.note


def test_counting_is_guaranteed_by_the_phase_graph_not_by_a_check():
    """Why the guard in _complete_rep was removed: WORK is the only way into HOLD, and HOLD the
    only way into RESET, so a rep cannot be counted without having reached the cap. A test
    showed the runtime guard could never fire, and a dead safety check advertises a protection
    that is not operating."""
    m = RepMachine(active(reps=1))
    c = Clock(m)
    c.arm_and_cue()
    c.step(force_n=RAMPING)
    # Returning home while still in RAMP does not reach RESET at all.
    for _ in range(20):
        step = c.step(force_n=RAMPING, travel_m=0.0)
        assert step.phase is Phase.RAMP
        assert m.reps_completed == 0


def test_a_patient_who_never_starts_is_an_abandoned_attempt_not_an_abort():
    """A patient who cannot move the finger is a clinical finding, not a fault."""
    m = RepMachine(active(reps=2), RepLimits(cue_timeout_s=5.0))
    c = Clock(m)
    c.arm_and_cue()
    c.until(lambda s: s.attempts_abandoned == 1, dt=0.5, force_n=PRETENSIONED)
    assert m.phase is not Phase.ABORTED
    assert m.reps_completed == 0


def test_a_patient_who_never_starts_does_not_loop_for_ever():
    """The set still ends after the prescribed number of attempts."""
    m = RepMachine(active(reps=3), RepLimits(cue_timeout_s=2.0))
    c = Clock(m)
    c.arm_and_cue()
    step = c.until(lambda s: s.finished, dt=0.5, force_n=PRETENSIONED)
    assert m.phase is Phase.SET_DONE
    assert m.reps_completed == 0 and m.attempts_abandoned == 3
    assert step.finished


def test_the_set_ends_after_the_prescribed_reps():
    m = RepMachine(active(reps=2))
    c = Clock(m)
    _run_active_rep(c)
    step = _run_active_rep(c)
    assert m.reps_completed == 2
    assert m.phase is Phase.SET_DONE and step.finished


def test_a_short_set_reports_both_numbers():
    """So a session log distinguishes 'did six' from 'was asked for six'."""
    m = RepMachine(active(reps=2), RepLimits(cue_timeout_s=2.0))
    c = Clock(m)
    _run_active_rep(c)
    step = c.until(lambda s: s.finished, dt=0.5, force_n=PRETENSIONED)
    assert (step.reps_completed, step.attempts_abandoned) == (1, 1)


# ---------------------------------------------------------------- the deadman
def test_a_device_driven_advance_stops_on_the_first_missing_frame():
    """While the arm pulls the finger, a lost reading would leave it pulling with no idea how
    hard. This is a deadman, not the dropout tolerance a patient-driven phase gets."""
    m = RepMachine(passive())
    c = Clock(m)
    c.arm_and_cue()
    c.until(lambda s: s.phase is Phase.WORK, force_n=CAP_N)
    step = c.step(force_n=None)
    assert step.law is Law.STILL
    assert not step.may_advance
    assert "holding still rather than pulling blind" in step.note


def test_a_low_confidence_reading_counts_as_missing():
    """The gauge reader reports confidence separately from a number precisely so this works."""
    m = RepMachine(passive())
    c = Clock(m)
    c.arm_and_cue()
    c.until(lambda s: s.phase is Phase.WORK, force_n=CAP_N)
    assert c.step(force_n=CAP_N, confident=False).law is Law.STILL


def test_a_stale_reading_stops_a_device_driven_advance():
    m = RepMachine(passive())
    c = Clock(m)
    c.arm_and_cue()
    c.until(lambda s: s.phase is Phase.WORK, force_n=CAP_N)
    assert c.step(force_n=CAP_N, age=5.0).law is Law.STILL


def test_the_advance_resumes_if_the_reading_comes_straight_back():
    """A single dropped frame must not end the exercise, only pause the advance."""
    m = RepMachine(passive())
    c = Clock(m)
    c.arm_and_cue()
    c.until(lambda s: s.phase is Phase.WORK, force_n=CAP_N)
    assert c.step(force_n=None).law is Law.STILL
    step = c.step(force_n=CAP_N)
    assert step.law is Law.PULL and step.may_advance


def test_a_persistent_dropout_abandons_a_device_driven_attempt():
    m = RepMachine(passive(), RepLimits(blind_grace_s=0.2))
    c = Clock(m)
    c.arm_and_cue()
    c.until(lambda s: s.phase is Phase.WORK, force_n=CAP_N)
    step = c.until(lambda s: s.finished, dt=0.1, force_n=None)
    assert m.phase is Phase.ABORTED
    assert "while the arm was moving the finger" in step.note


def test_a_patient_driven_phase_tolerates_a_dropout():
    """The asymmetry: a lost reading here leaves the arm static, which is harmless, so the
    exercise must not be abandoned for it."""
    m = RepMachine(active())
    c = Clock(m)
    c.arm_and_cue()
    c.step(force_n=RAMPING)
    c.step(force_n=CAP_N)
    assert m.phase is Phase.WORK
    for _ in range(30):
        step = c.step(force_n=None, travel_m=0.01)
        assert not step.finished
    assert m.phase in (Phase.WORK, Phase.HOLD)


# ---------------------------------------------------------------- stopping
def test_abandoning_is_terminal_and_counts_the_attempt():
    m = RepMachine(active(reps=3))
    c = Clock(m)
    c.arm_and_cue()
    c.step(force_n=RAMPING)
    step = m.abandon("gate latched RELEASE")
    assert m.phase is Phase.ABORTED and step.finished
    assert m.attempts_abandoned == 1
    assert step.instruction == "Stop and relax your hand"


def test_abandoning_before_a_rep_started_counts_nothing():
    m = RepMachine(active())
    m.step(inp(force_n=0.0))
    m.abandon("e-stop")
    assert m.attempts_abandoned == 0


def test_nothing_advances_after_abandoning():
    m = RepMachine(active())
    c = Clock(m)
    c.arm_and_cue()
    m.abandon("e-stop")
    for _ in range(10):
        step = c.step(force_n=CAP_N)
        assert step.finished and step.phase is Phase.ABORTED


# ---------------------------------------------------------------- what the UI reads
def test_every_step_carries_an_instruction_for_the_phase():
    m = RepMachine(passive())
    c = Clock(m)
    for _ in range(60):
        step = c.step(force_n=CAP_N, travel_m=0.001)
        assert step.instruction, f"no instruction in {step.phase.value}"


def test_the_driver_reported_matches_the_phase_not_the_mode():
    """What the dashboard keys off: it flips within a repetition."""
    m = RepMachine(active(reps=1))
    c = Clock(m)
    _run_active_rep(c)
    # The reset beat of an active rep is device-driven, unlike its work beat.
    assert m.block.mode_spec.work.driver is Driver.PATIENT
    assert m.block.mode_spec.reset.driver is Driver.DEVICE


@pytest.mark.parametrize("kw", [{"start_delta_n": 0.0}, {"cue_dwell_s": -1.0},
                                {"settle_s": 0.0}, {"home_m": -0.001}])
def test_impossible_limits_are_rejected(kw):
    with pytest.raises(ValueError):
        RepLimits(**kw)


# ---------------------------------------------------------------- helper
def _run_active_rep(c: Clock):
    """One complete active repetition: start, reach the cap, settle, hold, return."""
    c.until(lambda s: s.phase is Phase.CUEING, force_n=PRETENSIONED)
    c.step(force_n=RAMPING)
    c.until(lambda s: s.phase is Phase.WORK, force_n=CAP_N)
    c.until(lambda s: s.phase is Phase.HOLD, force_n=CAP_N, travel_m=0.01)
    c.until(lambda s: s.phase is Phase.RESET, force_n=CAP_N, travel_m=0.01)
    return c.until(lambda s: s.phase in (Phase.CUEING, Phase.SET_DONE),
                   force_n=PRETENSIONED, travel_m=0.0)


def test_a_pretension_that_erases_the_ramp_is_rejected_up_front():
    """The bug a test found rather than a finger: resting within start_delta_n of the cap makes
    starting and reaching the cap the same event, so RAMP has no duration and the patient is at
    the therapeutic force before being cued."""
    with pytest.raises(ValueError, match="no duration"):
        RepMachine(active(), RepLimits(pretension_fraction=0.97, start_delta_n=0.15))


def test_resting_at_or_above_the_cap_is_rejected():
    with pytest.raises(ValueError, match="below 1.0"):
        RepLimits(pretension_fraction=1.0)


@pytest.mark.parametrize("cap_gf", [40, 50, 60, 100, 200, 300, 450])
def test_the_whole_clinical_cap_range_works(cap_gf):
    """The reason pre-tension is a fraction of the cap and not an absolute force: the early
    post-surgical caps are the clinically interesting ones, and a fixed 0.5 N rejected a 60 gf
    prescription outright. Dynamic finger splints work in the 0-to-450 gf band."""
    m = RepMachine(Block(finger="index", mode="active", reps=2,
                         cap_n=grams_force_to_newtons(cap_gf), hold_s=0.5))
    assert 0 < m.pretension_n < grams_force_to_newtons(cap_gf)
    # And the ramp is a real phase at every one of them.
    assert grams_force_to_newtons(cap_gf) - m.pretension_n > RepLimits().start_delta_n


def test_the_resting_tension_scales_with_the_cap():
    """A gentler prescription rests gentler, rather than resting at a fixed force that could be
    most of a low cap."""
    gentle = RepMachine(Block(finger="index", mode="active", reps=2,
                              cap_n=grams_force_to_newtons(50), hold_s=0.5))
    firm = RepMachine(Block(finger="index", mode="active", reps=2,
                            cap_n=grams_force_to_newtons(400), hold_s=0.5))
    assert gentle.pretension_n < firm.pretension_n
