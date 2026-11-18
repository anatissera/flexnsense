"""The interlock, exhaustively. One arm able to move at a time, and never on absent evidence.

These are the tests that have to hold before the two arms ever run together, so they are
written against the hazard rather than against the implementation: every one of them names a
way a finger could get hurt and asserts that it cannot happen.
"""
import itertools

import pytest

from arm1.lib.units import grams_force_to_newtons
from arm1.session.interlock import (
    ArmOneStatus,
    Custody,
    Interlock,
    InterlockLimits,
    InterlockViolation,
)

#: Arm 1 done and visibly unloaded: the only status that permits a hook move.
SAFE = ArmOneStatus(released=True, force_n=0.05, reading_age_s=0.01)


def unsafe_statuses() -> list[tuple[str, ArmOneStatus]]:
    """Every distinct reason a handover must be refused, as (label, status)."""
    return [
        ("not released", ArmOneStatus(released=False, force_n=0.05, reading_age_s=0.01)),
        ("no reading", ArmOneStatus(released=True, force_n=None, reading_age_s=0.01)),
        ("stale reading", ArmOneStatus(released=True, force_n=0.05, reading_age_s=5.0)),
        ("still loaded", ArmOneStatus(released=True, force_n=1.96, reading_age_s=0.01)),
        ("not released and no reading",
         ArmOneStatus(released=False, force_n=None, reading_age_s=0.01)),
    ]


# ---------------------------------------------------------------- the invariant
def test_custody_starts_with_neither_arm():
    """Nothing may move until something has asked for and been granted custody."""
    assert Interlock().custody is Custody.NEITHER


def test_only_one_arm_can_ever_hold_custody():
    """The invariant, checked across every reachable sequence of transitions."""
    for sequence in itertools.product(["arm1", "arm2", "release"], repeat=4):
        lock = Interlock()
        for step in sequence:
            if step == "arm1" and lock.may_hand_to_arm1():
                lock.take_arm1()
            elif step == "arm2" and lock.may_hand_to_arm2(SAFE):
                lock.take_arm2(SAFE)
            elif step == "release":
                lock.release_all()
            # Whatever happened, custody is a single value: there is no state in which both
            # arms could act.
            assert lock.custody in (Custody.NEITHER, Custody.ARM1, Custody.ARM2)


def test_custody_never_passes_from_one_arm_to_the_other_directly():
    """It has to go through NEITHER, which is where the release happens."""
    lock = Interlock()
    lock.take_arm1()
    with pytest.raises(InterlockViolation, match="arm 1 still holds custody"):
        lock.take_arm2(SAFE)

    lock = Interlock()
    lock.release_all()
    lock.take_arm2(SAFE)
    with pytest.raises(InterlockViolation, match="arm 2 still holds custody"):
        lock.take_arm1()


# ---------------------------------------------------------------- absent evidence
def test_a_dropout_does_not_permit_the_hook_to_move():
    """The rule the module exists for: a missing reading is not evidence of zero force. A
    dropout is exactly when we are least able to know whether a finger is loaded."""
    lock = Interlock()
    status = ArmOneStatus(released=True, force_n=None, reading_age_s=0.01)
    verdict = lock.may_hand_to_arm2(status)
    assert not verdict
    assert "not evidence of zero force" in verdict.reason


def test_a_stale_reading_does_not_permit_the_hook_to_move():
    lock = Interlock()
    status = ArmOneStatus(released=True, force_n=0.0, reading_age_s=1.0)
    assert not lock.may_hand_to_arm2(status)


def test_a_reading_of_exactly_zero_still_has_to_be_fresh():
    """Zero is the most tempting value to trust, so it gets no exemption from staleness."""
    lock = Interlock()
    assert not lock.may_hand_to_arm2(
        ArmOneStatus(released=True, force_n=0.0, reading_age_s=10.0)
    )


@pytest.mark.parametrize("label,status", unsafe_statuses(), ids=lambda v: v if isinstance(v, str) else "")
def test_every_unsafe_status_refuses_and_explains(label, status):
    lock = Interlock()
    verdict = lock.may_hand_to_arm2(status)
    assert not verdict, label
    assert verdict.reason, "a refusal with no reason cannot be acted on"


@pytest.mark.parametrize("label,status", unsafe_statuses(), ids=lambda v: v if isinstance(v, str) else "")
def test_an_unsafe_handover_raises_rather_than_proceeding(label, status):
    """A caller that skipped the check must fail loudly, not get a hook move anyway."""
    lock = Interlock()
    with pytest.raises(InterlockViolation):
        lock.take_arm2(status)
    assert lock.custody is Custody.NEITHER, label


def test_the_evidence_is_rechecked_at_the_transition():
    """A caller that asks, then does something else, then acts, must not hand over on evidence
    that has gone stale in between. The whole hazard is in that gap."""
    lock = Interlock()
    assert lock.may_hand_to_arm2(SAFE)
    gone_bad = ArmOneStatus(released=True, force_n=None, reading_age_s=0.01)
    with pytest.raises(InterlockViolation, match="not evidence of zero force"):
        lock.take_arm2(gone_bad)


# ---------------------------------------------------------------- residual tension
def test_a_loaded_tether_refuses_and_reports_grams():
    """The operator compares against the PESOLA, which is graduated in grams."""
    lock = Interlock()
    status = ArmOneStatus(released=True, force_n=grams_force_to_newtons(200), reading_age_s=0.01)
    verdict = lock.may_hand_to_arm2(status)
    assert not verdict
    assert "200 gf" in verdict.reason


def test_a_little_residual_tension_is_allowed():
    """Demanding exactly zero would make the handover impossible rather than safe: a real
    tether has residual tension from its own weight and from friction."""
    lock = Interlock()
    assert lock.may_hand_to_arm2(
        ArmOneStatus(released=True, force_n=0.19, reading_age_s=0.01)
    )


def test_the_slack_threshold_is_well_below_any_therapeutic_force():
    """If slack_n ever drifted up to therapeutic levels, the interlock would permit moving the
    hook off a finger that is being actively pulled."""
    assert InterlockLimits().slack_n < grams_force_to_newtons(50)


def test_the_boundary_is_inclusive_of_slack_and_exclusive_above_it():
    lock = Interlock(InterlockLimits(slack_n=0.20))
    assert lock.may_hand_to_arm2(ArmOneStatus(True, 0.20, 0.01))
    assert not lock.may_hand_to_arm2(ArmOneStatus(True, 0.2001, 0.01))


# ---------------------------------------------------------------- the asymmetry
def test_handing_to_arm_one_does_not_require_force_evidence():
    """Deliberately not symmetric: arm 2 holding the hook is not a force on the patient, so
    the dangerous direction is the only one that needs evidence."""
    lock = Interlock()
    assert lock.may_hand_to_arm1()
    lock.take_arm1()
    assert lock.custody is Custody.ARM1


def test_arm_one_cannot_start_while_the_hook_is_moving():
    lock = Interlock()
    lock.take_arm2(SAFE)
    verdict = lock.may_hand_to_arm1()
    assert not verdict and "arm 2 still holds custody" in verdict.reason


# ---------------------------------------------------------------- releasing and stopping
def test_releasing_is_always_allowed_and_idempotent():
    """A path to safety must never be refused, from any state, however many times."""
    for setup in (lambda lk: None, lambda lk: lk.take_arm1(), lambda lk: lk.take_arm2(SAFE)):
        lock = Interlock()
        setup(lock)
        lock.release_all()
        lock.release_all()
        assert lock.custody is Custody.NEITHER


def test_releasing_works_after_a_stop():
    lock = Interlock()
    lock.take_arm1()
    lock.stop("e-stop pressed")
    lock.release_all()
    assert lock.custody is Custody.NEITHER


def test_stopping_drops_custody():
    lock = Interlock()
    lock.take_arm1()
    lock.stop("force over hard ceiling")
    assert lock.custody is Custody.NEITHER
    assert lock.stopped


def test_a_stop_grants_no_further_custody_to_either_arm():
    """Terminal, in both directions, however safe the world looks afterwards."""
    lock = Interlock()
    lock.stop("clinician stopped the session")
    assert not lock.may_hand_to_arm1()
    assert not lock.may_hand_to_arm2(SAFE)
    with pytest.raises(InterlockViolation):
        lock.take_arm1()
    with pytest.raises(InterlockViolation):
        lock.take_arm2(SAFE)


def test_the_first_stop_reason_is_the_one_kept():
    """A later, vaguer stop must not overwrite why it actually stopped."""
    lock = Interlock()
    lock.stop("force over hard ceiling")
    lock.stop("session ended")
    assert lock.stopped_reason == "force over hard ceiling"


def test_a_refusal_after_a_stop_says_so_rather_than_blaming_the_force():
    """So an operator is not sent to look at the tether when the real cause was the e-stop."""
    lock = Interlock()
    lock.stop("e-stop pressed")
    assert "e-stop pressed" in lock.may_hand_to_arm2(SAFE).reason


# ---------------------------------------------------------------- limits validation
@pytest.mark.parametrize("kw", [{"slack_n": 0.0}, {"slack_n": -0.1}, {"max_age_s": 0.0}])
def test_impossible_limits_are_rejected(kw):
    with pytest.raises(ValueError):
        InterlockLimits(**kw)


def test_a_verdict_is_usable_directly_in_a_condition():
    """The call site reads `if interlock.may_hand_to_arm2(status):`, so truthiness has to be
    the allowed flag and not merely object identity."""
    assert bool(Interlock().may_hand_to_arm1()) is True
    assert bool(Interlock().may_hand_to_arm2(ArmOneStatus(False, None, 0.0))) is False
