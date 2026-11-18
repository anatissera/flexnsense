import pytest

from arm1.control.safety import Action, SafetyGate, SafetyLimits

THRESHOLD = 1.96   # 200 gf, the clinical cap used throughout the docs
CEILING = 3.0


def limits(**kw) -> SafetyLimits:
    base = dict(f_threshold_n=THRESHOLD, f_max_n=CEILING, max_age_s=0.1, max_dropouts=2,
                max_travel_m=0.05)
    base.update(kw)
    return SafetyLimits(**base)


def gate(**kw) -> SafetyGate:
    return SafetyGate(limits(**kw))


# --------------------------------------------------------------------------- normal operation

def test_under_cap_holds():
    d = gate().check(force_n=1.0, age_s=0.01)
    assert d.action is Action.HOLD
    assert not d.may_move


def test_over_cap_yields():
    d = gate().check(force_n=2.5, age_s=0.01)
    assert d.action is Action.YIELD
    assert d.may_move
    assert d.force_n == 2.5


def test_exactly_at_cap_holds_not_yields():
    """The cap is the force the arm is allowed to hold, so equality is not an overshoot."""
    assert gate().check(force_n=THRESHOLD, age_s=0.01).action is Action.HOLD


def test_just_over_cap_yields():
    assert gate().check(force_n=THRESHOLD + 1e-9, age_s=0.01).action is Action.YIELD


# --------------------------------------------------------------------------- hard ceiling

def test_over_hard_ceiling_releases():
    d = gate().check(force_n=CEILING + 0.01, age_s=0.01)
    assert d.action is Action.RELEASE
    assert "ceiling" in d.reason


def test_exactly_at_ceiling_still_yields():
    """f_max is the limit that must not be exceeded; at it the arm is still yielding, which is
    what brings the force back down."""
    assert gate().check(force_n=CEILING, age_s=0.01).action is Action.YIELD


def test_ceiling_release_latches_even_if_force_recovers():
    g = gate()
    assert g.check(force_n=CEILING + 1.0, age_s=0.01).action is Action.RELEASE
    # A later, perfectly safe reading must not restart the pull.
    assert g.check(force_n=0.5, age_s=0.01).action is Action.RELEASE


# --------------------------------------------------------------------------- dropouts

def test_single_dropout_holds_does_not_release():
    """One lost frame at 30 Hz is 33 ms and routine; releasing on it would make the device
    unusable (docs/admittance-design.md)."""
    d = gate().check(force_n=None, age_s=0.01)
    assert d.action is Action.HOLD
    assert d.force_n is None


def test_dropouts_within_tolerance_hold():
    g = gate(max_dropouts=2)
    assert g.check(force_n=None, age_s=0.01).action is Action.HOLD
    assert g.check(force_n=None, age_s=0.01).action is Action.HOLD


def test_dropouts_past_tolerance_release():
    g = gate(max_dropouts=2)
    g.check(force_n=None, age_s=0.01)
    g.check(force_n=None, age_s=0.01)
    d = g.check(force_n=None, age_s=0.01)   # the third exceeds max_dropouts=2
    assert d.action is Action.RELEASE
    assert "without a reading" in d.reason


def test_a_good_reading_resets_the_dropout_count():
    """Isolated dropouts must not accumulate across a long session into a spurious release."""
    g = gate(max_dropouts=2)
    for _ in range(10):
        assert g.check(force_n=None, age_s=0.01).action is Action.HOLD
        assert g.check(force_n=1.0, age_s=0.01).action is Action.HOLD
    assert g.latched is None


def test_zero_dropout_tolerance_releases_on_the_first():
    g = gate(max_dropouts=0)
    assert g.check(force_n=None, age_s=0.01).action is Action.RELEASE


def test_dropout_during_yield_does_not_keep_moving():
    """The dangerous case: over the cap, then the marker is lost. Must not keep yielding on a
    force it can no longer see."""
    g = gate()
    assert g.check(force_n=2.5, age_s=0.01).action is Action.YIELD
    d = g.check(force_n=None, age_s=0.01)
    assert d.action is Action.HOLD
    assert not d.may_move


# --------------------------------------------------------------------------- staleness

def test_stale_reading_releases():
    d = gate(max_age_s=0.1).check(force_n=1.0, age_s=0.5)
    assert d.action is Action.RELEASE
    assert "stale" in d.reason


def test_stale_beats_a_safe_looking_force():
    """A low force from a frozen camera is not evidence of a low force now."""
    d = gate(max_age_s=0.1).check(force_n=0.0, age_s=10.0)
    assert d.action is Action.RELEASE


def test_age_exactly_at_limit_is_accepted():
    assert gate(max_age_s=0.1).check(force_n=1.0, age_s=0.1).action is Action.HOLD


# --------------------------------------------------------------------------- travel limit

def test_travel_over_limit_releases():
    d = gate(max_travel_m=0.05).check(force_n=1.0, age_s=0.01, travel_m=0.06)
    assert d.action is Action.RELEASE
    assert "travel" in d.reason


def test_travel_limit_applies_in_both_directions():
    d = gate(max_travel_m=0.05).check(force_n=1.0, age_s=0.01, travel_m=-0.06)
    assert d.action is Action.RELEASE


def test_travel_within_limit_is_fine():
    assert gate(max_travel_m=0.05).check(
        force_n=2.5, age_s=0.01, travel_m=0.04
    ).action is Action.YIELD


# --------------------------------------------------------------------------- latching

def test_release_latches_across_everything():
    g = gate()
    g.check(force_n=1.0, age_s=99.0)             # stale -> release
    for force in (0.0, 1.0, 2.5, None):
        assert g.check(force_n=force, age_s=0.01).action is Action.RELEASE


def test_reset_clears_the_latch():
    g = gate()
    g.check(force_n=1.0, age_s=99.0)
    assert g.check(force_n=1.0, age_s=0.01).action is Action.RELEASE
    g.reset()
    assert g.check(force_n=1.0, age_s=0.01).action is Action.HOLD


def test_reset_also_clears_dropout_count():
    g = gate(max_dropouts=2)
    g.check(force_n=None, age_s=0.01)
    g.check(force_n=None, age_s=0.01)
    g.reset()
    assert g.consecutive_dropouts == 0
    assert g.check(force_n=None, age_s=0.01).action is Action.HOLD


# --------------------------------------------------------------------------- limit validation

def test_threshold_must_be_below_ceiling():
    with pytest.raises(ValueError, match="f_threshold_n"):
        SafetyLimits(f_threshold_n=3.0, f_max_n=2.0)


def test_threshold_must_be_positive():
    with pytest.raises(ValueError, match="f_threshold_n"):
        SafetyLimits(f_threshold_n=0.0, f_max_n=2.0)


def test_negative_max_age_rejected():
    with pytest.raises(ValueError, match="max_age_s"):
        SafetyLimits(f_threshold_n=1.0, f_max_n=2.0, max_age_s=-1.0)


def test_negative_dropout_tolerance_rejected():
    with pytest.raises(ValueError, match="max_dropouts"):
        SafetyLimits(f_threshold_n=1.0, f_max_n=2.0, max_dropouts=-1)


def test_non_positive_travel_limit_rejected():
    with pytest.raises(ValueError, match="max_travel_m"):
        SafetyLimits(f_threshold_n=1.0, f_max_n=2.0, max_travel_m=0.0)


# ---------------------------------------------------------------- anticipation_n
def test_anticipation_defaults_to_zero_and_yields_at_the_literal_cap():
    """Backwards compatible: every test above this constructs limits() with no
    anticipation_n at all, and must see the exact same cap-crossing behaviour as before this
    field existed."""
    g = gate()
    at_cap = g.check(THRESHOLD + 0.001, age_s=0.0)
    just_under = g.check(THRESHOLD - 0.001, age_s=0.0)
    assert at_cap.action is Action.YIELD
    assert just_under.action is Action.HOLD


def test_anticipation_moves_the_yield_point_earlier():
    g = gate(anticipation_n=0.2)
    below_the_old_cap_but_past_the_new_one = g.check(THRESHOLD - 0.1, age_s=0.0)
    assert below_the_old_cap_but_past_the_new_one.action is Action.YIELD
    still_below_both = g.check(THRESHOLD - 0.3, age_s=0.0)
    assert still_below_both.action is Action.HOLD


def test_yield_from_n_is_the_cap_minus_the_anticipation():
    l = limits(anticipation_n=0.2)
    assert l.yield_from_n == pytest.approx(THRESHOLD - 0.2)


def test_zero_anticipation_leaves_yield_from_n_at_the_cap():
    assert limits().yield_from_n == THRESHOLD


def test_the_yield_reason_names_both_numbers_when_anticipating():
    g = gate(anticipation_n=0.2)
    d = g.check(THRESHOLD - 0.1, age_s=0.0)
    assert "yield start" in d.reason
    assert "anticipating" in d.reason
    assert f"{THRESHOLD:.2f}" in d.reason


def test_the_yield_reason_is_unchanged_with_no_anticipation():
    """The exact pre-existing wording, so nothing downstream parsing this message breaks for
    the (default, most common) case of no anticipation at all."""
    d = gate().check(THRESHOLD + 0.1, age_s=0.0)
    assert d.reason == f"force {THRESHOLD + 0.1:.2f} N over cap {THRESHOLD:.2f} N"


def test_anticipation_cannot_reach_or_exceed_the_cap():
    """It is a margin below the cap, not a second, lower cap: letting it reach f_threshold_n
    would mean yielding starts at force zero, and past it would be a negative threshold."""
    with pytest.raises(ValueError, match="anticipation_n"):
        limits(anticipation_n=THRESHOLD)
    with pytest.raises(ValueError, match="anticipation_n"):
        limits(anticipation_n=THRESHOLD + 0.5)


def test_anticipation_cannot_be_negative():
    with pytest.raises(ValueError, match="anticipation_n"):
        limits(anticipation_n=-0.1)
