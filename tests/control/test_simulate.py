import pytest

from control.admittance import AdmittanceParams
from control.safety import SafetyLimits
from control.simulate import simulate

SPRING = 53.9


def limits(**kw) -> SafetyLimits:
    base = dict(f_threshold_n=1.96, f_max_n=3.0, max_travel_m=0.15)
    base.update(kw)
    return SafetyLimits(**base)


def params(damping: float, **kw) -> AdmittanceParams:
    base = dict(damping_n_s_per_m=damping, v_max_m_s=0.030, deadband_n=0.05)
    base.update(kw)
    return AdmittanceParams(**base)


def test_no_pull_never_reaches_the_cap():
    r = simulate(limits(), params(10), pull_speed_m_s=0.0)
    assert r.peak_force_n == 0.0
    assert not r.released


def test_slow_pull_is_tracked_without_releasing():
    r = simulate(limits(), params(10), pull_speed_m_s=0.004, duration_s=20)
    assert not r.released
    assert r.peak_force_n < 2.1          # barely above the 1.96 N cap
    assert r.arm_travel_m > 0.0          # the arm did follow


def test_overshoot_matches_b_times_pull_speed():
    """The relationship the design doc relies on to bound B: to give way at speed v the law
    needs an overshoot of B*v, so that is where the force settles."""
    cap = limits().f_threshold_n
    for b, pull in ((10, 0.004), (20, 0.004), (10, 0.010)):
        r = simulate(limits(), params(b), pull_speed_m_s=pull, duration_s=20)
        predicted = cap + b * pull
        assert r.peak_force_n == pytest.approx(predicted, abs=0.1), (
            f"B={b}, pull={pull}: peak {r.peak_force_n:.2f} N, predicted {predicted:.2f} N"
        )


def test_high_damping_cannot_keep_up_with_a_brisk_pull():
    """B=100 needs 2 N of overshoot to match 20 mm/s, which the 1.04 N headroom cannot give,
    so the ceiling fires. This is why the design doc says 10-20, not 100."""
    r = simulate(limits(), params(100), pull_speed_m_s=0.020)
    assert r.released
    assert "ceiling" in r.release_reason


def test_low_damping_keeps_the_same_pull_under_the_ceiling():
    r = simulate(limits(), params(10), pull_speed_m_s=0.020)
    assert r.peak_force_n < 2.5          # well under the 3.0 N ceiling
    # It still ends, but on travel rather than force.
    assert not r.released or "travel" in r.release_reason


def test_a_flinch_reaches_the_ceiling_at_any_damping():
    """33 ms at 50 mm/s is 1.7 mm of stretch, ~0.09 N arriving between frames. No B absorbs
    that; the gate releasing is the designed response, not a tuning failure."""
    for b in (2, 5, 10, 20, 50):
        r = simulate(limits(), params(b), pull_speed_m_s=0.050)
        assert r.released, f"B={b} unexpectedly survived a 50 mm/s yank"
        assert "ceiling" in r.release_reason


def test_a_long_slow_pull_ends_on_the_travel_limit():
    """Following a steady pull walks the arm a long way; the travel limit is a routine
    end-of-exercise condition, not only an error path."""
    r = simulate(limits(max_travel_m=0.05), params(10), pull_speed_m_s=0.004, duration_s=60)
    assert r.released
    assert "travel" in r.release_reason


def test_a_stiffer_spring_reaches_the_cap_sooner():
    soft = simulate(limits(), params(10), pull_speed_m_s=0.004, spring_n_per_m=20.0, duration_s=20)
    stiff = simulate(limits(), params(10), pull_speed_m_s=0.004, spring_n_per_m=200.0, duration_s=20)
    assert stiff.arm_travel_m > soft.arm_travel_m


def test_faster_frame_rate_lowers_the_peak_of_a_fast_pull():
    """The flinch case is bounded by how much stretch accumulates between frames, so the loop
    rate, not B, is what moves that number."""
    slow = simulate(limits(), params(10), pull_speed_m_s=0.030, fps=10.0)
    fast = simulate(limits(), params(10), pull_speed_m_s=0.030, fps=120.0)
    assert fast.peak_force_n <= slow.peak_force_n
