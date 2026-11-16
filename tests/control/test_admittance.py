import pytest

from control.admittance import AdmittanceParams, step_from_velocity, yield_velocity

THRESHOLD = 1.96


def params(**kw) -> AdmittanceParams:
    base = dict(damping_n_s_per_m=100.0, v_max_m_s=0.01, deadband_n=0.05, max_step_m=0.002)
    base.update(kw)
    return AdmittanceParams(**base)


# --------------------------------------------------------------------------- yield_velocity

def test_under_threshold_does_not_move():
    assert yield_velocity(1.0, THRESHOLD, params()) == 0.0


def test_exactly_at_threshold_does_not_move():
    assert yield_velocity(THRESHOLD, THRESHOLD, params()) == 0.0


def test_inside_deadband_does_not_move():
    p = params(deadband_n=0.1)
    assert yield_velocity(THRESHOLD + 0.05, THRESHOLD, p) == 0.0


def test_just_past_deadband_moves():
    p = params(deadband_n=0.1)
    assert yield_velocity(THRESHOLD + 0.11, THRESHOLD, p) > 0.0


def test_velocity_is_proportional_to_overshoot():
    p = params(damping_n_s_per_m=100.0, deadband_n=0.0, v_max_m_s=10.0)
    assert yield_velocity(THRESHOLD + 1.0, THRESHOLD, p) == pytest.approx(0.01)
    assert yield_velocity(THRESHOLD + 2.0, THRESHOLD, p) == pytest.approx(0.02)


def test_velocity_is_capped_at_v_max():
    p = params(damping_n_s_per_m=1.0, v_max_m_s=0.01, deadband_n=0.0)
    assert yield_velocity(THRESHOLD + 100.0, THRESHOLD, p) == 0.01


def test_never_negative_even_far_under_threshold():
    """This law only ever reduces tension; it must not pull the finger back."""
    assert yield_velocity(0.0, THRESHOLD, params()) == 0.0
    assert yield_velocity(-5.0, THRESHOLD, params()) == 0.0


def test_higher_damping_yields_more_slowly():
    slow = yield_velocity(THRESHOLD + 1.0, THRESHOLD, params(damping_n_s_per_m=1000.0, deadband_n=0.0))
    fast = yield_velocity(THRESHOLD + 1.0, THRESHOLD, params(damping_n_s_per_m=10.0, deadband_n=0.0))
    assert slow < fast


# --------------------------------------------------------------------------- step_from_velocity

def test_step_is_velocity_times_dt():
    p = params(max_step_m=1.0)
    assert step_from_velocity(0.01, 0.033, p) == pytest.approx(0.01 * 0.033)


def test_step_is_capped_regardless_of_velocity():
    """The arithmetic backstop: no single cycle may command a large jump."""
    p = params(max_step_m=0.002)
    assert step_from_velocity(10.0, 0.033, p) == 0.002


def test_longer_dt_gives_a_longer_step():
    p = params(max_step_m=1.0)
    short = step_from_velocity(0.01, 0.020, p)
    long_ = step_from_velocity(0.01, 0.050, p)
    assert long_ > short


def test_zero_velocity_gives_no_step():
    assert step_from_velocity(0.0, 0.033, params()) == 0.0


def test_non_positive_dt_gives_no_step():
    """A stalled or backwards clock is not a reason to move."""
    assert step_from_velocity(0.01, 0.0, params()) == 0.0
    assert step_from_velocity(0.01, -0.5, params()) == 0.0


def test_absurdly_long_dt_gives_no_step():
    """After a long stall, dt * v would be a large jump; refuse rather than lurch."""
    assert step_from_velocity(0.01, 5.0, params()) == 0.0


# --------------------------------------------------------------------------- params validation

def test_non_positive_damping_rejected():
    with pytest.raises(ValueError, match="damping"):
        AdmittanceParams(damping_n_s_per_m=0.0, v_max_m_s=0.01)


def test_non_positive_v_max_rejected():
    with pytest.raises(ValueError, match="v_max_m_s"):
        AdmittanceParams(damping_n_s_per_m=100.0, v_max_m_s=0.0)


def test_negative_deadband_rejected():
    with pytest.raises(ValueError, match="deadband"):
        AdmittanceParams(damping_n_s_per_m=100.0, v_max_m_s=0.01, deadband_n=-0.1)


def test_non_positive_max_step_rejected():
    with pytest.raises(ValueError, match="max_step_m"):
        AdmittanceParams(damping_n_s_per_m=100.0, v_max_m_s=0.01, max_step_m=0.0)


# ---------------------------------------------------------------- the recover law
def params_for_recover(**kw):
    from control.admittance import AdmittanceParams

    base = dict(damping_n_s_per_m=10.0, v_max_m_s=0.035, deadband_n=0.05, max_step_m=0.001)
    base.update(kw)
    return AdmittanceParams(**base)


def test_recover_mirrors_yield():
    """The same law, driven by the shortfall instead of the overshoot: equal distances from
    the threshold on either side must produce equal speeds at gain 1."""
    from control.admittance import recover_velocity, yield_velocity

    p = params_for_recover()
    assert recover_velocity(1.0, 2.0, p) == yield_velocity(3.0, 2.0, p)


def test_recover_is_zero_at_and_above_the_threshold():
    from control.admittance import recover_velocity

    p = params_for_recover()
    assert recover_velocity(2.0, 2.0, p) == 0.0
    assert recover_velocity(2.5, 2.0, p) == 0.0


def test_recover_respects_the_deadband():
    """Resting-force noise must not twitch the arm back and forth around the setpoint."""
    from control.admittance import recover_velocity

    p = params_for_recover(deadband_n=0.1)
    assert recover_velocity(1.95, 2.0, p) == 0.0


def test_recover_is_clamped_to_v_max():
    from control.admittance import recover_velocity

    p = params_for_recover()
    assert recover_velocity(0.0, 50.0, p) == p.v_max_m_s


def test_recover_gain_scales_the_speed():
    from control.admittance import recover_velocity

    # A slack small enough that neither gain hits the v_max clamp, which would otherwise
    # flatten the two speeds together and hide a broken gain entirely.
    p = params_for_recover()
    assert recover_velocity(1.8, 2.0, p, gain=0.5) == pytest.approx(
        recover_velocity(1.8, 2.0, p) / 2
    )


def test_recover_is_never_negative():
    from control.admittance import recover_velocity

    p = params_for_recover()
    for force in (0.0, 1.0, 2.0, 3.0, 10.0):
        assert recover_velocity(force, 2.0, p) >= 0.0


# ---------------------------------------------------------------- the recovery floor
def test_recover_stops_at_the_floor_even_with_slack_remaining():
    """The bug this closes: without a floor, recovery kept commanding speed toward force
    zero, which on the bench meant the tether went slack enough to lose the camera's PESOLA
    markers before the tick-based boundary caught up. A floor stops the velocity law itself,
    independent of that boundary."""
    from control.admittance import recover_velocity

    p = params_for_recover()
    assert recover_velocity(0.3, 2.0, p, floor_n=0.5) == 0.0


def test_recover_runs_normally_above_the_floor():
    from control.admittance import recover_velocity

    p = params_for_recover()
    assert recover_velocity(1.8, 2.0, p, floor_n=0.5) > 0.0


def test_the_floor_is_exact_at_the_boundary():
    """At force_n == floor_n there is nothing left to recover toward; this is the same
    convention as yield_velocity's deadband, zero rather than an infinitesimal residual."""
    from control.admittance import recover_velocity

    p = params_for_recover()
    assert recover_velocity(0.5, 2.0, p, floor_n=0.5) == 0.0


def test_a_zero_floor_is_the_original_unbounded_behaviour():
    from control.admittance import recover_velocity

    p = params_for_recover()
    with_floor = recover_velocity(1.8, 2.0, p, floor_n=0.0)
    without = recover_velocity(1.8, 2.0, p)
    assert with_floor == without
