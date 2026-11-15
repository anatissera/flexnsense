import pytest

from session.calibration import (
    UNMEASURED_TETHER_ESTIMATE_N_PER_M,
    BenchCalibration,
    NotMeasured,
)


def full(**kw) -> BenchCalibration:
    base = dict(
        tether_n_per_m=54.0,
        ticks_per_metre=1200.0,
        yield_sign=-1,
        excursion_m={"index": 0.060, "middle": 0.065},
        displacement_uncertainty_m=0.001,
        measured_in_pose="session pose A",
    )
    base.update(kw)
    return BenchCalibration(**base)


# ---------------------------------------------------------------- absence is explicit
def test_a_fresh_calibration_knows_nothing_and_says_so():
    c = BenchCalibration()
    assert not c.complete
    assert c.missing == ["tether_n_per_m", "ticks_per_metre", "yield_sign", "excursion_m"]


def test_the_stiffness_is_never_silently_defaulted():
    """The whole point: no placeholder can become the number the arm runs on."""
    with pytest.raises(NotMeasured):
        BenchCalibration().require_tether_n_per_m()


def test_the_error_carries_the_procedure_not_just_the_name():
    """A person who hits this should not have to go find the file to learn what to do."""
    with pytest.raises(NotMeasured) as exc:
        BenchCalibration().require_tether_n_per_m()
    message = str(exc.value)
    assert "PESOLA is the force reference" in message
    assert "lib.stiffness.fit_stiffness" in message


def test_the_axis_reports_both_missing_fields_at_once():
    """So one bench trip fixes both, instead of discovering the second after the first."""
    with pytest.raises(NotMeasured) as exc:
        BenchCalibration().require_axis()
    assert exc.value.fields == ("ticks_per_metre", "yield_sign")


def test_a_half_measured_axis_still_fails():
    with pytest.raises(NotMeasured) as exc:
        BenchCalibration(ticks_per_metre=1200.0).require_axis()
    assert exc.value.fields == ("yield_sign",)


def test_a_complete_calibration_hands_the_numbers_over():
    assert full().complete
    assert full().require_axis() == (1200.0, -1)
    assert full().require_excursion_m("index") == 0.060


def test_an_unmeasured_finger_fails_even_when_others_are_measured():
    with pytest.raises(NotMeasured):
        full().require_excursion_m("ring")


def test_the_simulation_estimate_is_not_the_calibration():
    """It exists for control.simulate and must never leak into the control path, so nothing
    here defaults to it."""
    assert BenchCalibration().tether_n_per_m is None
    assert UNMEASURED_TETHER_ESTIMATE_N_PER_M == pytest.approx(53.9)


# ---------------------------------------------------------------- the physics it encodes
def test_reaching_the_cap_costs_real_range():
    """The 36 mm finding: on the estimated tether, a 1.96 N cap is 36 mm of stretch, which is
    range the patient spends on a ramp rather than on the exercise."""
    c = BenchCalibration(tether_n_per_m=UNMEASURED_TETHER_ESTIMATE_N_PER_M)
    assert c.stretch_to_reach_m(1.96) == pytest.approx(0.0364, abs=0.001)


def test_a_stiffer_tether_costs_less_range():
    soft = BenchCalibration(tether_n_per_m=54.0).stretch_to_reach_m(1.96)
    stiff = BenchCalibration(tether_n_per_m=400.0).stretch_to_reach_m(1.96)
    assert stiff < soft


# ---------------------------------------------------------------- the inherited error bar
def test_a_short_tick_measurement_is_mostly_hysteresis():
    """Why the procedure says to command a large tick delta: inferring dx from dF over 2 mm
    against a 1 mm hysteresis floor is a 50 % error on ticks_per_metre."""
    c = full(displacement_uncertainty_m=0.001)
    assert c.uncertainty_fraction_at(0.002) == pytest.approx(0.5)
    assert c.uncertainty_fraction_at(0.030) == pytest.approx(0.0333, abs=0.001)


def test_uncertainty_is_unknown_rather_than_zero_when_unmeasured():
    """Reporting 0 would claim a precision nobody measured."""
    assert full(displacement_uncertainty_m=None).uncertainty_fraction_at(0.01) is None


def test_a_zero_length_move_has_no_meaningful_error():
    with pytest.raises(ValueError):
        full().uncertainty_fraction_at(0.0)


# ---------------------------------------------------------------- validation and persistence
@pytest.mark.parametrize("kw", [
    {"tether_n_per_m": 0.0},
    {"tether_n_per_m": -54.0},
    {"ticks_per_metre": -1.0},
    {"yield_sign": 0},
    {"yield_sign": 2},
    {"excursion_m": {"index": 0.0}},
    {"displacement_uncertainty_m": -0.001},
])
def test_impossible_values_are_rejected(kw):
    with pytest.raises(ValueError):
        full(**kw)


def test_a_missing_file_is_an_empty_calibration_not_a_crash(tmp_path):
    """Nothing can be measured before the rig exists, so 'no file yet' is a normal state."""
    c = BenchCalibration.load(tmp_path / "nope.json")
    assert c.missing


def test_it_round_trips_through_json(tmp_path):
    path = tmp_path / "bench.json"
    full().save(path)
    assert BenchCalibration.load(path) == full()


def test_a_partially_measured_calibration_round_trips(tmp_path):
    """The state the rig will actually be in halfway through the bench session."""
    partial = BenchCalibration(tether_n_per_m=54.0, measured_in_pose="pose A")
    path = tmp_path / "partial.json"
    partial.save(path)
    assert BenchCalibration.load(path) == partial


def test_a_typo_in_a_key_is_not_silently_ignored():
    with pytest.raises(ValueError, match="unknown calibration keys"):
        BenchCalibration.from_dict({"tether_N_per_m": 54.0})
