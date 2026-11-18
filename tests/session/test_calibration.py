import pytest

from session.calibration import (
    HOW_TO_MEASURE,
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


# ---------------------------------------------------------------- the real rig's file
def test_the_committed_bench_calibration_loads_and_reports_what_is_missing():
    """config/calibration/bench.json is the real rig, partially measured. The point of the
    `None` fields is that this is a normal, loadable state rather than an error."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "config/calibration/bench.json"
    c = BenchCalibration.load(path)
    assert c.tether_n_per_m == pytest.approx(56.2)
    assert not c.complete
    # `yield_sign` is measured: the camera run of 2026-11-15 found -1 for motor 4, and the
    # URDF geometry independently agrees (tests/lib/test_kinematics.py). `ticks_per_metre` is
    # not, and on this arm a single-joint measurement of it cannot be taken at all, because no
    # joint moves the gripper along the tether; `scripts/session/run_session.py --coordinated` derives
    # the equivalent number from the pose instead. See docs/bench-measurements.md.
    assert c.missing == ["ticks_per_metre"]
    with pytest.raises(NotMeasured):
        c.require_axis()
    # The measured parts are usable now.
    assert c.stretch_to_reach_m(1.96) == pytest.approx(0.0349, abs=0.001)
    for finger in ("index", "middle", "ring", "pinky"):
        assert 0.09 < c.require_excursion_m(finger) < 0.12, finger


def test_a_full_range_rep_exceeds_the_default_travel_limit():
    """The consequence of the two measured numbers, pinned so it cannot be forgotten.

    At constant force the arm displaces one for one with the finger, so one full-range rep is
    the excursion minus the ramp to the cap. `SafetyLimits.max_travel_m` defaults to 0.05 m and
    is between a third and a half of that, so it would fire partway through the first rep of
    every block. See config/calibration/bench.md.
    """
    from pathlib import Path

    from lib.units import grams_force_to_newtons

    c = BenchCalibration.load(Path(__file__).resolve().parents[2] / "config/calibration/bench.json")
    cap = grams_force_to_newtons(200)
    ramp_m = c.stretch_to_reach_m(cap) - c.stretch_to_reach_m(cap * 0.25)
    for finger in ("index", "middle", "ring", "pinky"):
        arm_travel = c.require_excursion_m(finger) - ramp_m
        assert arm_travel > 0.05, (
            f"{finger}: a full-range rep needs {arm_travel * 1000:.0f} mm of arm travel, so if "
            "this ever stops exceeding 50 mm the default limit may have been fixed, and "
            "bench.md's table wants updating"
        )


def test_the_measured_tether_is_stiffer_than_the_datasheet_shortcut():
    """The direction of the correction matters: every force-per-millimetre figure derived from
    the old 53.9 N/m estimate understated the force, so the error was on the unsafe side."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "config/calibration/bench.json"
    measured = BenchCalibration.load(path).require_tether_n_per_m()
    assert measured > UNMEASURED_TETHER_ESTIMATE_N_PER_M


def test_the_measured_hysteresis_rules_out_inferring_displacement_from_force():
    """Why HOW_TO_MEASURE tells you to use a ruler on the arm. 6 mm over any move the tether's
    ~110 mm of travel allows is a worse error than a 1 mm ruler reading."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "config/calibration/bench.json"
    c = BenchCalibration.load(path)
    assert c.uncertainty_fraction_at(0.015) > 0.3      # 15 mm move: over 30 % error
    assert c.uncertainty_fraction_at(0.060) > 0.09     # even 60 mm: still ~10 %
    assert "ruler" in HOW_TO_MEASURE["ticks_per_metre"].lower()
    assert "Do NOT use dx = dF / k" in HOW_TO_MEASURE["ticks_per_metre"]
