"""The machine tuning profile: that it loads, that it validates, and that it is what runs.

The last of those is the one worth the most here. A profile that parses correctly but is not
actually consulted is indistinguishable from a working one until someone reads a session log
and finds the arm ran on a number nobody chose, so there are tests below that assert the
committed file, not a constant in a script, is what reaches `argparse`.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from session.tuning import (
    DEFAULT_TUNING_PATH,
    SafetyTuning,
    SessionTuning,
    load_tuning,
)

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def run_session():
    """Import the script by path: `scripts/` is not a package, so it is not importable."""
    spec = importlib.util.spec_from_file_location(
        "run_session", REPO / "scripts" / "session" / "run_session.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ the committed profile
def test_the_committed_profile_exists_and_parses():
    """It is committed on purpose (docs/contributing.md): it describes the shared rig, not one laptop."""
    assert DEFAULT_TUNING_PATH.exists(), (
        f"{DEFAULT_TUNING_PATH} is missing. It is meant to be committed, so that the CLI and "
        "the dashboard run the same numbers."
    )
    assert load_tuning().profile


def test_the_committed_profile_says_where_its_numbers_came_from():
    """Several of these values each pay for a specific observed failure. A profile with no
    provenance is one nobody can safely change later."""
    assert len(load_tuning().note) > 40


def test_the_committed_profile_matches_the_run_that_actually_worked():
    """The bench values from the first session that completed a whole set. Pinned, so that
    changing any of them is a deliberate diff against a named profile and not a drift."""
    t = load_tuning()
    assert (t.damping_n_s_per_m, t.v_max_mm_s, t.max_step_mm, t.max_travel_mm) == (
        10.0, 35.0, 4.0, 80.0
    )
    assert (t.recover_gain, t.recover_retrace_mm_s) == (1.0, 25.0)
    assert t.outlier_persist == 5
    assert t.coordinated is True and t.level is True
    assert (t.coordinated_speed, t.coordinated_accel) == (900, 30)
    assert (t.safety.ceiling_gf, t.safety.anticipation_gf) == (590.0, 5.0)
    assert (t.safety.max_dropouts, t.safety.max_age_s) == (45, 0.5)
    assert t.safety.yield_check is False


def test_the_loosened_safety_values_are_in_the_safety_block():
    """Not cosmetic. They are grouped so a reader cannot skim past them among the damping and
    the frame rate, and so anything consuming the profile can surface them as a group."""
    raw = json.loads(DEFAULT_TUNING_PATH.read_text())
    for key in ("ceiling_gf", "max_dropouts", "yield_check"):
        assert key in raw["safety"], f"{key} must stay inside the safety block"
        assert key not in raw, f"{key} must not also sit at the top level"


# ------------------------------------------------------------------ defaults really are used
def _parse(run_session, argv):
    """Everything `main()` would do up to the point of acting, and nothing after."""
    captured = {}

    async def fake_run(args):
        captured["args"] = args
        return 0

    original = run_session.run
    run_session.run = fake_run
    try:
        run_session.main(argv)
    finally:
        run_session.run = original
    return captured["args"]


def test_the_scripts_defaults_come_from_the_profile(run_session):
    """The regression that matters: someone re-hardcodes a default and the committed profile
    quietly stops being the thing the arm runs on."""
    t = load_tuning()
    args = _parse(run_session, ["--finger", "index"])
    assert args.damping == t.damping_n_s_per_m
    assert args.v_max_mm_s == t.v_max_mm_s
    assert args.max_step_mm == t.max_step_mm
    assert args.max_travel_mm == t.max_travel_mm
    assert args.recover == t.recover_gain
    assert args.recover_retrace_mm_s == t.recover_retrace_mm_s
    assert args.outlier_persist == t.outlier_persist
    assert args.coordinated == t.coordinated
    assert args.level == t.level
    assert args.coordinated_speed == t.coordinated_speed
    assert args.coordinated_accel == t.coordinated_accel
    assert args.ceiling_gf == t.safety.ceiling_gf
    assert args.anticipation_gf == t.safety.anticipation_gf
    assert args.max_dropouts == t.safety.max_dropouts
    assert args.max_age_s == t.safety.max_age_s
    assert args.yield_check == t.safety.yield_check
    assert args.blind_grace_s == t.safety.blind_grace_s
    assert args.slack_blind_grace_s == t.safety.slack_blind_grace_s


def test_a_different_profile_changes_the_defaults(run_session, tmp_path):
    """--tuning has to be read *before* the other defaults are computed, or --help would
    advertise one set of numbers while the session ran on another."""
    other = tmp_path / "other.json"
    other.write_text(json.dumps({
        "profile": "slow-and-careful",
        "damping_n_s_per_m": 40.0,
        "v_max_mm_s": 5.0,
        "max_step_mm": 0.5,
        "coordinated": False,
        "safety": {"ceiling_gf": 210.0, "max_dropouts": 2, "yield_check": True},
    }))
    args = _parse(run_session, ["--tuning", str(other), "--motor", "4"])
    assert args.damping == 40.0
    assert args.v_max_mm_s == 5.0
    assert args.max_step_mm == 0.5
    assert args.coordinated is False
    assert args.ceiling_gf == 210.0
    assert args.max_dropouts == 2
    assert args.yield_check is True


def test_an_explicit_flag_still_beats_the_profile(run_session):
    args = _parse(run_session, ["--max-step-mm", "0.25", "--ceiling-gf", "180"])
    assert args.max_step_mm == 0.25
    assert args.ceiling_gf == 180.0


def test_the_old_no_yield_check_spelling_still_works(run_session):
    """It is in the team's muscle memory and in the docs. BooleanOptionalAction keeps it."""
    assert _parse(run_session, ["--no-yield-check"]).yield_check is False
    assert _parse(run_session, ["--yield-check"]).yield_check is True


def test_coordinated_can_be_turned_off_now_that_it_defaults_on(run_session):
    """It used to be opt-in, so there has to be a way back to the single-joint path."""
    assert _parse(run_session, ["--no-coordinated", "--motor", "4"]).coordinated is False
    assert _parse(run_session, ["--no-level"]).level is False


# ------------------------------------------------------------------ validation
def test_a_missing_file_falls_back_to_the_builtin_defaults(tmp_path):
    """Unlike a bench calibration, every field here has a defensible default, so the code
    runs with or without the file and the two agree."""
    assert load_tuning(tmp_path / "nope.json") == SessionTuning()


def test_a_file_that_does_not_parse_is_an_error_not_a_fallback(tmp_path):
    """Someone meant to say something and it was not understood. Silently running the
    built-ins there would hide the edit entirely."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_tuning(bad)


def test_a_json_array_is_rejected(tmp_path):
    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="must hold a JSON object"):
        load_tuning(arr)


def test_a_misspelled_key_is_rejected_rather_than_ignored(tmp_path):
    """The failure this is here to prevent: an edit that is never read, and a session run on
    a number the person thought they had changed."""
    typo = tmp_path / "typo.json"
    typo.write_text(json.dumps({"max_step_m": 0.004}))
    with pytest.raises(ValueError, match="unknown tuning key"):
        load_tuning(typo)


def test_a_misspelled_safety_key_is_rejected_too(tmp_path):
    """Checked separately because the safety block is where a quiet miss matters most."""
    typo = tmp_path / "typo.json"
    typo.write_text(json.dumps({"safety": {"celing_gf": 590.0}}))
    with pytest.raises(ValueError, match="unknown safety tuning key"):
        load_tuning(typo)


def test_a_non_object_safety_block_is_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"safety": 590.0}))
    with pytest.raises(ValueError, match="'safety' must be an object"):
        load_tuning(bad)


@pytest.mark.parametrize("field", [
    "damping_n_s_per_m", "v_max_mm_s", "max_step_mm", "max_travel_mm",
    "outlier_persist", "coordinated_speed", "coordinated_accel",
])
def test_values_that_are_meaningless_at_zero_are_rejected(field):
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        SessionTuning(**{field: 0})


@pytest.mark.parametrize("field", ["recover_gain", "deadband_n", "recover_retrace_mm_s"])
def test_zero_is_allowed_where_zero_means_disabled(field):
    """Each of these has a real "off" setting, so zero is a choice and not a typo."""
    assert getattr(SessionTuning(**{field: 0.0}), field) == 0.0
    with pytest.raises(ValueError, match=f"{field} cannot be negative"):
        SessionTuning(**{field: -1.0})


def test_a_recovery_floor_of_null_means_use_the_resting_tension():
    """Not the same as zero, which would mean chasing the force all the way to true slack and
    losing the camera's markers. Null defers to the session's own pretension."""
    assert SessionTuning().recover_floor_gf is None
    with pytest.raises(ValueError, match="recover_floor_gf must be positive or null"):
        SessionTuning(recover_floor_gf=0.0)


@pytest.mark.parametrize("field", [
    "ceiling_gf", "max_dropouts", "max_age_s", "yield_check_window_s",
    "blind_grace_s", "slack_blind_grace_s",
])
def test_safety_values_must_be_positive(field):
    with pytest.raises(ValueError, match=f"safety.{field} must be positive"):
        SessionTuning(safety=SafetyTuning(**{field: 0}))


def test_anticipation_may_be_zero_meaning_yield_at_the_literal_cap():
    assert SessionTuning(safety=SafetyTuning(anticipation_gf=0.0)).safety.anticipation_gf == 0.0
    with pytest.raises(ValueError, match="anticipation_gf cannot be negative"):
        SessionTuning(safety=SafetyTuning(anticipation_gf=-1.0))


def test_a_profile_round_trips_through_its_own_dict_form():
    """`as_dict` is what the dashboard and the session log read, so it has to be complete."""
    t = load_tuning()
    data = t.as_dict()
    assert SessionTuning.from_mapping(data) == t
