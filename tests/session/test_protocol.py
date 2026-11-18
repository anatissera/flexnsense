from pathlib import Path

import pytest

from lib.units import grams_force_to_newtons
from session.calibration import BenchCalibration
from session.protocol import Block, Protocol

CAP_N = grams_force_to_newtons(200)
CEILING_N = grams_force_to_newtons(300)


def active(**kw) -> Block:
    return Block(**{"finger": "index", "mode": "active", "reps": 10, "cap_n": CAP_N, **kw})


def passive(**kw) -> Block:
    return Block(**{"finger": "index", "mode": "passive", "reps": 10, "cap_n": CAP_N,
                    "rom_target_m": 0.025, **kw})


# ---------------------------------------------------------------- the prescription is data
def test_a_block_is_just_numbers():
    b = active(reps=10, hold_s=3.0)
    assert b.reps == 10 and b.hold_s == 3.0
    assert b.mode_spec.name == "active"


def test_the_cap_is_set_in_grams_and_stored_in_newtons():
    """docs/contributing.md units rule: the clinician reads grams off the PESOLA, control math is SI."""
    b = Block.from_grams("index", "active", 10, cap_gf=200)
    assert b.cap_n == pytest.approx(1.96, abs=0.01)
    assert b.cap_gf == pytest.approx(200, abs=0.1)


# ---------------------------------------------------------------- the rom target rule
def test_a_pulling_mode_requires_a_rom_target():
    """Under the cap nothing tells a pulling arm to stop, so force alone cannot bound it."""
    with pytest.raises(ValueError, match="rom_target_m"):
        Block(finger="index", mode="passive", reps=10, cap_n=CAP_N)


def test_a_non_pulling_mode_rejects_a_rom_target():
    """Silently ignoring it would let a clinician think a limit is in force when it is not."""
    with pytest.raises(ValueError, match="no meaning"):
        Block(finger="index", mode="active", reps=10, cap_n=CAP_N, rom_target_m=0.02)


def test_a_passive_block_is_fine_with_one():
    assert passive().rom_target_m == 0.025


# ---------------------------------------------------------------- plain validation
@pytest.mark.parametrize("kw", [
    {"finger": "pinky"},        # not one of the five
    {"mode": "tension"},        # the name we deliberately did not use
    {"reps": 0},
    {"cap_n": 0.0},
    {"hold_s": -1.0},
])
def test_invalid_blocks_are_rejected(kw):
    with pytest.raises(ValueError):
        active(**kw)


def test_a_cap_at_or_above_the_ceiling_is_rejected():
    """The cap has to leave the controller headroom to give way in before the gate releases."""
    with pytest.raises(ValueError, match="ceiling"):
        Protocol(blocks=[active(cap_n=CEILING_N)], ceiling_n=CEILING_N)


def test_a_protocol_needs_at_least_one_block():
    with pytest.raises(ValueError, match="at least one block"):
        Protocol(blocks=[], ceiling_n=CEILING_N)


# ---------------------------------------------------------------- what the orchestrator reads
def test_total_reps_sums_the_blocks():
    p = Protocol(blocks=[active(reps=10), passive(reps=5)], ceiling_n=CEILING_N)
    assert p.total_reps == 15
    assert len(p) == 2


def test_consecutive_blocks_on_one_finger_need_no_rehooking():
    """fingers_in_order is what becomes arm 2 tasks, so a repeat must not produce one."""
    p = Protocol(
        blocks=[active(finger="index"), passive(finger="index"), active(finger="middle")],
        ceiling_n=CEILING_N,
    )
    assert p.fingers_in_order == ["index", "middle"]


def test_each_finger_change_produces_one_hook_move():
    p = Protocol(
        blocks=[active(finger="index"), active(finger="middle"), active(finger="ring")],
        ceiling_n=CEILING_N,
    )
    assert p.fingers_in_order == ["index", "middle", "ring"]


# ---------------------------------------------------------------- bench-dependent warnings
def test_an_unmeasured_rig_produces_no_warnings():
    """Writing a prescription before the rig is characterised is a normal state, not an error."""
    p = Protocol(blocks=[passive()], ceiling_n=CEILING_N)
    assert p.check_against(BenchCalibration()) == []


def test_a_rom_target_beyond_the_finger_is_flagged():
    calib = BenchCalibration(excursion_m={"index": 0.020})
    problems = passive(rom_target_m=0.030).check_against(calib)
    assert len(problems) == 1 and "exceeds the measured excursion" in problems[0]


def test_a_cap_the_patient_could_never_reach_is_flagged():
    """The 36 mm finding: on a soft tether, reaching the cap can eat the whole range, so the
    patient would stretch the spring to the end of their travel and never trigger the loop."""
    soft = BenchCalibration(tether_n_per_m=54.0, excursion_m={"index": 0.030})
    problems = active().check_against(soft)
    assert any("never reach the cap" in p for p in problems)


def test_a_stiff_enough_tether_raises_nothing():
    stiff = BenchCalibration(tether_n_per_m=400.0, excursion_m={"index": 0.060})
    assert active().check_against(stiff) == []


def test_warnings_say_which_block_they_came_from():
    calib = BenchCalibration(excursion_m={"index": 0.010})
    p = Protocol(blocks=[active(), passive(rom_target_m=0.030)], ceiling_n=CEILING_N)
    problems = p.check_against(calib)
    assert problems and problems[0].startswith("block 1 (index, passive):")


# ---------------------------------------------------------------- persistence
def test_a_protocol_round_trips_through_json(tmp_path):
    p = Protocol(blocks=[active(), passive()], ceiling_n=CEILING_N, name="day 3")
    path = tmp_path / "proto.json"
    p.save(path)
    back = Protocol.load(path)
    assert back == p


def test_a_cap_can_be_written_in_grams_in_the_file():
    """So a clinician-authored file reads in the unit they set it in."""
    p = Protocol.from_dict({
        "ceiling_gf": 300,
        "blocks": [{"finger": "index", "mode": "active", "reps": 10, "cap_gf": 200}],
    })
    assert p.blocks[0].cap_gf == pytest.approx(200, abs=0.1)
    assert p.ceiling_n == pytest.approx(grams_force_to_newtons(300))


def test_giving_both_units_is_rejected():
    """Two sources for one number is how they end up disagreeing."""
    with pytest.raises(ValueError, match="exactly one"):
        Block.from_dict({"finger": "index", "mode": "active", "reps": 10,
                         "cap_n": 1.96, "cap_gf": 200})


def test_a_typo_in_a_key_is_not_silently_ignored():
    with pytest.raises(ValueError, match="unknown block keys"):
        Block.from_dict({"finger": "index", "mode": "active", "reps": 10, "cap_gf": 200,
                         "hold_sec": 3})


# ------------------------------------------------------- the committed example protocols
PROTOCOL_DIR = Path(__file__).resolve().parents[2] / "config" / "protocols"


def _example_protocols():
    return sorted(PROTOCOL_DIR.glob("*.json"))


def test_there_is_at_least_one_example_protocol():
    """The simulation and the dashboard both need something real to load."""
    assert _example_protocols()


@pytest.mark.parametrize("path", _example_protocols(), ids=lambda p: p.name)
def test_every_example_protocol_loads(path):
    """A committed protocol that does not parse is one nobody can run the simulation on."""
    assert len(Protocol.load(path)) >= 1


@pytest.mark.parametrize("path", _example_protocols(), ids=lambda p: p.name)
def test_every_example_protocol_is_clean_against_the_bench(path):
    """Catches the quiet one: a cap edited upward until reaching it would take the finger's
    whole excursion, so the patient could never reach it and the exercise is all run-up."""
    bench = BenchCalibration.load(
        Path(__file__).resolve().parents[2] / "config" / "calibration" / "bench.json"
    )
    assert Protocol.load(path).check_against(bench) == []


@pytest.mark.parametrize("path", _example_protocols(), ids=lambda p: p.name)
def test_every_example_protocol_says_it_is_an_example(path):
    """These are made up. A file that could be mistaken for a real prescription is a
    patient-safety problem, not a documentation one."""
    assert "EXAMPLE" in (Protocol.load(path).name or "")


@pytest.mark.parametrize("path", _example_protocols(), ids=lambda p: p.name)
def test_every_example_protocol_names_its_mode_in_its_filename(path):
    """The mode is chosen by what is being treated, so the two belong together in the name:
    the same finger at the same cap is a different exercise under a different mode."""
    modes = {b.mode for b in Protocol.load(path)}
    assert any(m in path.stem for m in modes), (
        f"{path.name} runs mode(s) {sorted(modes)}, none of which appear in its filename"
    )
