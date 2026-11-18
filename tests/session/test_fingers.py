"""The boundary between the repo's two finger vocabularies.

Both are real and both are load-bearing, so these tests pin the reconciliation rather than a
preference: `hand_tracking.landmarks.FINGERS` keys on "pinky" (and `test_landmarks.py` asserts
on it, and the gauge CSV columns come from it), while `curlsim/types.ts` declares "little" as
the `Finger` type that `FLEXOR_MUSCLES` is keyed by. Neither gets renamed; the session layer
accepts both and speaks the clinical one.
"""
import pytest

from arm1.session.calibration import BenchCalibration
from arm1.session.fingers import FINGERS, normalise, normalise_keys
from arm1.session.protocol import Block


def test_the_canonical_names_match_the_dashboards_clinical_type():
    """curlsim's `Finger` is the clinical vocabulary and FLEXOR_MUSCLES is keyed by it, so a
    block has to round-trip to the UI untranslated."""
    assert FINGERS == ("thumb", "index", "middle", "ring", "little")


def test_the_tracker_vocabulary_is_accepted():
    """What a measurement physically arrives as, since landmarks.FINGERS says pinky."""
    assert normalise("pinky") == "little"


def test_canonical_names_pass_through_unchanged():
    for name in FINGERS:
        assert normalise(name) == name


@pytest.mark.parametrize("spelling", ["PINKY", " pinky ", "Little", "little finger"])
def test_spellings_that_arrive_from_csv_headers_and_hand_written_json(spelling):
    assert normalise(spelling) == "little"


def test_an_unknown_finger_lists_both_vocabularies():
    """The error is what a person at the bench reads, so it has to name what is accepted."""
    with pytest.raises(ValueError, match="pinky"):
        normalise("fourth")


# ---------------------------------------------------------------- where it is applied
def test_a_calibration_built_by_a_measurement_script_is_canonicalised():
    """The hole worth closing: a script constructs this directly rather than loading a file,
    so normalising only in from_dict would leave the key as "pinky" and the lookup would miss."""
    c = BenchCalibration(excursion_m={"pinky": 0.04})
    assert c.excursion_m == {"little": 0.04}


def test_either_vocabulary_looks_up_the_same_measurement():
    c = BenchCalibration(excursion_m={"pinky": 0.04})
    assert c.require_excursion_m("pinky") == 0.04
    assert c.require_excursion_m("little") == 0.04


def test_a_file_carrying_both_spellings_is_rejected():
    """Silently keeping one would discard a real measurement, and there is no way to tell which
    of the two is current."""
    with pytest.raises(ValueError, match="two spellings"):
        BenchCalibration(excursion_m={"pinky": 0.04, "little": 0.05})


def test_a_protocol_file_written_in_the_tracker_vocabulary_loads():
    b = Block.from_dict({"finger": "pinky", "mode": "active", "reps": 10, "cap_gf": 200})
    assert b.finger == "little"


def test_constructing_a_block_with_an_alias_says_what_to_use():
    """Direct construction stays strict, because a `Block` in code should read canonically, but
    the error names the fix rather than only rejecting."""
    with pytest.raises(ValueError, match="Use 'little'"):
        Block(finger="pinky", mode="active", reps=10, cap_n=1.96)


def test_a_block_round_trips_without_reintroducing_an_alias():
    b = Block.from_dict({"finger": "pinky", "mode": "active", "reps": 10, "cap_gf": 200})
    assert Block.from_dict(b.to_dict()) == b


# ---------------------------------------------------------------- the two sources, pinned
def test_the_tracking_side_really_does_say_pinky():
    """If `landmarks.FINGERS` is ever renamed, the alias here becomes dead weight and the
    measurement path changes vocabulary silently. This is the tripwire for that."""
    from arm1.hand_tracking.landmarks import FINGERS as TRACKER_FINGERS

    assert "pinky" in TRACKER_FINGERS
    assert "little" not in TRACKER_FINGERS
    # The two vocabularies differ in exactly one name; anything else means a second alias is
    # needed and `normalise` would start silently rejecting a finger.
    assert set(TRACKER_FINGERS) - set(FINGERS) == {"pinky"}
    assert set(FINGERS) - set(TRACKER_FINGERS) == {"little"}


def test_the_clinical_side_really_does_say_little():
    """curlsim's `Finger` type is the other half. Read from the file, like
    tests/hand_tracking/test_frame.py reads handFrame.ts, since there is no JS test runner."""
    import re
    from pathlib import Path

    types_ts = Path(__file__).resolve().parents[2] / "dashboard/app/lib/curlsim/types.ts"
    source = types_ts.read_text()
    match = re.search(r"export type Finger =(.*?);", source, re.DOTALL)
    assert match, f"Finger type not found in {types_ts}"
    assert set(re.findall(r'"([a-z]+)"', match.group(1))) == set(FINGERS)
