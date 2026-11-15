"""The TypeScript mirror of the snapshot has to say the same thing as the Python one.

Same arrangement as `tests/hand_tracking/test_frame.py` and `dashboard/app/lib/handFrame.ts`:
Python is the tested side, TypeScript is the mirror, and silent drift between them is the risk,
because nothing else in this repo type-checks the two against each other.
"""
import re
from pathlib import Path

from lib.units import grams_force_to_newtons
from session.instructions import INSTRUCTIONS, SHARED
from session.modes import MODES, Driver, Law, Phase
from session.protocol import FINGERS, Block
from session.snapshot import SCHEMA_VERSION, SessionSnapshot

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_TS = REPO_ROOT / "dashboard" / "app" / "lib" / "sessionSnapshot.ts"


def _ts_union(source: str, name: str) -> set[str]:
    """The string literals of an exported type alias, however it is line-wrapped."""
    match = re.search(rf"export type {name} =(.*?);", source, re.DOTALL)
    assert match, f"{name} not found in {SNAPSHOT_TS}"
    return set(re.findall(r'"([a-z0-9_]+)"', match.group(1)))


def test_the_schema_version_matches():
    source = SNAPSHOT_TS.read_text()
    match = re.search(r"export const SESSION_SCHEMA_VERSION = (\d+);", source)
    assert match, "SESSION_SCHEMA_VERSION not found"
    assert int(match.group(1)) == SCHEMA_VERSION


def test_every_phase_exists_on_both_sides():
    """A phase the UI does not know arrives as an unhandled string and renders as nothing."""
    assert _ts_union(SNAPSHOT_TS.read_text(), "Phase") == {p.value for p in Phase}


def test_every_driver_and_law_exists_on_both_sides():
    source = SNAPSHOT_TS.read_text()
    assert _ts_union(source, "Driver") == {d.value for d in Driver}
    assert _ts_union(source, "Law") == {law.value for law in Law}


def test_every_mode_exists_on_both_sides():
    assert _ts_union(SNAPSHOT_TS.read_text(), "ExerciseMode") == set(MODES)


def test_every_finger_exists_on_both_sides():
    source = SNAPSHOT_TS.read_text()
    match = re.search(r"finger:(.*?);", source, re.DOTALL)
    assert match
    assert set(re.findall(r'"([a-z]+)"', match.group(1))) == set(FINGERS)


def test_the_typescript_side_declares_every_wire_key_python_emits():
    """Walk the real payload rather than a hardcoded list, so a new field cannot be forgotten."""
    source = SNAPSHOT_TS.read_text()
    payload = SessionSnapshot(
        block=Block(finger="index", mode="passive", reps=10,
                    cap_n=grams_force_to_newtons(200), rom_target_m=0.025),
        rep_index=1, rep_total=10, phase=Phase.WORK,
        block_index=0, block_total=1, force_n=1.9, travel_m=0.02,
        progress=0.5, stopped_reason=None,
    ).to_dict()
    for key in payload:
        assert key in source, f"sessionSnapshot.ts does not mention the wire key {key!r}"
    for key in payload["block"]:
        assert key in source, f"sessionSnapshot.ts does not mention the block key {key!r}"


def test_the_running_phases_agree_with_the_instruction_table():
    """RUNNING_PHASES is what the UI uses to decide a cue is on screen, so it must be exactly
    the phases that have something to say to the patient."""
    source = SNAPSHOT_TS.read_text()
    match = re.search(r"RUNNING_PHASES: readonly Phase\[\] = \[(.*?)\]", source, re.DOTALL)
    assert match
    ts_running = {v for v in re.findall(r'"([a-z_]+)"', match.group(1))}
    per_mode_phases = {phase.value for _mode, phase in INSTRUCTIONS}
    assert ts_running == per_mode_phases
    assert ts_running.isdisjoint({p.value for p in SHARED})
