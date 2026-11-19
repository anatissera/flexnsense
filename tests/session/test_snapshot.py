import pytest

from arm1.lib.units import grams_force_to_newtons
from arm1.session.modes import Driver, Law, Phase
from arm1.session.protocol import Block
from arm1.session.snapshot import SCHEMA_VERSION, SessionSnapshot

CAP_N = grams_force_to_newtons(200)

ACTIVE_BLOCK = Block(finger="index", mode="active", reps=10, cap_n=CAP_N)
PASSIVE_BLOCK = Block(finger="middle", mode="passive", reps=10, cap_n=CAP_N, rom_target_m=0.025)


def snap(block=ACTIVE_BLOCK, phase=Phase.WORK, **kw) -> SessionSnapshot:
    base = dict(
        block=block, rep_index=3, rep_total=10, phase=phase,
        block_index=0, block_total=2, force_n=1.94, travel_m=0.0234,
    )
    base.update(kw)
    return SessionSnapshot(**base)


# ---------------------------------------------------------------- driver is the whole contract
def test_the_driver_flips_within_a_rep():
    """What the UI keys off: the same block reports a different driver in work and reset."""
    assert snap(ACTIVE_BLOCK, Phase.WORK).driver is Driver.PATIENT
    assert snap(ACTIVE_BLOCK, Phase.RESET).driver is Driver.DEVICE


def test_the_driver_flips_the_other_way_in_the_other_mode():
    assert snap(PASSIVE_BLOCK, Phase.WORK).driver is Driver.DEVICE
    assert snap(PASSIVE_BLOCK, Phase.RESET).driver is Driver.PATIENT


def test_device_driven_is_only_true_while_the_arm_pulls():
    assert snap(PASSIVE_BLOCK, Phase.WORK).device_driven
    assert snap(ACTIVE_BLOCK, Phase.RESET).device_driven
    assert not snap(ACTIVE_BLOCK, Phase.WORK).device_driven
    assert not snap(PASSIVE_BLOCK, Phase.RESET).device_driven


def test_holding_is_never_device_driven_even_in_passive_mode():
    """The arm is not moving the finger during a hold, so the UI must not show a demand."""
    assert not snap(PASSIVE_BLOCK, Phase.HOLD).device_driven
    assert snap(PASSIVE_BLOCK, Phase.HOLD).law is Law.STILL


# ---------------------------------------------------------------- the wire format
def test_both_modes_serialise_to_the_same_shape():
    """The abstraction: adding a mode must not add a field, or the dashboard has to branch."""
    a = snap(ACTIVE_BLOCK, Phase.WORK).to_dict()
    p = snap(PASSIVE_BLOCK, Phase.WORK).to_dict()
    assert a.keys() == p.keys()
    assert a["block"].keys() == p["block"].keys()


def test_the_payload_carries_the_phase_driver_and_instruction_together():
    """These three have to arrive in one message: aligning them is the point of the snapshot."""
    d = snap(PASSIVE_BLOCK, Phase.WORK).to_dict()
    assert d["phase"] == "work"
    assert d["driver"] == "device"
    assert d["instruction"] == {"key": "passive.work", "text": "Stay relaxed, let it stretch"}


def test_the_instruction_names_the_finger_from_the_block():
    d = snap(PASSIVE_BLOCK, Phase.RESET).to_dict()
    assert "middle" in d["instruction"]["text"]


def test_force_is_reported_in_both_units():
    """Newtons for the control math, grams because that is what the clinician's cap is set in
    and what the PESOLA is graduated in."""
    d = snap(force_n=1.96).to_dict()
    assert d["forceN"] == pytest.approx(1.96, abs=0.001)
    assert d["forceGf"] == pytest.approx(200, abs=0.5)


def test_a_dropout_is_sent_as_null_and_never_as_zero():
    """Zero newtons means 'no tension measured', which is a completely different claim from
    'the marker was not seen'. Collapsing them is how a watchdog gets talked out of firing."""
    d = snap(force_n=None).to_dict()
    assert d["forceN"] is None and d["forceGf"] is None


def test_a_missing_progress_is_null_rather_than_zero():
    assert snap(progress=None).to_dict()["progress"] is None
    assert snap(progress=0.0).to_dict()["progress"] == 0.0


def test_travel_is_sent_in_millimetres_for_display():
    assert snap(travel_m=0.0234).to_dict()["travelMm"] == pytest.approx(23.4, abs=0.05)


def test_the_payload_is_versioned():
    assert snap().to_dict()["schema"] == SCHEMA_VERSION


def test_the_rom_target_appears_only_where_it_applies():
    assert snap(PASSIVE_BLOCK).to_dict()["block"]["romTargetMm"] == pytest.approx(25.0)
    assert snap(ACTIVE_BLOCK).to_dict()["block"]["romTargetMm"] is None


# ---------------------------------------------------------------- the edges
def test_an_idle_session_serialises_without_a_block():
    d = SessionSnapshot(
        block=None, rep_index=0, rep_total=0, phase=Phase.IDLE,
        block_index=0, block_total=0, force_n=None, travel_m=0.0,
    ).to_dict()
    assert d["block"] is None
    assert d["instruction"]["text"] == "Ready when you are"


def test_an_idle_session_claims_no_patient_demand():
    """With nothing prescribed, the UI must not render an instruction as something to do."""
    s = SessionSnapshot(
        block=None, rep_index=0, rep_total=0, phase=Phase.IDLE,
        block_index=0, block_total=0, force_n=None, travel_m=0.0,
    )
    assert s.driver is Driver.DEVICE and not s.device_driven


def test_an_abort_carries_its_reason_and_the_same_words_in_both_modes():
    for block in (ACTIVE_BLOCK, PASSIVE_BLOCK):
        d = snap(block, Phase.ABORTED, stopped_reason="force over hard ceiling").to_dict()
        assert d["stoppedReason"] == "force over hard ceiling"
        assert d["instruction"]["text"] == "Stop and relax your hand"
        assert d["instruction"]["key"] == "aborted"
        assert d["law"] == "release"
