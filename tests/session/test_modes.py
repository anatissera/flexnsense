import pytest

from session.instructions import INSTRUCTIONS, SHARED, instruction_for
from session.modes import (
    ACTIVE,
    MODES,
    PASSIVE,
    Driver,
    Law,
    Phase,
    Target,
    mode_by_name,
    plan_for,
)

#: Phases a running repetition passes through, so the ones an instruction must exist for.
RUNNING_PHASES = (Phase.ARMED, Phase.CUEING, Phase.RAMP, Phase.WORK, Phase.HOLD, Phase.RESET)


def test_the_two_modes_mirror_each_other():
    """The claim the whole design rests on: one skeleton, with the beats in opposite order."""
    assert ACTIVE.work.driver is Driver.PATIENT and ACTIVE.reset.driver is Driver.DEVICE
    assert PASSIVE.work.driver is Driver.DEVICE and PASSIVE.reset.driver is Driver.PATIENT
    assert ACTIVE.work.law is PASSIVE.reset.law is Law.YIELD
    assert ACTIVE.reset.law is PASSIVE.work.law is Law.PULL


def test_the_driver_flips_within_a_rep_in_both_modes():
    for mode in MODES.values():
        assert mode.work.driver is not mode.reset.driver, mode.name


@pytest.mark.parametrize("mode", MODES.values(), ids=lambda m: m.name)
def test_exactly_one_phase_per_mode_is_device_driven(mode):
    """A rep has one beat where the arm is the source of motion, never two and never none."""
    assert sum(p.device_driven for p in (mode.work, mode.reset)) == 1


def test_only_a_pulling_mode_needs_a_rom_target():
    assert PASSIVE.needs_rom_target
    assert not ACTIVE.needs_rom_target


def test_a_patient_driven_phase_has_no_position_target():
    """The patient's own range is the limit; a target there would fight them."""
    for mode in MODES.values():
        for plan in (mode.work, mode.reset):
            if plan.driver is Driver.PATIENT:
                assert plan.target is Target.NONE, mode.name


def test_a_device_driven_pull_always_has_a_target():
    """Under the cap nothing tells a pulling arm to stop, so force alone cannot bound it."""
    for mode in MODES.values():
        for plan in (mode.work, mode.reset):
            if plan.device_driven:
                assert plan.target is not Target.NONE, mode.name


def test_device_driven_is_pull_and_never_yield():
    """`may_move` is true only on YIELD, so a yielding phase must never be device-driven: that
    pairing is the one that would read as 'advance while over the cap'."""
    for mode in MODES.values():
        for plan in (mode.work, mode.reset):
            if plan.law is Law.YIELD:
                assert not plan.device_driven, mode.name
                assert plan.driver is Driver.PATIENT, mode.name


def test_ramp_and_work_share_a_plan():
    """Below the cap the yield law yields nothing and the pull law advances, which is already
    right for each mode. The two phases differ only in what they mean to the log."""
    for mode in MODES.values():
        assert plan_for(mode, Phase.RAMP) == plan_for(mode, Phase.WORK) == mode.work


def test_arming_is_always_the_device_taking_up_slack():
    for mode in MODES.values():
        plan = plan_for(mode, Phase.ARMED)
        assert plan.driver is Driver.DEVICE
        assert plan.law is Law.PULL


def test_cueing_names_the_driver_about_to_act_but_does_not_move():
    for mode in MODES.values():
        plan = plan_for(mode, Phase.CUEING)
        assert plan.law is Law.STILL
        assert plan.driver is mode.work.driver


def test_hold_does_not_move_in_either_mode():
    for mode in MODES.values():
        assert plan_for(mode, Phase.HOLD).law is Law.STILL


def test_aborting_releases_rather_than_holding():
    """A stuck arm holding tension against a finger with no way to see the force is the worst
    state this machine has, so the abort plan unloads rather than freezing."""
    for mode in MODES.values():
        assert plan_for(mode, Phase.ABORTED).law is Law.RELEASE


def test_every_phase_has_a_plan():
    for mode in MODES.values():
        for phase in Phase:
            assert plan_for(mode, phase) is not None, (mode.name, phase)


def test_unknown_mode_names_the_known_ones():
    with pytest.raises(ValueError, match="active"):
        mode_by_name("tension")


# ---------------------------------------------------------------- the two tables agree
def test_every_mode_has_an_instruction_for_every_running_phase():
    """The check that keeps the mode registry and the instruction table from drifting apart."""
    for name in MODES:
        for phase in RUNNING_PHASES:
            assert (name, phase) in INSTRUCTIONS, f"missing instruction: {name} in {phase.value}"


def test_no_instruction_row_refers_to_an_unknown_mode():
    for name, _phase in INSTRUCTIONS:
        assert name in MODES, f"instruction table mentions unknown mode {name!r}"


def test_shared_and_per_mode_phases_do_not_overlap():
    """A phase worded per mode must not also be worded globally, or one silently wins."""
    per_mode = {phase for _name, phase in INSTRUCTIONS}
    assert per_mode.isdisjoint(SHARED)


def test_every_device_driven_phase_tells_the_patient_to_relax():
    """A patient tensing against an arm that is pulling is the one thing the force cap cannot
    distinguish from a stiff joint, so the wording has to ask for it."""
    for name, mode in MODES.items():
        for phase in RUNNING_PHASES:
            if plan_for(mode, phase).device_driven:
                text = instruction_for(name, phase, "index").lower()
                assert "relax" in text, f"{name} in {phase.value} does not ask for relaxation"
