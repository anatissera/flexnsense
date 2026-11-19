"""The session machine, and above all that it never works around the interlock.

The sequencing tests are ordinary. The ones that matter are the last two groups: that a hook
move cannot start while a finger may still be loaded, and that nothing here retries or assumes
its way past a refusal.
"""
import pytest

from arm1.lib.units import grams_force_to_newtons
from arm1.session.arm2 import HookOutcome, HookResult, ManualHookMover
from arm1.session.interlock import ArmOneStatus, Custody, Interlock
from arm1.session.orchestrator import Command, Orchestrator, SessionState
from arm1.session.protocol import Block, Protocol

CAP_N = grams_force_to_newtons(200)
CEILING_N = grams_force_to_newtons(300)

#: Arm 1 released and visibly slack: the only status that lets the hook move.
SLACK = ArmOneStatus(released=True, force_n=0.05, reading_age_s=0.01)
#: Arm 1 still pulling.
LOADED = ArmOneStatus(released=False, force_n=CAP_N, reading_age_s=0.01)
#: The camera cannot see the marker.
BLIND = ArmOneStatus(released=True, force_n=None, reading_age_s=0.01)


def block(finger="index", mode="active", reps=10, **kw) -> Block:
    if mode == "passive":
        kw.setdefault("rom_target_m", 0.025)
    return Block(finger=finger, mode=mode, reps=reps, cap_n=CAP_N, **kw)


def orch(*blocks, mover=None) -> tuple[Orchestrator, ManualHookMover]:
    mover = mover or ManualHookMover()
    protocol = Protocol(blocks=list(blocks) or [block()], ceiling_n=CEILING_N)
    return Orchestrator(protocol, mover), mover


def run_to_exercising(o: Orchestrator, mover: ManualHookMover) -> None:
    """Drive the normal path: release, hand over, hook confirmed, arm 1 running."""
    o.step(SLACK)                 # IDLE -> AWAITING_SLACK
    o.step(SLACK)                 # -> POSITIONING, hook requested
    mover.confirm()
    o.step(SLACK)                 # -> EXERCISING


# ---------------------------------------------------------------- the normal path
def test_a_session_starts_by_releasing_arm_one():
    """Before anything else: the hook has to move, so arm 1 is unloaded first."""
    o, _ = orch()
    step = o.step(SLACK)
    assert step.command is Command.RELEASE_ARM1
    assert o.state is SessionState.AWAITING_SLACK


def test_the_hook_moves_once_arm_one_is_slack():
    o, mover = orch()
    o.step(SLACK)
    step = o.step(SLACK)
    assert step.command is Command.MOVE_HOOK and step.finger == "index"
    assert o.custody is Custody.ARM2
    assert mover.awaiting_confirmation == "index"


def test_arm_one_runs_the_block_once_the_hook_is_confirmed():
    o, mover = orch()
    o.step(SLACK)
    o.step(SLACK)
    mover.confirm()
    step = o.step(SLACK)
    assert step.command is Command.RUN_BLOCK
    assert step.block is not None and step.block.finger == "index"
    assert o.custody is Custody.ARM1


def test_a_finished_block_advances_to_the_next():
    o, mover = orch(block("index"), block("middle"))
    run_to_exercising(o, mover)
    step = o.block_finished(reps_completed=10)
    assert step.command is Command.RELEASE_ARM1
    assert o.state is SessionState.IDLE
    assert o.block is not None and o.block.finger == "middle"
    assert o.custody is Custody.NEITHER


def test_a_whole_two_block_session_runs_end_to_end():
    o, mover = orch(block("index"), block("middle"))
    run_to_exercising(o, mover)
    o.block_finished(10)
    run_to_exercising(o, mover)
    step = o.block_finished(10)
    assert o.state is SessionState.DONE
    assert step.command is Command.RELEASE_ARM1
    assert o.completed_reps == [10, 10]
    assert o.progress == (2, 2)


def test_reps_are_recorded_as_reported_not_as_prescribed():
    """A patient who managed six of ten is a clinical fact; the log is the deliverable."""
    o, mover = orch(block(reps=10))
    run_to_exercising(o, mover)
    o.block_finished(reps_completed=6)
    assert o.completed_reps == [6]


def test_two_blocks_on_one_finger_need_no_rehooking():
    """The hook move is the expensive and risky step, so it only happens on a finger change."""
    o, mover = orch(block("index", "active"), block("index", "passive"))
    run_to_exercising(o, mover)
    o.block_finished(10)
    step = o.step(SLACK)
    assert step.command is Command.RUN_BLOCK
    assert o.state is SessionState.EXERCISING
    assert mover.awaiting_confirmation is None


def test_a_finger_change_does_need_rehooking():
    o, mover = orch(block("index"), block("ring"))
    run_to_exercising(o, mover)
    o.block_finished(10)
    assert o.needs_hook_move
    o.step(SLACK)
    step = o.step(SLACK)
    assert step.command is Command.MOVE_HOOK and step.finger == "ring"


# ---------------------------------------------------------------- the interlock is not bypassed
def test_the_hook_does_not_move_while_arm_one_is_loaded():
    """The hazard this layer exists for: a hook dragged off a finger that is still pulling."""
    o, mover = orch()
    o.step(LOADED)
    step = o.step(LOADED)
    assert step.command is Command.WAIT
    assert o.custody is Custody.NEITHER
    assert mover.awaiting_confirmation is None


def test_the_hook_does_not_move_while_the_force_cannot_be_seen():
    """A dropout is not evidence of zero force, and it is exactly when we know least."""
    o, mover = orch()
    o.step(BLIND)
    step = o.step(BLIND)
    assert step.command is Command.WAIT
    assert "not evidence of zero force" in step.reason
    assert mover.awaiting_confirmation is None


def test_waiting_for_slack_is_not_a_stall_and_recovers():
    """It is a normal state: once the evidence arrives, the session continues by itself."""
    o, mover = orch()
    o.step(LOADED)
    for _ in range(5):
        assert o.step(LOADED).command is Command.WAIT
    step = o.step(SLACK)
    assert step.command is Command.MOVE_HOOK
    assert mover.awaiting_confirmation == "index"


def test_both_arms_are_never_able_to_move_at_once():
    """Walked across the whole normal path, since this is the invariant everything rests on."""
    o, mover = orch(block("index"), block("middle"))
    seen = set()
    for action in ("step", "step", "confirm", "step", "finish") * 2:
        if action == "step":
            o.step(SLACK)
        elif action == "confirm" and mover.awaiting_confirmation:
            mover.confirm()
        elif action == "finish" and o.state is SessionState.EXERCISING:
            o.block_finished(10)
        seen.add(o.custody)
        assert o.custody in (Custody.NEITHER, Custody.ARM1, Custody.ARM2)
    # And all three states really were visited, so the walk proved something.
    assert seen == {Custody.NEITHER, Custody.ARM1, Custody.ARM2}


def test_arm_one_never_exercises_while_arm_two_holds_custody():
    o, mover = orch()
    o.step(SLACK)
    o.step(SLACK)
    assert o.custody is Custody.ARM2
    # Polling repeatedly while the move is unfinished must not start the block.
    for _ in range(3):
        assert o.step(SLACK).command is Command.WAIT
    assert o.state is SessionState.POSITIONING


# ---------------------------------------------------------------- failure is never assumed away
def test_a_failed_hook_move_stops_the_session_rather_than_retrying():
    """A hook in an unknown place is not resolved by trying the same thing again, and the next
    step would be pulling on a finger."""
    o, mover = orch()
    o.step(SLACK)
    o.step(SLACK)
    mover.report_failure("the hook slipped off")
    step = o.step(SLACK)
    assert step.command is Command.STOP
    assert o.state is SessionState.ABORTED
    assert "the hook slipped off" in step.reason


def test_the_hook_position_is_forgotten_while_it_moves():
    """Keeping the previous answer would let a failed move leave a stale 'hooked' claim that
    the next block would trust."""
    o, mover = orch(block("index"), block("middle"))
    run_to_exercising(o, mover)
    assert o.hooked_finger == "index"
    o.block_finished(10)
    o.step(SLACK)
    o.step(SLACK)
    assert o.hooked_finger is None


def test_a_hook_confirmed_on_the_wrong_finger_stops_the_session():
    """Exercising a finger the hook was never confirmed on is the same hazard by another route."""
    o, _ = orch(block("middle"))

    class WrongFinger:
        def request(self, finger): pass
        def poll(self): return HookOutcome(HookResult.CONFIRMED, "index", "confirmed")
        def abort(self): pass

    o.hook_mover = WrongFinger()
    o.step(SLACK)
    o.step(SLACK)
    step = o.step(SLACK)
    assert step.command is Command.STOP
    assert "refusing to exercise" in step.reason


def test_the_hook_position_starts_unknown():
    """Never assumed: the first block always moves the hook, even to the finger it may be on."""
    o, _ = orch()
    assert o.hooked_finger is None
    assert o.needs_hook_move


# ---------------------------------------------------------------- aborting
def test_aborting_is_terminal_and_stops_everything():
    o, mover = orch()
    run_to_exercising(o, mover)
    step = o.abort("force over hard ceiling")
    assert step.command is Command.STOP
    assert o.finished and o.custody is Custody.NEITHER
    assert o.step(SLACK).command is Command.STOP


def test_aborting_works_from_every_state():
    for setup in (
        lambda o, m: None,
        lambda o, m: o.step(SLACK),
        lambda o, m: (o.step(SLACK), o.step(SLACK)),
        lambda o, m: run_to_exercising(o, m),
    ):
        o, mover = orch()
        setup(o, mover)
        assert o.abort("e-stop").command is Command.STOP
        assert o.state is SessionState.ABORTED


def test_aborting_cancels_a_pending_hook_move():
    """So an operator is not left being asked to confirm a hook on a stopped session."""
    o, mover = orch()
    o.step(SLACK)
    o.step(SLACK)
    assert mover.awaiting_confirmation == "index"
    o.abort("clinician stopped the session")
    assert mover.awaiting_confirmation is None


def test_an_abort_keeps_its_original_reason():
    o, _ = orch()
    o.abort("force over hard ceiling")
    o.abort("session ended")
    assert o.step(SLACK).reason == "force over hard ceiling"


def test_a_stopped_interlock_stops_the_session_even_if_nothing_told_the_orchestrator():
    """The gate latching RELEASE reaches the interlock first; the session must notice."""
    lock = Interlock()
    o = Orchestrator(Protocol(blocks=[block()], ceiling_n=CEILING_N), ManualHookMover(), lock)
    lock.stop("gate latched RELEASE")
    step = o.step(SLACK)
    assert step.command is Command.STOP and "gate latched" in step.reason


def test_a_block_finishing_after_an_abort_is_ignored():
    """A late report from arm 1 must not resurrect a stopped session."""
    o, mover = orch(block("index"), block("middle"))
    run_to_exercising(o, mover)
    o.abort("e-stop")
    o.block_finished(10)
    assert o.state is SessionState.ABORTED
    assert o.completed_reps == []


def test_a_stray_block_finished_outside_exercising_is_ignored():
    o, _ = orch()
    step = o.block_finished(10)
    assert step.command is Command.WAIT
    assert o.completed_reps == []


# ---------------------------------------------------------------- the manual mover
def test_the_manual_mover_is_what_runs_today():
    """The policy does not exist, so the implementation that satisfies the interface is a
    person confirming. Nothing above arm2 can tell the difference."""
    mover = ManualHookMover()
    assert mover.poll() is None
    mover.request("pinky")
    assert mover.awaiting_confirmation == "little"   # either vocabulary accepted
    mover.confirm()
    outcome = mover.poll()
    assert outcome is not None and outcome.confirmed and outcome.finger == "little"
    assert mover.poll() is None                      # consumed exactly once


def test_the_manual_mover_refuses_to_confirm_nothing():
    """A confirmation with no request pending would be a confirmation of the previous move."""
    with pytest.raises(RuntimeError):
        ManualHookMover().confirm()
    with pytest.raises(RuntimeError):
        ManualHookMover().report_failure("nope")


def test_the_unimplemented_movers_raise_rather_than_report_success():
    """A hook mover that claims success it did not achieve would have the session pull on a
    finger the hook is not attached to."""
    from arm1.session.arm2 import PolicyHookMover, ReplayHookMover

    with pytest.raises(NotImplementedError, match="no policy has been trained"):
        PolicyHookMover()
    with pytest.raises(NotImplementedError, match="open-loop"):
        ReplayHookMover()


def test_a_mid_block_release_waits_for_the_caller_rather_than_guessing():
    """The documented boundary: during EXERCISING, `released` is ambiguous between the gate
    latching and the block ending normally, and only the control loop knows which. Guessing
    would either end blocks on a safety fault as though they succeeded, or abort sessions that
    merely finished. So it waits, and nothing moves while it does."""
    o, mover = orch()
    run_to_exercising(o, mover)
    step = o.step(ArmOneStatus(released=True, force_n=0.0, reading_age_s=0.01))
    assert step.command is Command.WAIT
    assert o.state is SessionState.EXERCISING
    assert o.custody is Custody.ARM1          # still arm 1's, so arm 2 cannot move
    # And the caller reporting it is what ends the session.
    assert o.abort("gate latched RELEASE").command is Command.STOP
