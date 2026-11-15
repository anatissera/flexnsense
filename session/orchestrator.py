"""The session machine: sequences blocks across fingers and hands the arms to each other.

One per session, owning both arms through `session.interlock`. A block is one finger under one
mode for a number of repetitions; the orchestrator runs them in order, and a change of finger
between consecutive blocks is what makes it hand control to arm 2 to move the hook.

Pure: no clock, no I/O, no hardware. It consumes what it is told (arm 1's status, arm 2's
outcome, a block finishing) and emits a command. That keeps the sequencing testable without a
robot, a camera or a browser, which is the rule the rest of the repo follows, and it is why the
interlock's refusals can be tested against every reachable state.

The safety-relevant part is not the sequencing, it is in `session.interlock`: arm 1 and arm 2
are never both able to move, and custody passes to arm 2 only on positive evidence that arm 1
is released and the tether is slack. This module's job is to never work around that. It has no
path that forces a handover and none that retries one; a refused handover makes it wait, and a
failed hook move stops the session.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from session.arm2 import HookMover, HookOutcome
from session.interlock import ArmOneStatus, Custody, Interlock, InterlockLimits
from session.protocol import Block, Protocol


class SessionState(Enum):
    """Where the session is. One block's worth of this cycles per block."""

    IDLE = "idle"
    #: Waiting for arm 1 to release and the tether to go slack so the hook can move. A normal
    #: state, not an error: it is where the interlock's evidence is being gathered.
    AWAITING_SLACK = "awaiting_slack"
    #: Arm 2 has custody and is moving the hook to this block's finger.
    POSITIONING = "positioning"
    #: Arm 1 has custody and is running the rep machine for this block.
    EXERCISING = "exercising"
    DONE = "done"
    #: Terminal. A human decides what happens next.
    ABORTED = "aborted"


class Command(Enum):
    """What the caller should do as a result of the last step. One per step, never two."""

    #: Nothing to do: keep feeding status. Most steps return this.
    WAIT = "wait"
    #: Unload arm 1's spring and stop it. Issued before every hook move.
    RELEASE_ARM1 = "release_arm1"
    #: Ask arm 2 to move the hook. The orchestrator has already requested it on the mover.
    MOVE_HOOK = "move_hook"
    #: Start arm 1's rep machine on `Step.block`.
    RUN_BLOCK = "run_block"
    #: Stop everything. Terminal.
    STOP = "stop"


@dataclass(frozen=True)
class Step:
    """The result of one `step()`: what to do, and why."""

    command: Command
    state: SessionState
    reason: str
    #: The block to run, set only with `Command.RUN_BLOCK`.
    block: Block | None = None
    #: The finger to hook, set only with `Command.MOVE_HOOK`.
    finger: str | None = None


class Orchestrator:
    """Runs a protocol across both arms. Feed it status, act on what it returns."""

    def __init__(
        self,
        protocol: Protocol,
        hook_mover: HookMover,
        interlock: Interlock | None = None,
        limits: InterlockLimits | None = None,
    ):
        self.protocol = protocol
        self.hook_mover = hook_mover
        self.interlock = interlock if interlock is not None else Interlock(limits)
        self.state = SessionState.IDLE
        self.block_index = 0
        #: Reps arm 1 reported per completed block, so a short block is visible afterwards
        #: rather than being rounded up to what was prescribed.
        self.completed_reps: list[int] = []
        #: The finger the hook is currently on, as far as anything has confirmed. `None` until
        #: the first confirmed move: the hook's position is never assumed.
        self.hooked_finger: str | None = None

    # ------------------------------------------------------------------ what is going on
    @property
    def block(self) -> Block | None:
        """The block being run or about to be run, or None when the protocol is finished."""
        if self.block_index >= len(self.protocol):
            return None
        return self.protocol.blocks[self.block_index]

    @property
    def finished(self) -> bool:
        return self.state in (SessionState.DONE, SessionState.ABORTED)

    @property
    def needs_hook_move(self) -> bool:
        """Whether this block's finger differs from where the hook is.

        Two consecutive blocks on one finger need no re-hooking, which is why
        `Protocol.fingers_in_order` collapses repeats.
        """
        block = self.block
        return block is not None and self.hooked_finger != block.finger

    # ------------------------------------------------------------------ the one entry point
    def step(self, arm1: ArmOneStatus) -> Step:
        """Advance the session by one decision, given what arm 1 looks like right now.

        `arm1` is required on every call, not only when it matters, because the interlock
        re-checks its evidence at the moment of handover and stale evidence is not evidence.

        **The caller owns reporting a mid-block release.** If arm 1's gate latches RELEASE
        while a block is running, this returns `WAIT` until something calls `abort()`: the
        session waits rather than ending, and nothing moves in the meantime. That is a
        deliberate gap, not an oversight. During `EXERCISING`, `arm1.released` becoming true
        is ambiguous between "the gate latched" and "the block just ended normally", and only
        the control loop knows which, because it knows the reason. Inferring it here would
        mean either ending blocks on a safety fault as though they had succeeded, or aborting
        sessions that merely finished. So whoever drives this loop has to pass the gate's
        latched decision to `abort()`.
        """
        if self.interlock.stopped:
            return self._stopped()
        if self.state is SessionState.DONE:
            return Step(Command.WAIT, self.state, "session complete")

        if self.state is SessionState.IDLE:
            return self._begin_block()
        if self.state is SessionState.AWAITING_SLACK:
            return self._try_handover(arm1)
        if self.state is SessionState.POSITIONING:
            return self._check_hook()
        if self.state is SessionState.EXERCISING:
            return Step(Command.WAIT, self.state, "arm 1 is exercising")
        return self._stopped()

    # ------------------------------------------------------------------ events from arm 1
    def block_finished(self, reps_completed: int) -> Step:
        """Arm 1 reports its rep machine reached the end of the block.

        `reps_completed` is recorded as reported rather than as prescribed: a patient who
        managed six of ten is a clinical fact, and the session log is the deliverable.
        """
        if self.state is not SessionState.EXERCISING:
            return Step(
                Command.WAIT, self.state,
                f"a block finished while the session was {self.state.value}; ignored",
            )
        self.completed_reps.append(reps_completed)
        self.interlock.release_all()
        self.block_index += 1
        if self.block is None:
            self.state = SessionState.DONE
            return Step(Command.RELEASE_ARM1, self.state, "every block is done")
        self.state = SessionState.IDLE
        return Step(Command.RELEASE_ARM1, self.state, "block finished, moving on")

    def abort(self, reason: str) -> Step:
        """Stop the session. Terminal, from any state, and never refused.

        The gate latching RELEASE, a human pressing stop, and the clinician stopping remotely
        all arrive here.
        """
        self.interlock.stop(reason)
        self.state = SessionState.ABORTED
        self.hook_mover.abort()
        return Step(Command.STOP, self.state, reason)

    # ------------------------------------------------------------------ internals
    def _begin_block(self) -> Step:
        block = self.block
        if block is None:
            self.state = SessionState.DONE
            return Step(Command.RELEASE_ARM1, self.state, "nothing left to run")
        if self.needs_hook_move:
            # Arm 1 has to be unloaded before the hook can move, so ask for the release and
            # then wait for the evidence rather than assuming the release worked.
            self.state = SessionState.AWAITING_SLACK
            return Step(
                Command.RELEASE_ARM1, self.state,
                f"the hook has to move to the {block.finger}, so arm 1 releases first",
            )
        return self._start_exercising(block, "the hook is already on this finger")

    def _try_handover(self, arm1: ArmOneStatus) -> Step:
        block = self.block
        assert block is not None  # AWAITING_SLACK is only entered with a block pending
        verdict = self.interlock.may_hand_to_arm2(arm1)
        if not verdict:
            # Waiting here is the designed behaviour, not a stall: the interlock is telling us
            # a finger may still be loaded. Nothing in this module may work around it.
            return Step(Command.WAIT, self.state, f"waiting for slack: {verdict.reason}")
        self.interlock.take_arm2(arm1)
        self.state = SessionState.POSITIONING
        # The hook's position is unknown from the moment it starts moving until it is
        # confirmed, so forget where it was rather than keeping a stale answer.
        self.hooked_finger = None
        self.hook_mover.request(block.finger)
        return Step(
            Command.MOVE_HOOK, self.state,
            f"arm 1 is released and slack; moving the hook to the {block.finger}",
            finger=block.finger,
        )

    def _check_hook(self) -> Step:
        outcome = self.hook_mover.poll()
        if outcome is None:
            return Step(Command.WAIT, self.state, "arm 2 is moving the hook")
        return self._hook_outcome(outcome)

    def _hook_outcome(self, outcome: HookOutcome) -> Step:
        block = self.block
        assert block is not None
        self.interlock.release_all()
        if not outcome.confirmed:
            # No retry. A hook in an unknown place is not something to resolve by trying the
            # same thing again, and the next step would be pulling on a finger.
            return self.abort(f"hook move to the {outcome.finger} failed: {outcome.detail}")
        if outcome.finger != block.finger:
            return self.abort(
                f"arm 2 confirmed the hook on the {outcome.finger} but this block is for the "
                f"{block.finger}; refusing to exercise a finger the hook was not confirmed on"
            )
        self.hooked_finger = outcome.finger
        return self._start_exercising(block, f"hook confirmed on the {outcome.finger}")

    def _start_exercising(self, block: Block, why: str) -> Step:
        verdict = self.interlock.may_hand_to_arm1()
        if not verdict:
            return Step(Command.WAIT, self.state, f"arm 1 cannot start: {verdict.reason}")
        self.interlock.take_arm1()
        self.state = SessionState.EXERCISING
        return Step(Command.RUN_BLOCK, self.state, why, block=block)

    def _stopped(self) -> Step:
        self.state = SessionState.ABORTED
        return Step(
            Command.STOP, self.state, self.interlock.stopped_reason or "session stopped"
        )

    # ------------------------------------------------------------------ for the dashboard
    @property
    def custody(self) -> Custody:
        return self.interlock.custody

    @property
    def progress(self) -> tuple[int, int]:
        """`(blocks finished, blocks total)`."""
        return len(self.completed_reps), len(self.protocol)
