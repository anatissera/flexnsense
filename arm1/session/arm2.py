"""Arm 2, the hook mover, behind one interface with three implementations.

The orchestrator only ever needs one thing from arm 2: "move the hook to finger F, tell me when
you are done or when you failed". Keeping that the whole contract is what lets the session run
end to end today, because **the trained policy does not exist yet**: the demonstrations are
being recorded (`arm2/data_collection/`, the `JustoDario/flexnsense_skill1` dataset), and the
recording README is explicit that training and running a policy on the arm are not covered.

So the implementation that actually runs is `ManualHookMover`: a person moves the hook and
confirms on the dashboard. Nothing above this module knows which implementation is in use, and
nothing should.

Why the interface is poll-based rather than something to await: the orchestrator
(`session.orchestrator`) is a pure state machine with no clock and no I/O, so it asks for an
outcome and gets `None` while the move is still in progress. Whatever blocking, async or
hardware work an implementation needs stays inside it.

**Success is reported, never assumed.** `PolicyHookMover` is a deliberate stub that raises
rather than returning a plausible outcome: a policy that silently reports success it did not
achieve would have the orchestrator start pulling on a finger the hook is not attached to.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from arm1.session.fingers import normalise


class HookResult(Enum):
    """How a hook move ended."""

    #: The hook is on the requested finger and confirmed to be there.
    CONFIRMED = "confirmed"
    #: The move did not happen, or could not be confirmed. The orchestrator stops the session:
    #: it never retries on its own, because a hook in an unknown place is not something to
    #: resolve by trying the same thing again.
    FAILED = "failed"


@dataclass(frozen=True)
class HookOutcome:
    """The end of one hook move."""

    result: HookResult
    finger: str
    #: Why it failed, or how it was confirmed. Goes into the session log either way.
    detail: str = ""

    @property
    def confirmed(self) -> bool:
        return self.result is HookResult.CONFIRMED


@runtime_checkable
class HookMover(Protocol):
    """What the orchestrator requires of arm 2. Three implementations satisfy it."""

    def request(self, finger: str) -> None:
        """Start moving the hook to `finger`. Returns immediately."""
        ...

    def poll(self) -> HookOutcome | None:
        """The outcome, or None while the move is still in progress."""
        ...

    def abort(self) -> None:
        """Stop and give up on the current move. Must be safe to call at any time."""
        ...


class ManualHookMover:
    """A person moves the hook and confirms it. The implementation that runs today.

    Also the honest fallback for the other two: if a policy is unavailable or has failed, a
    human with the same interface keeps the session working, and the orchestrator cannot tell
    the difference.

    The confirmation is a deliberate human action, `confirm()`, which is why this is safer than
    it looks: the thing being confirmed is "I can see the hook on the index finger", which no
    vision check currently in this repo can assert.
    """

    def __init__(self) -> None:
        self._pending: str | None = None
        self._outcome: HookOutcome | None = None

    @property
    def awaiting_confirmation(self) -> str | None:
        """The finger a person is being asked to hook, or None if nothing is pending.

        The dashboard renders a prompt from this.
        """
        return self._pending

    def request(self, finger: str) -> None:
        self._pending = normalise(finger)
        self._outcome = None

    def confirm(self) -> None:
        """A person states the hook is on the requested finger."""
        if self._pending is None:
            raise RuntimeError("nothing to confirm: no hook move was requested")
        self._outcome = HookOutcome(HookResult.CONFIRMED, self._pending, "confirmed by operator")
        self._pending = None

    def report_failure(self, detail: str) -> None:
        """A person states the hook could not be placed, with what went wrong."""
        if self._pending is None:
            raise RuntimeError("nothing to fail: no hook move was requested")
        self._outcome = HookOutcome(HookResult.FAILED, self._pending, detail)
        self._pending = None

    def poll(self) -> HookOutcome | None:
        outcome, self._outcome = self._outcome, None
        return outcome

    def abort(self) -> None:
        self._pending = None
        self._outcome = None


class ReplayHookMover:
    """Replays a recorded teleoperated trajectory for each finger.

    Not implemented. Kept as a named class rather than a comment because it is the middle rung
    between a human and a policy, and the recordings it would need already exist.

    It would still need a confirmation step: replaying a trajectory open-loop puts the hook
    wherever the hand happens to be now, which is not necessarily where it was during the
    recording.
    """

    def __init__(self, episodes: dict[str, object] | None = None):
        raise NotImplementedError(
            "ReplayHookMover is not implemented. Recorded episodes exist "
            "(JustoDario/flexnsense_skill1, in Station's Parquet format), but replaying one "
            "open-loop would place the hook wherever the hand is now rather than where it was "
            "during the recording, so it needs a confirmation step before it is usable. Use "
            "ManualHookMover."
        )


class PolicyHookMover:
    """The trained behaviour-cloning policy.

    Not implemented, and raises rather than stubbing out a success. A hook mover that reports
    success it did not achieve would have the orchestrator start pulling on a finger the hook
    is not attached to, which is the failure this whole interface exists to prevent.

    What is still missing, as of this commit: the policy itself, and the format gap between
    Station's own Parquet recordings and what the SmolVLA and ACT notebooks consume
    (`colab_train/`). See docs/session-orchestration.md.
    """

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "PolicyHookMover is not implemented: no policy has been trained for the hook move "
            "yet, and the Station-Parquet to LeRobotDataset conversion has no owner. This "
            "raises rather than reporting a plausible outcome, because a hook mover that claims "
            "success it did not achieve would have the session pull on a finger the hook is not "
            "attached to. Use ManualHookMover."
        )
