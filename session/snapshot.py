"""The one state shape the dashboard renders, identical for every mode.

The dashboard is a view and a command surface, not the brain: the browser can be closed, and
the patient-facing screen must not be the thing deciding when to stop pulling on a finger. So
session state lives here, in Python, and the dashboard subscribes to a stream of these.

The field that makes this work is `driver`. It is the only thing the UI needs in order to
decide whether the guide hand is a **demand** ("do this now") or a **preview** ("this is about
to happen to you"), which is a different animation and a different colour but the same panel.
Everything else renders the same way in both modes, so adding a third mode changes the
instruction table and nothing in the dashboard.

`driver` sits next to `phase` rather than inside `block`, because it flips within a repetition:
in `active` the patient drives the work beat and the device drives the reset, and in `passive`
it is the other way round.

This mirrors `hand_tracking/frame.py`: a dataclass here, a matching TypeScript type in
`dashboard/app/lib/sessionSnapshot.ts`, and the same schema version on both sides.

What this replaces: `GuideHand.tsx` currently runs its own `performance.now()` timeline over
hardcoded `REPS`, `HOLD_S` and `RELEASE_S`. That is why what the patient hears cannot be
aligned with what the arm is doing, by construction. Rendering this snapshot instead makes the
instruction, the animation and the force the same state by definition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lib.units import newtons_to_grams_force
from session.instructions import instruction_for, instruction_key
from session.modes import Driver, Law, Phase, plan_for
from session.protocol import Block

SCHEMA_VERSION = 2

_FORCE_PLACES = 3
_MM_PLACES = 1
_PROGRESS_PLACES = 3


@dataclass(frozen=True)
class SessionSnapshot:
    """Everything the dashboard needs for one frame of display.

    Deliberately flat and small: it is sent at the loop's rate, and anything derived that the
    UI can compute itself does not belong in it.
    """

    #: The block being exercised, or None when the session has not started.
    block: Block | None
    #: 1-based index of the repetition in progress, and the block's total.
    rep_index: int
    rep_total: int
    phase: Phase
    #: Which block this is within the protocol, and how many there are.
    block_index: int
    block_total: int
    #: Latest force reading in newtons, or None when the marker was not seen. Never a guess:
    #: a dropout is reported as a dropout, which is what the safety gate consumes.
    force_n: float | None
    #: Arm travel from the start of this repetition, in metres. Signed, positive in the
    #: slack direction.
    travel_m: float
    #: Repetitions actually counted, and attempts that started but did not complete.
    #:
    #: Not derivable from `rep_index`, which is only where the machine is now. These two are
    #: the clinical deliverable this layer exists to produce (`session/reps.py`): a repetition
    #: is counted only once the force reached the cap, so a patient who moved a little and
    #: came back shows as an abandoned attempt rather than as a rep. A screen that inferred
    #: "completed" from the index would report the number the machine was *attempting*, which
    #: is the dishonest counting the rep machine was built to replace.
    reps_completed: int = 0
    attempts_abandoned: int = 0
    #: 0 to 1 through the current phase where that is meaningful, else None. The UI shows a
    #: bar; it does not use this to decide anything.
    progress: float | None = None
    #: Set when the session has stopped and why, for the banner the clinician sees.
    stopped_reason: str | None = None

    @property
    def driver(self) -> Driver:
        """Who is the source of motion right now. The field the whole UI keys off."""
        if self.block is None:
            return Driver.DEVICE
        return plan_for(self.block.mode_spec, self.phase).driver

    @property
    def law(self) -> Law:
        return Law.STILL if self.block is None else plan_for(self.block.mode_spec, self.phase).law

    @property
    def device_driven(self) -> bool:
        """True when the arm, not the patient, is moving the finger in this phase."""
        if self.block is None:
            return False
        return plan_for(self.block.mode_spec, self.phase).device_driven

    @property
    def instruction(self) -> str:
        if self.block is None:
            return instruction_for("active", Phase.IDLE, "index")
        return instruction_for(self.block.mode, self.phase, self.block.finger)

    @property
    def instruction_key(self) -> str:
        mode = "active" if self.block is None else self.block.mode
        return instruction_key(mode, self.phase)

    def to_dict(self) -> dict[str, Any]:
        """The wire format. Keys are camelCase to match the dashboard's other payloads."""
        out: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "phase": self.phase.value,
            "driver": self.driver.value,
            "law": self.law.value,
            "deviceDriven": self.device_driven,
            "instruction": {"key": self.instruction_key, "text": self.instruction},
            "rep": {"index": self.rep_index, "of": self.rep_total,
                    "completed": self.reps_completed,
                    "abandoned": self.attempts_abandoned},
            "block": None,
            "blockIndex": {"index": self.block_index, "of": self.block_total},
            "forceN": None if self.force_n is None else round(self.force_n, _FORCE_PLACES),
            # Grams alongside newtons because the clinician's cap is set in grams and the
            # PESOLA is graduated in them, so this is the number a human compares against.
            "forceGf": None if self.force_n is None else round(newtons_to_grams_force(self.force_n), 1),
            "travelMm": round(self.travel_m * 1000.0, _MM_PLACES),
            "progress": None if self.progress is None else round(self.progress, _PROGRESS_PLACES),
            "stoppedReason": self.stopped_reason,
        }
        if self.block is not None:
            out["block"] = {
                "finger": self.block.finger,
                "mode": self.block.mode,
                "reps": self.block.reps,
                "capN": round(self.block.cap_n, _FORCE_PLACES),
                "capGf": round(self.block.cap_gf, 1),
                "holdS": self.block.hold_s,
                "romTargetMm": (
                    None
                    if self.block.rom_target_m is None
                    else round(self.block.rom_target_m * 1000.0, _MM_PLACES)
                ),
            }
        return out
