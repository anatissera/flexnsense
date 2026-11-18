"""The machine's own settings: what the bench proved, so nobody has to retype it.

This is the other half of the split that `session/protocol.py` starts. A `Block` there holds
what a *clinician* decides, per patient: which finger, which mode, how many reps, what force
cap, how long to hold. Everything in this file is what the *rig* needs in order to carry that
out at all, and a clinician has no business choosing it: virtual damping, the camera's outlier
window, how fast three joints may chase a goal. The dividing question is "would a different
patient change this number?" If yes it belongs in a prescription, if no it belongs here.

Why a committed file and not argparse defaults. These values were found by debugging against
the real arm over many sessions, and several of them are not preferences, they are each paying
for a specific observed failure (the notes below say which). Held only as defaults inside
`scripts/session/run_session.py`, they are reachable by one script and invisible to everything else,
so the dashboard would have had to hardcode its own copy and the two would drift. Held here,
the CLI and the dashboard read the same numbers, and changing one is a reviewable diff against
a named profile rather than a flag someone did or did not remember to type.

**Not the same kind of thing as `session/calibration.py`.** That module refuses to have
defaults at all, on purpose: its numbers describe the physical rig, and a wrong default there
is a wrong force or a wrong distance applied to a healing finger. The numbers here are
tuning: a wrong one makes the arm jerky, or slow, or quick to abandon a repetition, and the
safety layer is still between all of them and the patient. So defaults are not only allowed
here, shipping them is the entire point.

Units follow docs/contributing.md, with the boundary in the same place as everywhere else: the fields a
person reads off an instrument or types at a prompt stay in grams-force and millimetres, and
the conversion to newtons and metres happens where the control code reads them. The field
names say which, so `v_max_mm_s` is millimetres and `deadband_n` is newtons.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

__all__ = ["SessionTuning", "SafetyTuning", "load_tuning", "DEFAULT_TUNING_PATH"]

#: The committed profile every entry point reads unless told otherwise. Lives next to
#: `config/calibration/`, which is committed for the same reason: it describes the shared rig,
#: not one person's machine (docs/contributing.md).
DEFAULT_TUNING_PATH = Path(__file__).resolve().parent.parent / "config" / "session-tuning.json"


@dataclass(frozen=True)
class SafetyTuning:
    """The settings that decide when the gate gives up, kept in their own block deliberately.

    Three of these are *looser* than the conservative values the code shipped with, because
    the conservative values stopped working sessions on this rig. That is a legitimate thing
    to change and an illegitimate thing to hide, so they live in a block of their own with the
    reason attached, rather than mixed in with the damping and the frame rate where a reviewer
    would skim past them. Anything that reads this file should surface this block to whoever is
    about to run a session. The register-level current cap in Station's `station.yaml` is
    unaffected by every field here and remains the backstop that does not depend on our code
    (docs/elrobot.md).
    """

    #: The hard ceiling: above this the gate latches RELEASE unconditionally, whatever else is
    #: true. **Looser than the 200 gf clinical cap**, and set to near the PESOLA's 600 gf full
    #: scale, because the camera emits bogus spikes of a few frames (see `outlier_persist`)
    #: that reached 559 gf with the finger at rest. At 200 a single bogus frame ended the
    #: session. This is the ceiling on *believing a reading*, not a licence to pull this hard:
    #: `Block.cap_n` is what the arm actually works against, and it stays clinical.
    ceiling_gf: float = 590.0
    #: Start giving way this far below the cap, so the arm already has speed by the time the
    #: reading reaches it instead of starting from a standstill there. Pays for loop latency.
    anticipation_gf: float = 5.0
    #: Consecutive frames with no usable reading before the gate releases. **Much looser than
    #: the 2 it shipped with**: the real loop runs well under the camera's nominal 30 fps and
    #: drops frames in bursts, so 2 released constantly in the middle of healthy repetitions.
    #: Does not cover a device-driven phase, which has its own stricter deadman below.
    max_dropouts: int = 45
    #: A reading older than this is treated as stale rather than as a reading.
    max_age_s: float = 0.5
    #: Whether to release when yielding fails to lower the force within `yield_check_window_s`.
    #: **Off**, unlike the shipped default: the check also fires when a person flexes faster
    #: than `v_max_mm_s`, which is real but harmless here, and it fired on every session on
    #: this bench. The ceiling and the dropout count are untouched by this and still run.
    yield_check: bool = False
    #: With `yield_check` on, how long yielding may run without the force coming down before
    #: concluding the direction is wrong.
    yield_check_window_s: float = 0.5
    #: `session.reps.RepMachine`'s own deadman: how long a device-driven phase that can load
    #: the finger may go with no usable reading before the repetition is abandoned. Strict,
    #: because a lost reading here leaves the arm moving a finger with no idea how hard.
    blind_grace_s: float = 0.3
    #: The same, for a phase that can only ever *reduce* tension. Longer on purpose: the
    #: successful end of such a phase is exactly when the tether goes slack, which is when the
    #: camera loses the PESOLA's markers, so the strict grace aborted resets for succeeding.
    slack_blind_grace_s: float = 3.0


@dataclass(frozen=True)
class SessionTuning:
    """Everything a session needs that is not in the prescription. See the module docstring."""

    #: Which profile this is, for the log and the dashboard. Not used in any arithmetic.
    profile: str = "default"
    #: Free text saying where the numbers came from. Not used in any arithmetic.
    note: str = ""

    # --- the admittance law (control/admittance.py) ---
    #: Virtual damping B in `v = (F - F_threshold) / B`, N.s/m. Larger gives way more slowly.
    damping_n_s_per_m: float = 10.0
    #: Speed ceiling along the tether, whatever the overshoot.
    v_max_mm_s: float = 35.0
    #: Overshoot below this is treated as zero, so resting noise does not twitch the arm.
    deadband_n: float = 0.05
    #: Largest displacement in one cycle. Bounds the *yield* step's smoothness against a
    #: flexing finger, and nothing else: the recovery step has its own bound below, because
    #: sharing one number meant speeding up the return also loosened this.
    max_step_mm: float = 4.0
    #: Total travel along the tether before the gate releases.
    max_travel_mm: float = 80.0

    # --- following the finger back (control/loop.py) ---
    #: How eagerly recovery chases slack, relative to how eagerly yielding gives way. 0
    #: disables it and leaves the arm parked wherever the furthest flex pushed it.
    recover_gain: float = 1.0
    #: Once the force reaches the recovery floor and the law has nothing further to say, keep
    #: retracing what is left at this fixed, slow speed. Not optional in practice: a finger
    #: relaxes faster than the arm can follow, so the law reaches its floor with travel still
    #: owed and stops dead short of home. Deliberately well under `v_max_mm_s`, since there is
    #: no force reading left to justify any urgency once this is the only thing moving.
    recover_retrace_mm_s: float = 25.0
    #: Stop chasing the force down at this reading rather than at true slack. `null` means the
    #: session's own resting tension, which is the right answer unless measured otherwise: a
    #: tether recovered past that goes slack enough to lose the camera's markers.
    recover_floor_gf: float | None = None

    # --- the pull law, for a device-driven mode (control/loop.py PullPlan) ---
    #: How fast a device-driven pull travels. A setting rather than a computed response:
    #: under the cap the force says nothing about how fast to move a relaxed finger. Quicker
    #: than the yield speed is legitimate here, because the path is decided before the stroke
    #: begins instead of being chased frame by frame, which is also what makes a `cpm`
    #: repetition the same motion every time.
    pull_mm_s: float = 20.0
    #: Largest displacement a pull may take in one cycle, separate from `max_step_mm` for the
    #: same reason the recovery step's is: that number bounds the yield step's smoothness
    #: against a flexing finger, and a pull is not that motion.
    pull_max_step_mm: float = 4.0
    #: How fast ARMED takes up slack. Slower than `pull_mm_s` on purpose: it is the one pull
    #: that starts with the tether slack and a finger already through the hook, so it is the
    #: stroke that knows least about what it is about to touch.
    arm_mm_s: float = 8.0

    # --- reading the gauge (gauge_reader/) ---
    #: How many consecutive held-back-looking readings the outlier gate needs before trusting
    #: a sudden jump as real. Higher than the gauge reader's own default of 3 because a bogus
    #: ~555 gf reading persisted for exactly 3 frames on this bench and was trusted.
    outlier_persist: int = 5

    # --- pulling with three joints (control/loop.py CoordinatedAxis) ---
    #: Whether to pull along the tether with several joints at once. On, because no single
    #: joint on this arm moves the gripper along the tether's line.
    coordinated: bool = True
    #: Drop the z component of the tether direction and renormalise, keeping the pull
    #: horizontal instead of following the gripper's own tilt.
    level: bool = True
    #: Raw ST3215 goal-speed units, scaled per joint each cycle so all joints arrive together.
    #: Re-tune with scripts/arm/jog_coordinated.py if the arm's load changes.
    coordinated_speed: int = 900
    #: Raw ST3215 acceleration units, written once.
    coordinated_accel: int = 30

    safety: SafetyTuning = SafetyTuning()

    def __post_init__(self) -> None:
        # Guard the values whose meaning breaks at zero or below, rather than letting a
        # typo'd profile reach the arm. `recover_gain` and `deadband_n` are allowed to be
        # zero: both mean "disabled", which is a real choice.
        for name in ("damping_n_s_per_m", "v_max_mm_s", "max_step_mm", "max_travel_mm",
                     "pull_mm_s", "pull_max_step_mm", "arm_mm_s",
                     "outlier_persist", "coordinated_speed", "coordinated_accel"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name in ("recover_gain", "deadband_n", "recover_retrace_mm_s"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} cannot be negative, got {value}")
        if self.recover_floor_gf is not None and self.recover_floor_gf <= 0:
            raise ValueError(
                f"recover_floor_gf must be positive or null, got {self.recover_floor_gf}"
            )
        for name in ("ceiling_gf", "max_dropouts", "max_age_s", "yield_check_window_s",
                     "blind_grace_s", "slack_blind_grace_s"):
            value = getattr(self.safety, name)
            if value <= 0:
                raise ValueError(f"safety.{name} must be positive, got {value}")
        if self.safety.anticipation_gf < 0:
            raise ValueError(
                f"safety.anticipation_gf cannot be negative, got {self.safety.anticipation_gf}"
            )

    def as_dict(self) -> dict[str, Any]:
        """The profile as plain JSON-able data, for the dashboard and the session log."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SessionTuning":
        """Build from parsed JSON, rejecting unknown keys rather than ignoring them.

        A silently ignored key is the failure mode worth spending code on here: someone edits
        the profile, misspells `max_step_mm`, sees no error, and runs a session believing a
        number that was never read. Both the top level and the nested safety block are
        checked, since the safety block is exactly where a quiet miss matters most.
        """
        known = {f.name for f in fields(cls)} - {"safety"}
        safety_data = data.get("safety", {})
        if not isinstance(safety_data, Mapping):
            raise ValueError(f"'safety' must be an object, got {type(safety_data).__name__}")
        known_safety = {f.name for f in fields(SafetyTuning)}
        unknown = sorted(set(data) - known - {"safety"})
        if unknown:
            raise ValueError(
                f"unknown tuning key(s) {unknown}; known keys are {sorted(known | {'safety'})}"
            )
        unknown_safety = sorted(set(safety_data) - known_safety)
        if unknown_safety:
            raise ValueError(
                f"unknown safety tuning key(s) {unknown_safety}; "
                f"known keys are {sorted(known_safety)}"
            )
        return cls(
            **{k: v for k, v in data.items() if k in known},
            safety=SafetyTuning(**dict(safety_data)),
        )


def load_tuning(path: str | Path | None = None) -> SessionTuning:
    """The committed tuning profile, or the built-in defaults if the file is absent.

    A missing file is not an error, unlike a missing bench calibration: every field here has a
    defensible default that is the same number the file holds, so the built-ins and the
    committed profile agree and the code runs either way. A file that exists but does not parse
    *is* an error, because that means someone meant to say something and it was not understood.
    """
    path = Path(DEFAULT_TUNING_PATH if path is None else path)
    if not path.exists():
        return SessionTuning()
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must hold a JSON object, got {type(data).__name__}")
    try:
        return SessionTuning.from_mapping(data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
