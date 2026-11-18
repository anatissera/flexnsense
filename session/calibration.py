"""The numbers only a bench measurement can give, held as explicit absences.

Every field here starts as `None`, and asking for one that has not been measured raises
`NotMeasured` carrying the procedure for measuring it. That is the whole point of this module:
the session layer can be written, tested and reviewed before the rig is characterised, without
a placeholder constant ever silently becoming the number the arm runs on. There is no default,
because a wrong default for any of these is a wrong force or a wrong distance applied to a
healing finger.

What is missing and why it matters:

``tether_n_per_m``
    Stiffness of the whole tether, spring plus cord plus attachments, not the spring alone.
    Sets how much stretch it takes to reach the cap, which is how much of the patient's range
    is spent on a rising ramp before any of it is the exercise. Fit it with
    `lib.stiffness.fit_stiffness` from the sweep described below.

``ticks_per_metre`` and ``yield_sign``
    How a displacement along the tether maps onto the servo's position register, and which way
    the register has to move to *reduce* tension. `control.loop.AxisCalibration` consumes both.
    Pose-dependent, so they are only valid for the pose the session runs in. Measured with a
    ruler against the arm, **not** inferred from the force: see the note in `HOW_TO_MEASURE`,
    which the measured 6 mm tether hysteresis settles.

``excursion_m``
    How far the attachment point on each fingertip travels along the tether axis through the
    prescribed range of motion. Sets the per-rep travel allowance and bounds `rom_target_m` in
    a passive prescription. Measured per finger, because they differ.

Both the estimate used by `control.simulate` and the reason it must not leak into the control
path are below.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from session.fingers import normalise, normalise_keys

#: Nominal stiffness of a PESOLA 600 g: full scale over ~110 mm of pointer travel. **For
#: `control.simulate` only.** It is a datasheet-shaped guess that ignores the cord and the
#: attachments, and the whole reason `tether_n_per_m` exists is that the real number has to
#: replace it before anything touches a finger. Never import this from a control path.
UNMEASURED_TETHER_ESTIMATE_N_PER_M = 53.9

#: Keyed by field name, the procedure that produces it. Raised to the caller verbatim, so a
#: person who hits the error gets the instructions rather than having to find this file.
HOW_TO_MEASURE: dict[str, str] = {
    "tether_n_per_m": (
        "Anchor one hook of the PESOLA to something fixed and pull the other by hand. At each "
        "of 50, 100, 150, 200, 250, 300, 400 and 500 gf on the dial, measure the distance "
        "between a mark on the anchor and a mark on the moving hook. Do it going up and coming "
        "back down, so the spring's own hysteresis is visible. Feed the (grams, mm) pairs to "
        "lib.stiffness.fit_stiffness and read k_n_per_m. No calibrated weights are needed: the "
        "PESOLA is the force reference."
    ),
    "ticks_per_metre": (
        "Measure the displacement with a RULER ON THE ARM, not by inferring it from the force. "
        "Lock the arm in the session pose (scripts/arm/lock_arm.py), with no finger anywhere near "
        "it. Mark the gripper's position against a fixed reference, command a known tick delta "
        "on the motor that moves along the tether axis, and measure how far the mark moved: "
        "ticks_per_metre = d_ticks / d_metres. Move as far as the pose safely allows, since the "
        "~1 mm ruler reading is a fixed error that a longer move divides down (3 % over 30 mm). "
        "Valid only for the pose it was measured in.\n\n"
        "    Do NOT use dx = dF / k for this, which an earlier version of these instructions "
        "recommended. The measured tether hysteresis is 6 mm (config/measurements/tether_stiffness.csv), and "
        "any displacement inferred from a force reading inherits that as a floor: 40 % error "
        "over a 15 mm move, and 10 % even over 60 mm, which is most of the tether's ~110 mm of "
        "usable travel. A ruler at 30 mm beats the force method at 100 mm by an order of "
        "magnitude."
    ),
    "yield_sign": (
        "Falls out of the ticks_per_metre measurement: it is the sign of the tick delta that "
        "made the force go *down*. Getting it backwards means yielding tightens the spring, "
        "which is why control.loop checks it against the measured force rather than trusting it."
    ),
    "excursion_m": (
        "Put the hand in the exact session pose and run `python -m hand_tracking run --jsonl "
        "out.jsonl`. The --jsonl flag is required: a bare `run` logs only flexion angles, not "
        "the world coordinates this needs. Move the finger slowly from full extension to full "
        "flexion. For each frame take the fingertip position MINUS the wrist position, both "
        "from `world`; do not difference the raw fingertip positions between frames, because "
        "MediaPipe's `world` is hand-centred and its origin shifts with the pose, so the "
        "difference would mix the fingertip's travel with that shift. The excursion is the "
        "change in that tip-minus-wrist vector between the two extremes, projected onto the "
        "tether axis (lib.geometry.Axis). Repeat per finger, or at least measure the worst "
        "case. Either finger vocabulary is accepted: the tracker says \"pinky\" and a "
        "prescription says \"little\", and session.fingers reconciles them."
    ),
}


class NotMeasured(RuntimeError):
    """Raised when something asks for a bench constant that has not been measured yet.

    Carries the procedure, so the failure is actionable where it happens.
    """

    def __init__(self, *fields: str):
        self.fields = tuple(fields)
        body = "\n\n".join(f"  {f}:\n    {HOW_TO_MEASURE[f]}" for f in fields)
        super().__init__(
            f"not measured yet: {', '.join(fields)}. This cannot run on a placeholder.\n\n"
            f"{body}\n\nSee docs/session-orchestration.md for what each number affects."
        )


@dataclass(frozen=True)
class BenchCalibration:
    """One rig, one pose. `None` means "not measured", never "zero" and never "default"."""

    tether_n_per_m: float | None = None
    ticks_per_metre: float | None = None
    #: +1 or -1: the direction the position register moves to reduce tension.
    yield_sign: int | None = None
    #: Finger name to excursion along the tether axis, in metres.
    excursion_m: Mapping[str, float] = field(default_factory=dict)
    #: Floor on any displacement inferred from a force reading, in metres.
    #:
    #: The tether's own hysteresis: at a given force the spring sits at a different length
    #: going up than coming back down (`lib.stiffness.compare_directions`, `.gap_mm`). Nothing
    #: that converts a force into a distance can do better than this, which makes it the error
    #: bar on two things at once. `ticks_per_metre` is measured through `dx = dF / k`, so it
    #: inherits this as a relative error (`uncertainty_fraction_at`), and the per-rep travel
    #: accounting cannot resolve a position better than this either. Optional: without it the
    #: constants are still usable, they just carry no stated error.
    displacement_uncertainty_m: float | None = None
    #: Free-text note on the pose and rig these numbers were taken in. They are not
    #: transferable to a different pose, so an unlabelled set is a trap.
    measured_in_pose: str | None = None

    def __post_init__(self) -> None:
        # Canonicalise the finger keys however the calibration was built, not only when it was
        # loaded from a file: a measurement script that constructs one directly speaks the
        # tracker's vocabulary ("pinky"), and a prescription looks it up as "little".
        object.__setattr__(self, "excursion_m", normalise_keys(self.excursion_m))
        if self.tether_n_per_m is not None and self.tether_n_per_m <= 0:
            raise ValueError(f"tether_n_per_m must be positive, got {self.tether_n_per_m}")
        if self.ticks_per_metre is not None and self.ticks_per_metre <= 0:
            raise ValueError(f"ticks_per_metre must be positive, got {self.ticks_per_metre}")
        if self.yield_sign is not None and self.yield_sign not in (1, -1):
            raise ValueError(f"yield_sign must be +1 or -1, got {self.yield_sign}")
        for finger, value in self.excursion_m.items():
            if value <= 0:
                raise ValueError(f"excursion for {finger!r} must be positive, got {value}")
        if self.displacement_uncertainty_m is not None and self.displacement_uncertainty_m < 0:
            raise ValueError(
                f"displacement_uncertainty_m cannot be negative, got "
                f"{self.displacement_uncertainty_m}"
            )

    # ------------------------------------------------------------------ what is still missing
    @property
    def missing(self) -> list[str]:
        """Field names with no measurement yet, in the order they have to be measured."""
        out = []
        if self.tether_n_per_m is None:
            out.append("tether_n_per_m")
        if self.ticks_per_metre is None:
            out.append("ticks_per_metre")
        if self.yield_sign is None:
            out.append("yield_sign")
        if not self.excursion_m:
            out.append("excursion_m")
        return out

    @property
    def complete(self) -> bool:
        return not self.missing

    # ------------------------------------------------------------------ guarded access
    def require_tether_n_per_m(self) -> float:
        if self.tether_n_per_m is None:
            raise NotMeasured("tether_n_per_m")
        return self.tether_n_per_m

    def require_axis(self) -> tuple[float, int]:
        """`(ticks_per_metre, yield_sign)`, for building a `control.loop.AxisCalibration`."""
        gaps = [f for f in ("ticks_per_metre", "yield_sign") if getattr(self, f) is None]
        if gaps:
            raise NotMeasured(*gaps)
        assert self.ticks_per_metre is not None and self.yield_sign is not None
        return self.ticks_per_metre, self.yield_sign

    def require_excursion_m(self, finger: str) -> float:
        """The excursion for `finger`, accepting either finger vocabulary.

        A measurement comes off the hand tracker as "pinky" and a prescription names it
        "little"; `session.fingers` owns that reconciliation, so neither side has to.
        """
        key = normalise(finger)
        if key not in self.excursion_m:
            raise NotMeasured("excursion_m")
        return self.excursion_m[key]

    def uncertainty_fraction_at(self, stretch_m: float) -> float | None:
        """Relative error on a distance inferred from force, over a stretch of `stretch_m`.

        `None` when the hysteresis has not been measured. This is what makes a short
        `ticks_per_metre` measurement untrustworthy: inferring the displacement from `dF / k`
        over a 2 mm move with a 1 mm hysteresis floor is a 50 % error on the constant, while
        the same floor over a 30 mm move is 3 %. Command a large tick delta, not a small one.
        """
        if self.displacement_uncertainty_m is None:
            return None
        if stretch_m <= 0:
            raise ValueError(f"stretch_m must be positive, got {stretch_m}")
        return self.displacement_uncertainty_m / stretch_m

    def stretch_to_reach_m(self, force_n: float) -> float:
        """How far the tether must stretch to reach `force_n`, from rest.

        The number that decides how much of a patient's range is run-up rather than exercise,
        so `session.reps` uses it to size the pre-tension in `Phase.ARMED`.
        """
        return force_n / self.require_tether_n_per_m()

    # ------------------------------------------------------------------ persistence
    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "BenchCalibration":
        known = {"tether_n_per_m", "ticks_per_metre", "yield_sign", "excursion_m",
                 "displacement_uncertainty_m", "measured_in_pose"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown calibration keys: {sorted(unknown)}")
        excursion = raw.get("excursion_m") or {}
        if not isinstance(excursion, Mapping):
            raise ValueError("excursion_m must be an object of finger name to metres")
        return cls(
            tether_n_per_m=_opt_float(raw.get("tether_n_per_m")),
            ticks_per_metre=_opt_float(raw.get("ticks_per_metre")),
            yield_sign=_opt_int(raw.get("yield_sign")),
            excursion_m={k: float(v) for k, v in excursion.items()},  # type: ignore[call-overload]
            displacement_uncertainty_m=_opt_float(raw.get("displacement_uncertainty_m")),
            measured_in_pose=_opt_str(raw.get("measured_in_pose")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "tether_n_per_m": self.tether_n_per_m,
            "ticks_per_metre": self.ticks_per_metre,
            "yield_sign": self.yield_sign,
            "excursion_m": dict(self.excursion_m),
            "displacement_uncertainty_m": self.displacement_uncertainty_m,
            "measured_in_pose": self.measured_in_pose,
        }

    @classmethod
    def load(cls, path: str | Path) -> "BenchCalibration":
        """Read a calibration file. A missing file is an empty calibration, not an error.

        Nothing can be measured before the rig exists, so "no file yet" is a normal state; the
        failure belongs at the point something tries to *use* an unmeasured number.
        """
        p = Path(path)
        if not p.is_file():
            return cls()
        return cls.from_dict(json.loads(p.read_text()))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def _opt_float(v: object) -> float | None:
    return None if v is None else float(v)  # type: ignore[arg-type]


def _opt_int(v: object) -> int | None:
    return None if v is None else int(v)  # type: ignore[arg-type]


def _opt_str(v: object) -> str | None:
    return None if v is None else str(v)
