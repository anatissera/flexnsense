"""The exercise prescription, as data.

Everything a session needs to be decided is known before it starts: which finger, which mode,
how many reps, what force cap, how long to hold, how far to stretch. There is nothing to
learn, so this is a file the state machine reads, not weights. "Ten reps on the index finger"
is a number in a prescription; it is not something a policy is trained to contemplate. That is
what lets a clinician change it without retraining anything, and what lets the sequencing be
unit tested.

Units, per docs/contributing.md: the clinician sets a cap in grams because that is what the PESOLA reads,
and `Block.cap_n` stores it in newtons because that is what the control math runs on. Use
`Block.from_grams` to build one from a clinician's number so the conversion happens in one
place.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from lib.units import grams_force_to_newtons, newtons_to_grams_force
from session.calibration import BenchCalibration
from session.fingers import FINGERS, normalise
from session.modes import Mode, mode_by_name

__all__ = ["FINGERS", "Block", "Protocol", "PRETENSION_FRACTION"]

#: The fraction of the cap `Phase.ARMED` pre-tensions the tether to, and therefore the force a
#: repetition starts from. Lives here rather than only in `session.reps.RepLimits` because
#: `Block.check_against` needs it to tell whether a cap allows the stretch it prescribes, and
#: `session.reps` imports this module (so the dependency cannot run the other way). `RepLimits`
#: reads it from here, which keeps one number in one place.
PRETENSION_FRACTION = 0.25


@dataclass(frozen=True)
class Block:
    """One finger, one mode, a number of repetitions. The unit the orchestrator sequences."""

    finger: str
    mode: str
    reps: int
    #: The clinician's force cap, in newtons. Above this the arm gives way (patient-driven
    #: phases) or stops advancing (device-driven phases).
    cap_n: float
    #: Seconds to hold at end of range before the reset beat.
    hold_s: float = 3.0
    #: How far a device-driven pull may stretch the finger, in metres, measured from where the
    #: repetition started. Required for a mode that pulls, meaningless for one that does not:
    #: under the cap nothing else tells a pulling arm to stop.
    rom_target_m: float | None = None

    def __post_init__(self) -> None:
        if self.finger not in FINGERS:
            # Raises with the accepted spellings; a clinician writing "pinky" in a protocol
            # file should be told the canonical name, not just rejected.
            raise ValueError(
                f"block names the finger {self.finger!r}, which is not canonical. Use "
                f"{normalise(self.finger)!r}."
                if _is_alias(self.finger)
                else f"unknown finger {self.finger!r}; known fingers are {list(FINGERS)}"
            )
        mode = mode_by_name(self.mode)  # raises with the list of known modes
        if self.reps < 1:
            raise ValueError(f"reps must be at least 1, got {self.reps}")
        if self.cap_n <= 0:
            raise ValueError(f"cap_n must be positive, got {self.cap_n}")
        if self.hold_s < 0:
            raise ValueError(f"hold_s cannot be negative, got {self.hold_s}")
        if mode.needs_rom_target and self.rom_target_m is None:
            raise ValueError(
                f"mode {self.mode!r} pulls the finger, so it needs rom_target_m: a device-driven "
                "pull is not bounded by force alone, because under the cap nothing tells it to "
                "stop"
            )
        if not mode.needs_rom_target and self.rom_target_m is not None:
            raise ValueError(
                f"mode {self.mode!r} never pulls toward a target, so rom_target_m has no meaning "
                "here; the patient's own range is the limit"
            )
        if self.rom_target_m is not None and self.rom_target_m <= 0:
            raise ValueError(f"rom_target_m must be positive, got {self.rom_target_m}")

    @classmethod
    def from_grams(cls, finger: str, mode: str, reps: int, cap_gf: float, **kw) -> "Block":
        """Build a block from a cap in grams-force, which is the unit a clinician sets."""
        return cls(finger=finger, mode=mode, reps=reps, cap_n=grams_force_to_newtons(cap_gf), **kw)

    @property
    def cap_gf(self) -> float:
        """The cap back in grams, for anything a human reads."""
        return newtons_to_grams_force(self.cap_n)

    @property
    def mode_spec(self) -> Mode:
        return mode_by_name(self.mode)

    def check_against(self, calib: BenchCalibration,
                      pretension_fraction: float = PRETENSION_FRACTION) -> list[str]:
        """Problems that only the bench numbers can reveal. Empty list means nothing to say.

        Returned rather than raised: an unmeasured rig is a normal state for a prescription to
        be written in, so this is a warning channel, not validation. The hard failure belongs
        where something tries to *run* on an unmeasured number (`session.calibration`).

        `pretension_fraction` is what `session.reps.RepLimits` will actually arm to, and it
        matters here because a repetition's stretch is measured from the *rep's* start, where
        the tether is already pre-tensioned, not from true slack. Counting from slack
        overstates the stretch a cap allows by exactly that fraction.
        """
        problems: list[str] = []
        excursion = calib.excursion_m.get(self.finger)
        if self.rom_target_m is not None and excursion is not None:
            if self.rom_target_m > excursion:
                problems.append(
                    f"rom_target_m {self.rom_target_m * 1000:.0f} mm exceeds the measured "
                    f"excursion of the {self.finger} ({excursion * 1000:.0f} mm), so the pull "
                    "would reach the end of the finger's range before reaching its target"
                )
        if self.rom_target_m is not None and calib.tether_n_per_m is not None:
            # Whether this prescription's own cap allows its own stretch. A pulling arm stops
            # at the cap, so a target further out than the cap permits is a stretch the
            # session can never perform: it ends short every repetition and reports "stopped
            # at the cap" forever, which reads as a stiff patient when it is actually an
            # inconsistent prescription. Found by writing an example protocol that had exactly
            # this wrong (90 gf cap, 12 mm target, 56.2 N/m tether: the cap arrives at 9.6 mm).
            # Measured from the rep's start, where ARMED has already taken the tether to
            # `pretension_fraction` of the cap, so only the remainder is available as stretch.
            headroom_n = self.cap_n * (1.0 - pretension_fraction)
            reachable_m = calib.stretch_to_reach_m(headroom_n)
            if self.rom_target_m > reachable_m:
                problems.append(
                    f"the {self.cap_gf:.0f} gf cap is reached after "
                    f"{reachable_m * 1000:.0f} mm of stretch from the rep's start (the tether "
                    f"is already at {newtons_to_grams_force(self.cap_n * pretension_fraction):.0f} "
                    f"gf when it begins), short of the {self.rom_target_m * 1000:.0f} mm "
                    "rom_target_m, so the arm would stop at the cap every repetition and never "
                    f"reach the target. Raise the cap to about "
                    f"{newtons_to_grams_force(calib.require_tether_n_per_m() * self.rom_target_m / (1.0 - pretension_fraction)):.0f} "
                    "gf, or lower the target"
                )
        if calib.tether_n_per_m is not None and excursion is not None:
            ramp_m = calib.stretch_to_reach_m(self.cap_n)
            if ramp_m >= excursion:
                problems.append(
                    f"reaching the {self.cap_gf:.0f} gf cap takes {ramp_m * 1000:.0f} mm of "
                    f"stretch, which is the whole {excursion * 1000:.0f} mm excursion of the "
                    f"{self.finger}: the patient would never reach the cap. Pre-tension in "
                    "ARMED has to cover most of that, or the cap is too high for this tether"
                )
        return problems

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "finger": self.finger,
            "mode": self.mode,
            "reps": self.reps,
            "cap_n": self.cap_n,
            "hold_s": self.hold_s,
        }
        if self.rom_target_m is not None:
            out["rom_target_m"] = self.rom_target_m
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "Block":
        known = {"finger", "mode", "reps", "cap_n", "cap_gf", "hold_s", "rom_target_m"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown block keys: {sorted(unknown)}")
        if ("cap_n" in raw) == ("cap_gf" in raw):
            raise ValueError("a block needs exactly one of cap_n or cap_gf, not both or neither")
        cap_n = (
            float(raw["cap_n"])  # type: ignore[arg-type]
            if "cap_n" in raw
            else grams_force_to_newtons(float(raw["cap_gf"]))  # type: ignore[arg-type]
        )
        rom = raw.get("rom_target_m")
        return cls(
            # Normalised on the way in, so a file written against the tracker's vocabulary
            # loads without the author having to know which spelling the session layer uses.
            finger=normalise(str(raw["finger"])),
            mode=str(raw["mode"]),
            reps=int(raw["reps"]),  # type: ignore[arg-type]
            cap_n=cap_n,
            hold_s=float(raw.get("hold_s", 3.0)),  # type: ignore[arg-type]
            rom_target_m=None if rom is None else float(rom),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class Protocol:
    """An ordered list of blocks: the whole session.

    The order is the order the orchestrator runs them in, and a change of finger between
    consecutive blocks is what tells it to hand control to arm 2 to move the hook.
    """

    blocks: Sequence[Block]
    #: The absolute force ceiling for the session. No block's cap may reach it, and the safety
    #: gate latches RELEASE above it. Kept here rather than per block because it is a property
    #: of the patient and the rig, not of one exercise.
    ceiling_n: float
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("a protocol needs at least one block")
        if self.ceiling_n <= 0:
            raise ValueError(f"ceiling_n must be positive, got {self.ceiling_n}")
        for i, b in enumerate(self.blocks):
            if b.cap_n >= self.ceiling_n:
                raise ValueError(
                    f"block {i} ({b.finger}, {b.mode}) has cap {b.cap_gf:.0f} gf at or above the "
                    f"session ceiling of {newtons_to_grams_force(self.ceiling_n):.0f} gf; the cap "
                    "has to leave the controller headroom to give way in before the gate releases"
                )

    def __iter__(self) -> Iterator[Block]:
        return iter(self.blocks)

    def __len__(self) -> int:
        return len(self.blocks)

    @property
    def total_reps(self) -> int:
        return sum(b.reps for b in self.blocks)

    @property
    def fingers_in_order(self) -> list[str]:
        """Each finger the hook has to be moved to, in order, collapsing consecutive repeats.

        What the orchestrator turns into arm 2 tasks: two blocks on the same finger need no
        re-hooking between them.
        """
        out: list[str] = []
        for b in self.blocks:
            if not out or out[-1] != b.finger:
                out.append(b.finger)
        return out

    def check_against(self, calib: BenchCalibration) -> list[str]:
        """Every block's bench-dependent warnings, prefixed with which block they came from."""
        problems = []
        for i, b in enumerate(self.blocks):
            problems += [f"block {i} ({b.finger}, {b.mode}): {p}" for p in b.check_against(calib)]
        return problems

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "ceiling_n": self.ceiling_n,
            "blocks": [b.to_dict() for b in self.blocks],
        }
        if self.name is not None:
            out["name"] = self.name
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "Protocol":
        unknown = set(raw) - {"blocks", "ceiling_n", "ceiling_gf", "name"}
        if unknown:
            raise ValueError(f"unknown protocol keys: {sorted(unknown)}")
        if ("ceiling_n" in raw) == ("ceiling_gf" in raw):
            raise ValueError("a protocol needs exactly one of ceiling_n or ceiling_gf")
        ceiling = (
            float(raw["ceiling_n"])  # type: ignore[arg-type]
            if "ceiling_n" in raw
            else grams_force_to_newtons(float(raw["ceiling_gf"]))  # type: ignore[arg-type]
        )
        blocks = raw.get("blocks")
        if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
            raise ValueError("blocks must be a list")
        name = raw.get("name")
        return cls(
            blocks=[Block.from_dict(b) for b in blocks],  # type: ignore[arg-type]
            ceiling_n=ceiling,
            name=None if name is None else str(name),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Protocol":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def _is_alias(name: str) -> bool:
    """Whether `name` is a known non-canonical spelling, so the error can name the fix."""
    try:
        return normalise(name) != name
    except ValueError:
        return False
