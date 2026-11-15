"""Finger names, and the one place the repo's two vocabularies are reconciled.

This repo has two names for the fifth digit, both load-bearing, neither wrong:

``pinky``
    The tracking side. `hand_tracking.landmarks.FINGERS` keys on it, which makes it the keys
    of `all_flexion_deg()` and the column headers of the gauge CSV
    (`hand_tracking/cli.py`, `CSV_COLUMNS`), and `tests/hand_tracking/test_landmarks.py`
    asserts on it. Anything that *measures* a finger speaks this.

``little``
    The clinical side. `dashboard/app/lib/curlsim/types.ts` declares it as the `Finger` type,
    and `FLEXOR_MUSCLES` is keyed by it, so the stiffness and recovery modelling in
    `study.ts`, `generator.ts` and `MuscleCallout.tsx` all speak this. It is also the correct
    clinical term: hand therapy literature says "little finger" or "fifth digit", and "pinky"
    is colloquial.

A prescription is clinician-facing and ends up on the patient's screen, so the session layer
canonicalises on **little**. But a measurement comes off the hand tracker, so it arrives as
**pinky**, and a measurement script should not have to carry a translation table: the module
with the boundary owns the mapping, and this is it.

`normalise` therefore accepts either spelling everywhere the session layer takes a finger
name, and emits the clinical one. Renaming either vocabulary was considered and rejected:
renaming `landmarks.py` would break the CSV columns of a merged, working feature and buy
nothing clinically, and renaming curlsim would put a colloquialism in the clinician's
prescription and break its alignment with `FLEXOR_MUSCLES`.
"""
from __future__ import annotations

#: The canonical, clinician-facing names. Matches `curlsim/types.ts`'s `Finger` exactly, so a
#: `Block` round-trips to the dashboard untranslated.
FINGERS = ("thumb", "index", "middle", "ring", "little")

#: Accepted spellings that are not canonical, mapped to the canonical one. Only the fifth digit
#: differs today; this exists so a second divergence has an obvious home rather than becoming
#: an `if` somewhere.
ALIASES = {
    "pinky": "little",
    #: MANO's own ordering (index, middle, pinky, ring, thumb) uses "pinky" too, so a WiLoR
    #: source lands on the same alias.
    "fifth": "little",
    "little finger": "little",
}


def normalise(name: str) -> str:
    """The canonical name for `name`, accepting either vocabulary.

    Case and surrounding whitespace are ignored, because these names arrive from CSV headers
    and hand-written JSON as often as from code.
    """
    key = name.strip().lower()
    if key in FINGERS:
        return key
    if key in ALIASES:
        return ALIASES[key]
    raise ValueError(
        f"unknown finger {name!r}; known names are {list(FINGERS)} "
        f"(also accepted: {sorted(ALIASES)})"
    )


def normalise_keys(mapping) -> dict:
    """`normalise` every key of a mapping, rejecting two spellings of the same finger.

    A file carrying both `pinky` and `little` is ambiguous about which measurement is current,
    and silently keeping one would discard a real measurement, so it is an error.
    """
    out: dict = {}
    seen: dict[str, str] = {}
    for raw_key, value in mapping.items():
        key = normalise(str(raw_key))
        if key in out:
            raise ValueError(
                f"{seen[key]!r} and {raw_key!r} are both the {key}; a file cannot carry two "
                "spellings of one finger, because there is no way to tell which measurement "
                "is current"
            )
        seen[key] = str(raw_key)
        out[key] = value
    return out
