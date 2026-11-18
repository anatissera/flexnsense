"""Merge-write helper shared by the bench-measurement scripts (stiffness, excursion,
ticks_per_metre) into one file shaped like `session.calibration.BenchCalibration.to_dict()`.

Not importing `BenchCalibration` itself: `session/` lives on `feature/session-orchestration`,
not this branch, so a script here cannot import it yet. This reads and writes the same six keys
by hand and rejects anything else, so once `session/` lands the switch to
`BenchCalibration.save()` is a straight swap, not a migration.

`config/calibration/` is already the committed home for calibration code consumes (the arm's
joint-range calibration, `follower-<serial>.json`), so the bench constants live there too,
alongside `bench.md` documenting them - not in `data/`, which holds raw sweeps
(`config/measurements/tether_stiffness.csv`) rather than fitted constants.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_PATH = "config/calibration/bench.json"

#: Must stay identical to `session.calibration.BenchCalibration.from_dict`'s known-key set
#: (tests/lib/test_bench_calibration_io.py pins this list so a drift fails loudly here rather
#: than only showing up as a surprise once that module lands on this branch).
KNOWN_KEYS = {
    "tether_n_per_m", "ticks_per_metre", "yield_sign", "excursion_m",
    "displacement_uncertainty_m", "measured_in_pose",
}


def load(path: str | Path = DEFAULT_PATH) -> dict[str, Any]:
    """A missing file is an empty calibration, not an error: nothing has been measured yet."""
    p = Path(path)
    if not p.is_file():
        return {"excursion_m": {}}
    data = json.loads(p.read_text())
    unknown = set(data) - KNOWN_KEYS
    if unknown:
        raise ValueError(f"{p}: unknown calibration key(s) {sorted(unknown)}")
    data.setdefault("excursion_m", {})
    return data


def save(updates: dict[str, Any], path: str | Path = DEFAULT_PATH) -> dict[str, Any]:
    """Merge `updates` into the file at `path` and write it back. Returns the merged dict.

    `excursion_m` merges finger by finger; every other key is replaced outright, since each
    measurement script is the sole source for its own key.
    """
    unknown = set(updates) - KNOWN_KEYS
    if unknown:
        raise ValueError(f"unknown calibration key(s): {sorted(unknown)}")
    data = load(path)
    for key, value in updates.items():
        if key == "excursion_m":
            data["excursion_m"].update(value)
        else:
            data[key] = value
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n")
    return data
