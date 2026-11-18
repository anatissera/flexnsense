"""Fingertip excursion along the tether axis, from a recorded hand-tracking session.

Fills `excursion_m` in `config/calibration/bench.json`, shaped like
`session.calibration.BenchCalibration.to_dict()` (that module isn't on this branch yet, so this
writes the same keys by hand - see lib/bench_calibration_io.py). No ruler on the hand, no robot:
just the camera and a slow, deliberate motion.

Record one finger at a time, in the exact session pose, moving it slowly from full extension to
full flexion and back:

    .venv/bin/python -m hand_tracking run --jsonl data/excursion-middle.jsonl

The CSV `hand_tracking run` writes by default has angles only; the world landmarks this needs
are in `--jsonl` only, hence passing it explicitly above.

Then fit the excursion:

    python scripts/measure/measure_tip_excursion.py data/excursion-middle.jsonl --finger middle
    python scripts/measure/measure_tip_excursion.py data/excursion-middle.jsonl           # all 5, one file

Why this isn't just "distance between the two end frames": MediaPipe's `world` landmarks are
hand-centred, so their origin drifts a little as the hand's overall pose shifts during the
motion; differencing two raw `world` points mixes that drift into the fingertip's own travel.
This takes tip-minus-wrist per frame first, which cancels it, then fits the dominant axis through
the *whole* trajectory (hand_tracking.excursion.fit_excursion) rather than trusting that the
recording happened to start and stop exactly at the two extremes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `hand_tracking`

from lib import bench_calibration_io  # noqa: E402
from hand_tracking.excursion import FINGERTIP_BY_NAME, fit_excursion, tip_minus_wrist  # noqa: E402

#: Below this, the fitted axis is not trustworthy enough to report as a clean excursion.
MIN_VARIANCE_EXPLAINED = 0.8

#: Below this, there was no real motion to measure - a stationary finger's noise floor has
#: variance_explained == 1.0 too (the degenerate zero-variance case), so it needs its own check.
MIN_EXCURSION_M = 0.002


def load_vectors(path: Path, finger: str) -> list[tuple[float, float, float]]:
    vectors = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            points = [(p["x"], p["y"], p["z"]) for p in frame["world"]]
            vectors.append(tip_minus_wrist(points, finger))
    if not vectors:
        raise SystemExit(f"error: no frames with world landmarks in {path}")
    return vectors


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl", help="a HandFrame JSONL file from `hand_tracking run --jsonl`")
    p.add_argument("--finger", choices=sorted(FINGERTIP_BY_NAME),
                   help="only this finger (default: fit all 5 from the same recording)")
    p.add_argument("--check-mm", type=float,
                   help="a ruler measurement of the same excursion in mm, to sanity-check against")
    p.add_argument("--out", default=bench_calibration_io.DEFAULT_PATH,
                   help=f"merge excursion_m into this JSON (default %(default)s)")
    p.add_argument("--no-write", action="store_true", help="print only, don't touch --out")
    args = p.parse_args()

    path = Path(args.jsonl)
    fingers = [args.finger] if args.finger else sorted(FINGERTIP_BY_NAME)

    results: dict[str, float] = {}
    for finger in fingers:
        vectors = load_vectors(path, finger)
        fit = fit_excursion(vectors)
        no_motion = fit.excursion_m < MIN_EXCURSION_M
        clean = fit.variance_explained >= MIN_VARIANCE_EXPLAINED
        flag = "" if not no_motion and clean else "  <-- no real motion" if no_motion else "  <-- not a clean pull"
        print(f"{finger:8s}  excursion {fit.excursion_m * 1000:6.1f} mm  "
              f"({fit.n_samples} frames, variance explained {fit.variance_explained:.2f}, "
              f"off-axis RMS {fit.perpendicular_rms_m * 1000:.2f} mm){flag}")
        if no_motion:
            if args.finger:
                print(f"  warning: {finger} barely moved in this recording; not writing it")
        elif clean:
            results[finger] = fit.excursion_m
        elif args.finger:
            print(f"  warning: {finger} motion does not look like a clean 1-D pull "
                  "(off-axis wobble is a large fraction of the total movement); not writing it")

    if args.check_mm is not None:
        if args.finger is None:
            print("note: --check-mm compares against a single finger, pass --finger too")
        elif args.finger in results:
            measured_mm = results[args.finger] * 1000.0
            diff_pct = (measured_mm - args.check_mm) / args.check_mm * 100.0
            print(f"\nruler check: tracker {measured_mm:.1f} mm vs ruler {args.check_mm:.1f} mm "
                  f"({diff_pct:+.1f}%)")
            if abs(diff_pct) > 15.0:
                print("  more than 15% off: MediaPipe's world-landmark scale is approximate, "
                      "not calibrated. The rom_target this feeds inherits this bias.")

    if not results:
        print("\nnothing clean enough to write.")
        return 1

    if args.no_write:
        print(f"\n--no-write: not touching {args.out}")
        return 0

    data = bench_calibration_io.save({"excursion_m": results}, args.out)
    print(f"\nwrote {sorted(results)} to {args.out}")
    print(f"current excursion_m: {data['excursion_m']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
