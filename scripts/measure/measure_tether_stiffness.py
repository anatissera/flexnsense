"""Measure the tether's stiffness with the PESOLA and a ruler. No arm, no camera, no weights.

`k` is the number that moves the most downstream: it sets the overshoot the admittance law needs
(docs/admittance-design.md, "Picking B"), and it is what turns a force change measured by the
camera into the displacement that caused it, which is how `ticks_per_metre` gets measured without
putting a ruler on a moving arm.

The one-minute version of this measurement is `k = 600 gf / L`, where L is the distance between
the printed 0 g and 600 g marks: that is the *spring's* stiffness off the datasheet geometry. This
script measures the **tether's** stiffness instead, spring plus cord plus hooks plus attachments,
which is what the finger actually pulls against and is always softer than the spring alone.

Rig:

1. Anchor one PESOLA hook to something that will not move.
2. Pick two reference marks to measure between: one on the anchor, one on the hook that moves.
   Any two, as long as they stay put for the whole sweep; the fit absorbs the offset.
3. Pull the free hook by hand until the needle sits on the target, hold it there, and read the
   distance between the marks with a ruler or calliper.

Run it as a sweep up and then back down through the same force levels: the gap between the two is
the spring's own hysteresis, and it bounds how precisely any displacement inferred from a force
reading can ever be known.

    python scripts/measure/measure_tether_stiffness.py                  # prompt through both sweeps, log, fit
    python scripts/measure/measure_tether_stiffness.py --grams 50,100,200,300,400
    python scripts/measure/measure_tether_stiffness.py --fit config/measurements/tether_stiffness.csv   # re-fit a log
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `lib`

from lib import bench_calibration_io  # noqa: E402
from lib.stiffness import Point, compare_directions, fit_stiffness  # noqa: E402
from lib.units import newtons_to_grams_force  # noqa: E402

DEFAULT_GRAMS = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0]
DEFAULT_OUT = "config/measurements/tether_stiffness.csv"
FIELDS = ["direction", "grams", "mm", "timestamp"]


def prompt_sweep(direction: str, grams: list[float]) -> list[Point]:
    """Ask for one ruler reading per force level. Blank skips a level, 'q' ends the sweep."""
    print(f"\n--- sweep {direction} ---")
    points: list[Point] = []
    for g in grams:
        while True:
            raw = input(f"  pull to {g:g} gf, distance between the marks in mm "
                        "(blank = skip, q = end sweep): ").strip()
            if raw.lower() == "q":
                return points
            if not raw:
                break
            try:
                points.append((g, float(raw.replace(",", "."))))
            except ValueError:
                print("    not a number, try again")
                continue
            break
    return points


def write_csv(path: Path, rows: list[tuple[str, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for direction, grams, mm in rows:
            w.writerow({"direction": direction, "grams": grams, "mm": mm, "timestamp": now})


def read_csv(path: Path) -> dict[str, list[Point]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    by_direction: dict[str, list[Point]] = {}
    for r in rows:
        by_direction.setdefault(r["direction"], []).append((float(r["grams"]), float(r["mm"])))
    return by_direction


def report(by_direction: dict[str, list[Point]], calibration_out: str | None = None) -> int:
    """Print the fit per sweep, the fit over everything, and the hysteresis between sweeps."""
    all_points = [p for points in by_direction.values() for p in points]
    if len(all_points) < 2:
        print("error: fewer than 2 readings in total, nothing to fit")
        return 1

    for direction, points in sorted(by_direction.items()):
        if len(points) < 2:
            print(f"\n{direction}: {len(points)} reading(s), not enough to fit")
            continue
        fit = fit_stiffness(points)
        print(f"\n{direction}: k = {fit.k_n_per_m:.1f} N/m  ({fit.k_gf_per_mm * 10:.1f} gf per "
              f"10 mm), zero-force reading {fit.x0_mm:.1f} mm")
        print(f"      R^2 = {fit.r2:.4f}, worst residual {fit.residual_max_mm:.2f} mm, "
              f"{fit.n_points} points")

    combined = fit_stiffness(all_points)
    print(f"\nboth sweeps: k = {combined.k_n_per_m:.1f} N/m, R^2 = {combined.r2:.4f}, "
          f"worst residual {combined.residual_max_mm:.2f} mm")
    print(f"  -> use k = {combined.k_n_per_m:.1f} N/m downstream; "
          f"1.96 N (200 gf) stretches it {1.96 / combined.k_n_per_m * 1000:.0f} mm")

    up, down = by_direction.get("up", []), by_direction.get("down", [])
    pairs = compare_directions(up, down)
    worst_gap_m = None
    if pairs:
        print("\nhysteresis (same force, both directions):")
        for p in pairs:
            gap_gf = newtons_to_grams_force(abs(p.gap_mm) / 1000.0 * combined.k_n_per_m)
            print(f"  {p.grams:6.0f} gf: up {p.up_mm:7.2f} mm, down {p.down_mm:7.2f} mm, "
                  f"gap {p.gap_mm:+6.2f} mm (~{gap_gf:.0f} gf)")
        worst = max(pairs, key=lambda p: abs(p.gap_mm))
        worst_gap_m = abs(worst.gap_mm) / 1000.0
        print(f"  worst gap {worst.gap_mm:+.2f} mm at {worst.grams:g} gf: a displacement inferred "
              "from a force reading is only good to about this.")
    else:
        print("\nno force level was measured in both directions, so no hysteresis figure.")

    if combined.residual_max_mm > 2.0:
        print("\nwarning: a residual over 2 mm is more than ruler error. Check whether the cord "
              "was still taking up slack at the low end, and consider refitting without those "
              "points: the controller only needs k over the range it will actually work in.")

    if calibration_out:
        updates = {"tether_n_per_m": combined.k_n_per_m}
        if worst_gap_m is not None:
            updates["displacement_uncertainty_m"] = worst_gap_m
        bench_calibration_io.save(updates, calibration_out)
        print(f"\nwrote tether_n_per_m" +
              (" and displacement_uncertainty_m" if worst_gap_m is not None else "") +
              f" to {calibration_out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit", metavar="CSV", help="fit an existing log and exit (no measuring)")
    p.add_argument("--out", default=DEFAULT_OUT, help=f"where to write the sweep (default {DEFAULT_OUT})")
    p.add_argument("--grams", default=",".join(f"{g:g}" for g in DEFAULT_GRAMS),
                   help="comma-separated force levels in gf (default: %(default)s)")
    p.add_argument("--one-way", action="store_true",
                   help="only the up sweep; skips the hysteresis figure")
    p.add_argument("--calibration-out", default=bench_calibration_io.DEFAULT_PATH,
                   help=f"merge tether_n_per_m (and the hysteresis figure, as "
                        f"displacement_uncertainty_m) into this file (default %(default)s)")
    p.add_argument("--no-write-calibration", action="store_true",
                   help="print only, don't touch --calibration-out")
    args = p.parse_args()
    calibration_out = None if args.no_write_calibration else args.calibration_out

    if args.fit:
        return report(read_csv(Path(args.fit)), calibration_out)

    grams = [float(g) for g in args.grams.split(",")]
    print(__doc__.split("Rig:")[1].split("Run it as")[0].strip())
    print(f"\nForce levels: {', '.join(f'{g:g}' for g in grams)} gf")

    rows: list[tuple[str, float, float]] = []
    by_direction: dict[str, list[Point]] = {}
    directions = ["up"] if args.one_way else ["up", "down"]
    for direction in directions:
        levels = grams if direction == "up" else list(reversed(grams))
        points = prompt_sweep(direction, levels)
        by_direction[direction] = points
        rows += [(direction, g, mm) for g, mm in points]

    if not rows:
        print("\nnothing recorded.")
        return 1
    out = Path(args.out)
    write_csv(out, rows)
    print(f"\nwrote {len(rows)} readings to {out}")
    return report(by_direction, calibration_out)


if __name__ == "__main__":
    sys.exit(main())
