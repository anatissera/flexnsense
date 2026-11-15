# Bench calibration: the numbers the session layer cannot invent

Machine-readable form in `bench.json`, loaded by `session.calibration.BenchCalibration.load`.
A `null` means **not measured yet**, never zero and never a default: asking for one raises
`NotMeasured` carrying the procedure, so nothing runs on a placeholder.

## Measured

### Tether stiffness: 56.2 N/m (5.73 gf/mm)

Measured 2026-11-15 by hand-pulling the PESOLA against a fixed anchor through 50 to 500 gf and
back, reading the distance between a fixed and a moving reference mark at each level
(`scripts/measure_tether_stiffness.py`, fit by `lib.stiffness.fit_stiffness`). Raw sweep in
`data/tether_stiffness.csv`.

| Sweep | k | r2 | Worst residual |
|---|---|---|---|
| up | 56.19 N/m | 0.9986 | 1.8 mm |
| down | 56.40 N/m | 0.9960 | 2.3 mm |
| both | 56.17 N/m | 0.9950 | 3.2 mm |

This is the **whole tether**, spring plus cord plus attachments, which is what the controller
actually pulls against. It replaces the datasheet shortcut used before measuring (600 gf over
~110 mm of printed scale = 53.5 N/m). Worth noting the direction of the correction: the real
tether is **stiffer** than the shortcut, so every force-per-millimetre figure derived from the
old number understated the force. Nothing in the design broke, but the error was on the unsafe
side.

The up and down fits agreeing to 0.4 % is what makes this trustworthy, unlike the servo-current
fit that reported r2 = 0.886 while fitting noise (`docs/force-calibration-result.md`). Here the
residuals are at the ruler's own reading precision.

### Displacement uncertainty: 6 mm

The worst up/down gap in the same sweep, at 400 gf (`lib.stiffness.compare_directions`). This
is the tether's own hysteresis and it is the floor on **any** distance inferred from a force
reading.

It is larger than anyone expected, and it settled a design question rather than just being
logged: it ruled out measuring `ticks_per_metre` from the force at all. Inferring a
displacement as `dF / k` inherits the 6 mm as error, which is 40 % over a 15 mm move and still
10 % over 60 mm, and 60 mm is most of the tether's ~110 mm of usable travel. A ruler against
the arm has a fixed ~1 mm reading error, so 3 % over 30 mm. The ruler wins by an order of
magnitude, and `session.calibration.HOW_TO_MEASURE` now says so explicitly, including that an
earlier version of those instructions recommended the force method, so nobody reconstructs it.

## Not measured yet

| Field | Blocked on | What it gates |
|---|---|---|
| `ticks_per_metre` | The arm being powered | Any arm motion at all: `control.loop.AxisCalibration` |
| `yield_sign` | Same, falls out of the same measurement | Which way the arm moves to reduce tension |
| `excursion_m` | A hand in the session pose, per finger | The per-rep travel allowance and `rom_target_m` |

`excursion_m` needs only a camera and a hand, not the arm, so it is not blocked by the same
thing. The script exists (`scripts/measure_tip_excursion.py`, `hand_tracking/excursion.py`);
the measurement has not been taken.

## Pose dependence

`tether_n_per_m` and the hysteresis are properties of the tether and transfer anywhere.
`ticks_per_metre` and `yield_sign` will **not**: they depend on the arm's pose, like the force
calibration did (`docs/force-calibration.md`). When they are measured, replace
`measured_in_pose` with a description of the pose they were taken in, and re-measure if the
session pose changes.
