# Bench calibration: the numbers the session layer cannot invent

Machine-readable form in `bench.json`, loaded by `session.calibration.BenchCalibration.load`.
A `null` means **not measured yet**, never zero and never a default: asking for one raises
`NotMeasured` carrying the procedure, so nothing runs on a placeholder.

## Measured

### Tether stiffness: 56.2 N/m (5.73 gf/mm)

Measured 2026-11-15 by hand-pulling the PESOLA against a fixed anchor through 50 to 500 gf and
back, reading the distance between a fixed and a moving reference mark at each level
(`scripts/measure/measure_tether_stiffness.py`, fit by `lib.stiffness.fit_stiffness`). Raw sweep in
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

### Fingertip excursion: 98 to 114 mm

Measured 2026-11-15 from hand-tracking recordings, one per finger
(`scripts/measure/measure_tip_excursion.py`).

| Finger | Excursion |
|---|---|
| index | 114.1 mm |
| middle | 111.1 mm |
| ring | 99.9 mm |
| pinky | 98.3 mm |

Thumb not measured.

**Caveat on what this number is.** It was taken with no tether rigged, so it is the travel along
the dominant axis of the finger's own motion, not the projection onto a particular pull
direction. Any real tether axis will be at some angle to that, so these are an **upper bound**
on the projected excursion. Re-measure once the tether geometry is fixed if the margin matters.

### What those two numbers together mean for the travel limit

This is the consequence worth reading before anyone runs a session. With the measured tether and
a 200 gf cap resting at a quarter of it:

- reaching the cap from slack takes **34.9 mm** of stretch;
- at rest the tether is already stretched 8.7 mm, so the `RAMP` phase costs **26.2 mm** of the
  patient's range before any of it is the exercise;
- the rest is exercise, and at constant force the arm displaces one for one with the finger.

| Finger | Excursion | Ramp | Exercise, so arm travel | Against the 50 mm default |
|---|---|---|---|---|
| index | 114.1 mm | 26.2 mm | 88.0 mm | **1.8x over** |
| middle | 111.1 mm | 26.2 mm | 84.9 mm | **1.7x over** |
| ring | 99.9 mm | 26.2 mm | 73.8 mm | **1.5x over** |
| pinky | 98.3 mm | 26.2 mm | 72.2 mm | **1.4x over** |

So `SafetyLimits.max_travel_m`, which defaults to 0.05 m, is between a third and a half of what
a single full-range repetition needs. It would fire partway through the first rep of every
block. docs/session-orchestration.md already called for splitting that limit into a per-rep
allowance and an absolute bound; these are the numbers to size the per-rep one from, and the
answer is roughly 90 mm plus a margin for the worst finger, not 50 mm.

Note this does not mean the exercise is impossible, only that the limit is wrong. The tether's
own travel is not the constraint during constant-force following: the stretch stays constant at
34.9 mm while the arm follows, so only the ramp consumes the PESOLA's ~110 mm of scale.

## Not measured yet

| Field | Blocked on | What it gates |
|---|---|---|
| `ticks_per_metre` | The arm being powered | Any arm motion at all: `control.loop.AxisCalibration` |
| `yield_sign` | Same, falls out of the same measurement | Which way the arm moves to reduce tension |

Procedure for both: [docs/bench-measurements.md](../../docs/bench-measurements.md), measurement 3.
Note that it needs `--motor <id>`, which nothing in the repo records; that guide explains how to
identify it safely with the tether slack.

## Pose dependence

`tether_n_per_m` and the hysteresis are properties of the tether and transfer anywhere.
`ticks_per_metre` and `yield_sign` will **not**: they depend on the arm's pose, like the force
calibration did (`docs/force-calibration.md`). When they are measured, replace
`measured_in_pose` with a description of the pose they were taken in, and re-measure if the
session pose changes.
