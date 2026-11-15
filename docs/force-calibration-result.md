# Force calibration result: servo current cannot sense force on this arm

Run on 2026-11-15 against the real follower bus (`5B8E114979`), both sweep directions, 12
points each from 50 to 600 gf. **Negative result: the ST3215's current reading carries no
usable force signal through a 345:1 reduction.** Force sensing moves entirely to the camera.

This is the outcome [force-calibration.md](./force-calibration.md) was written to detect, and
it is why that experiment was day-one work rather than something assumed and built on.

## The data

Mean current (mA) per motor at each applied force, torque enabled, PESOLA pulling on the
gripper along the therapy axis:

| gf | m1 | m2 | m3 | m4 | m5 | m6 | m7 | m8 |
|---|---|---|---|---|---|---|---|---|
| 50 | 0.0 | 0.0 | 1.0 | 1.1 | 0.3 | 0.0 | 0.0 | 0.0 |
| 200 | 0.0 | 0.0 | 0.9 | 1.1 | 0.3 | 0.0 | 0.0 | 0.0 |
| 400 | 0.0 | 0.0 | 0.0 | 0.4 | 0.3 | 0.3 | 0.0 | 0.0 |
| 600 | 0.0 | 0.2 | 0.0 | 0.0 | 0.5 | 0.9 | 0.0 | 0.0 |

The whole signal lives between 0 and ~1 mA, which is the quantisation floor of the reading.
Resting current with nothing applied reads the same 0-1 mA.

## Why the fit looked good and was not

`--fit` reported R^2 = 0.886 (up) and 0.863 (down), which looks like a real relationship until
you check three things:

- **The slope is negative.** It fits "more force, less current", which is backwards.
- **The best channel changes between sweeps** (motor4 going up, motor3 coming down). A real
  signal would come from the same joint in both.
- **The implied scale is absurd**: 348 gf per mA, against a total observed span of ~1 mA.

A straight line through noise that happens to drift still reports a high R^2. The sanity
checks, not the R^2, are what decide.

## Why it happens

600 gf at the gripper, divided by the 345:1 reduction, is roughly 1.7 gf reflected at the
motor shaft - below what gearbox friction alone absorbs, and far below the current sensor's
resolution. The servo is confirmed `ST-3215-C018` (345:1) from its own label, so this is not a
wrong-variant problem.

## Ruled out before concluding

- **Torque was enabled.** Read back directly from each motor's `RAM_TORQUE_ENABLE` byte
  (0x28): `1` on all 8 motors.
- **The PESOLA was attached** to the gripper, pulling along the therapy axis.
- **The joint ranges were calibrated** first (`scripts/calibrate.py` confirmed all 8 motors
  report a real range), so position readings were meaningful.
- **NormaCore's own example** (`example_follow.py`) shows the same currents, so this is not an
  artefact of our client.

## What this changes in the design

| | Planned | Actual |
|---|---|---|
| Fast inner loop | servo current, ~100 Hz | **does not exist** |
| Control loop rate | ~100 Hz | ~30 Hz, the camera's rate |
| If the camera reading is lost | current covers the gap | **no second source** |

The last row is the one that matters for safety. The admittance design already said a
low-confidence force reading must lead to "release and stop, never guess" - that was a backup
behaviour, and it is now the only behaviour. The gauge reader's confidence output
([scale-reader.md](./scale-reader.md)) stops being a nice-to-have and becomes the thing the
whole safety case rests on.

## Raw data

Committed alongside this doc: `data/calib_up.csv`, `data/calib_down.csv` (13 lines each,
header + 12 points). Re-fit with `python scripts/log_force_calibration.py --fit data/calib_up.csv`.
