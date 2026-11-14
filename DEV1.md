# Task: Characterize post-surgery finger/tendon recovery from finger-curl data

Hackathon project. Build a small, well-tested Python package plus a demo notebook or script
that turns finger-curl recordings into per-finger recovery metrics.

## 1. Inputs

For each finger (thumb, index, middle, ring, little), per curl trial:

- `t`: time stamps, seconds (assume uniform sampling; resample if not)
- `x(t)`: fingertip displacement along the curl path (mm, or normalized by hand length)
- `F(t)`: fingertip force from the strain gauge (N)

Metadata per trial: `patient_id`, `finger`, `hand` (injured/contralateral), `session`
(e.g. week since surgery), `trial_index`.

Reference data (use whichever exists, in this priority order):
1. Contralateral finger of the same patient, same session
2. Same finger, healthy baseline or normative dataset
3. If neither exists, report raw values and `normalized=False`

Define a loader that returns a tidy table (one row per sample) and validates:
- no NaNs after preprocessing, monotonic time, plausible ranges
- consistent units across files (raise on mismatch)

## 2. Preprocessing

- Low-pass filter x(t) and F(t) (zero-phase Butterworth, make the cutoff a parameter, default ~10 Hz for x and ~20 Hz for F).
- Zero the force baseline using the first 0.2 s of rest.
- Segment each recording into repetitions: a curl starts when x rises above 5% of its range and ends when it returns below it. Segment into phases: flexion (loading), hold, extension (unloading).
- Reject trials with fewer than 3 clean reps or with a large baseline drift, and report why.

## 3. Metrics (compute per finger, per session)

Compute each metric per repetition, then report the median and IQR across repetitions.

### 3.1 Motion (from x)
| Metric | Definition |
|---|---|
| `rom` | max(x) − min(x) per rep, in mm and as % of hand length if available |
| `curl_time` | duration of the flexion phase |
| `peak_velocity` | max of dx/dt |
| `smoothness_sparc` | spectral arc length of the speed profile (SPARC); also report normalized jerk |
| `repeatability_cv` | coefficient of variation of `rom` and of `curl_time` across reps |

### 3.2 Force (from F)
| Metric | Definition |
|---|---|
| `peak_force` | max(F) per rep |
| `time_to_peak` | time from force onset to peak |
| `rfd_100`, `rfd_200` | mean force slope over the first 100 and 200 ms after force onset (N/s) |
| `force_steadiness_cv` | CV of F during the hold phase (middle 60% of the hold) |
| `force_decay` | % drop in peak force from first to last rep (endurance/fatigue) |
| `force_relaxation` | % drop in F over a sustained hold, fit F(t) = F0·exp(−t/τ), report τ |

### 3.3 Force–displacement (combined; these are "apparent" mechanical properties)
| Metric | Definition |
|---|---|
| `apparent_stiffness` | slope of a linear fit to F vs x over the 20–80% range of the flexion phase (N/mm) |
| `work_loading` | ∫F dx over the flexion phase |
| `hysteresis` | (loop area between loading and unloading F–x curves) / `work_loading` |
| `force_at_displacement` | F at 25%, 50%, 75% of the reference ROM (strength vs. posture) |

IMPORTANT labeling rule: these combine tendon, muscle, adhesion and soft-tissue effects. In all
outputs name them "apparent" or "effective", never "tendon stiffness" or "tendon stress/strain".
True stress needs cross-sectional area and true strain needs tendon elongation (ultrasound);
neither is available here.

## 4. Normalization and recovery index

1. Normalize each metric to the reference: `norm = metric / reference` (as %).
   For metrics where lower is better (`curl_time`, `hysteresis`, `repeatability_cv`,
   `force_steadiness_cv`, `force_decay`), use `reference / metric`.
2. Cap normalized values to [0, 120%].
3. Group scores:
   - Motion score = mean of `rom`, `curl_time`, `peak_velocity`, `smoothness_sparc`
   - Strength score = mean of `peak_force`, `rfd_200`, `force_at_displacement` (mean of the three points)
   - Mechanics score = mean of `apparent_stiffness`, `hysteresis`, `force_relaxation`
   - Consistency score = mean of `repeatability_cv`, `force_steadiness_cv`, `force_decay`
4. `recovery_index` per finger and session = weighted mean of the group scores.
   Default weights: motion 0.3, strength 0.35, mechanics 0.2, consistency 0.15.
   Make weights configurable and include a sensitivity analysis (vary each weight ±50% and
   report the range of the index and any change in finger ranking).
5. Across sessions per finger, fit a saturating curve `R(t) = R_inf·(1 − exp(−t/τ))` and report
   `R_inf`, `τ` and `time_to_90pct` (interpolate if not directly observed).

## 5. Outputs

- `metrics.csv`: one row per (patient, finger, session, rep) with all raw metrics
- `summary.csv`: one row per (patient, finger, session) with medians, group scores, recovery index
- Figures (matplotlib, saved as PNG):
  1. x(t) and F(t) for one example trial, with phases annotated
  2. F–x loop for the injured vs. reference finger of one session
  3. Radar or bar chart of group scores per finger for the latest session
  4. Recovery index vs. session for all five fingers, with fitted curves
- A short `REPORT.md` auto-generated for each patient: recovery index per finger, best/worst
  finger, time_to_90pct, and any flagged data-quality issues.

## 6. Engineering requirements

- Python 3.10+, numpy, scipy, pandas, matplotlib. No heavy dependencies.
- Package layout: `loader.py`, `preprocess.py`, `metrics.py`, `scoring.py`, `plots.py`, `cli.py`.
- A CLI: `python -m curlchar run --data DIR --reference contralateral --out OUT_DIR`.
- Unit tests (pytest) using synthetic curls: a known sinusoidal x(t) with a known force profile
  and checks that ROM, peak force, RFD, stiffness and hysteresis come out within 2% of the known values.
- A synthetic data generator that simulates a recovering finger: ROM, peak force and speed rise
  over weeks along a saturating curve, plus noise and drift. Use it for the demo and tests.
- Handle missing fingers or sessions gracefully (skip and log, don't crash).

## 7. Caveats to include in the README (verbatim intent)

- Only flexor-driven curl behavior is characterized. Extensors, FDP vs. FDS separately, thumb-specific tendons and intrinsic muscles are NOT measured by curls alone.
- Force-displacement properties are apparent/effective, not true tendon stress, strain or stiffness.
- The recovery index is a heuristic composite. Weights are subjective; report the sensitivity analysis.
- Results are for research and demonstration, not for clinical decision-making.

## 8. Optional stretch goal: MyoSim (MyoSuite) model fit

Only if time allows, and clearly separated from the core pipeline:

- Load the MyoFinger (or MyoHand) model from MyoSim (MuJoCo). In this model tendons are rigid
  (no elasticity), and the muscle and tendon form one actuator unit; there is no explicit tendon strain.
- Replay the measured tip trajectory x(t) as the target; use inverse dynamics or static
  optimization to estimate muscle activations and tendon forces.
- Fit a single **force-capacity factor** (0–1) per finger, scaling the flexor actuators' max
  isometric force so that the simulated fingertip force matches the measured peak force F.
  Fit at the finger or flexor-group level, not per muscle (the problem is underdetermined).
- Report per finger: force-capacity factor, required activation for a given force, and estimated
  tendon excursion (moment arm × joint angle change from the model).
- Label every one of these "model-fitted" in outputs. Verify the muscle names and tendon
  attributes in the model XML before relying on them.

## 9. Definition of done

1. `pip install -e .` works and `pytest` passes.
2. Running the CLI on the synthetic dataset produces `metrics.csv`, `summary.csv`, the four
   figures and the per-patient `REPORT.md`.
3. The README documents every metric with its formula, units and "higher/lower is better".
4. The sensitivity analysis and the caveats section are included.
