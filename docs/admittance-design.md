# The controller, with the camera as the only force sensor

Design notes for roadmap item 3, written after the force-calibration result came back negative
([force-calibration-result.md](./force-calibration-result.md)). The clinical behaviour is
unchanged from [admittance-control.md](./admittance-control.md); what changes is that there is
exactly one force signal, it arrives at ~30 Hz, and it can disappear for a frame at a time.

## The loop

Everything the controller needs already exists and is tested; this is the wiring between them.

```
camera frame
  -> gauge_reader.StickerPipeline.process(frame, ts)   # existing, merged in #5
       -> StickerFrameResult: .newtons | None, .stable, .found, .reason
  -> safety gate                                       # new, this item
  -> admittance law                                    # new, this item
  -> lib.station.set_position(bus, motor, ticks)        # existing, hardware-verified
```

`StickerPipeline` already does the hard part: red-blob detection, the sticker-relative fraction
that survives the scale moving, the Kalman filter, and a `stable` flag. It reports `newtons =
None` on any frame where it did not actually see the marker, and **deliberately does not
extrapolate** through a dropout, because the filter's guess during a pull can be far off. That
is exactly the semantics a safety gate wants: a missing reading is reported as missing, not as
a plausible-looking number.

## Safety gate: what to do with each frame

One decision table, evaluated before the control law ever runs. Everything that is not a
confident reading resolves to RELEASE.

| Frame | Meaning | Action |
|---|---|---|
| `newtons is None` | marker not seen this frame | tolerate up to `MAX_DROPOUTS` consecutive, then RELEASE |
| `newtons > F_max` | over the hard ceiling | RELEASE immediately, no grace |
| stale (`now - ts > MAX_AGE`) | camera stalled or pipeline blocked | RELEASE |
| `newtons <= F_threshold` | under the cap | HOLD (the arm is the anchor) |
| `newtons > F_threshold` | over the cap | YIELD, per the law below |

RELEASE means: command a position that unloads the spring, then stop commanding. Not "hold
current position" - a stuck arm holding tension against a finger with no way to see that force
is the single worst state this machine can be in.

`MAX_DROPOUTS` exists because one lost frame at 30 Hz is 33 ms and is routine (a hand passes,
a glare flickers); releasing on every single one would make the device unusable. Two or three
consecutive is already 100 ms of blindness while under load, which is not. Start at 2 and tune
against real footage, never upward without a reason.

## The control law

Unchanged in form from the original design, with a deadband:

```python
e = F - F_threshold
if e < DEADBAND:
    v = 0.0                               # anchor
else:
    v = min(e / B, V_MAX)                  # yield, proportional to the excess
```

`v` is a speed along the tether axis. Converting it to the next position command, one frame at
a time:

```python
step_ticks = v * dt * TICKS_PER_METRE      # dt is the real frame interval, not a nominal 1/30
goal = current_position + sign * step_ticks
```

Three properties this has to keep, all of them more important now than they were when a
100 Hz current loop was the plan:

- **`dt` from timestamps, never a constant.** The camera's interval is not exactly 33 ms and
  stutters under load. A hardcoded `1/30` silently turns into a speed error whenever a frame
  is late.
- **Per-step clamp.** `step_ticks` capped so a single frame can never command a large jump,
  independent of what `v` says. This is the backstop for an arithmetic mistake upstream.
- **Absolute travel limit.** Total displacement from the session's start position is bounded,
  checked every frame. Hitting it is a RELEASE, not a clamp.

## What 30 Hz actually costs

The honest version, for the writeup and for anyone reading this later:

- **Human finger movement is slow** (sub-Hz for a deliberate pull), so 30 Hz is not inherently
  too slow to follow it. This was never the problem.
- **Reaction latency is the problem.** At 30 Hz, worst-case time between the force crossing
  the cap and the first corrective command is one frame interval plus processing, ~40-50 ms.
  During that window the force keeps rising at whatever rate the patient is pulling.
- **The mitigation is `V_MAX` and `F_max`, not loop speed.** A low `V_MAX` means the arm
  cannot make things worse quickly; `F_max` well above `F_threshold` but well below anything
  injurious gives the gate room to fire before the spring reaches a harmful tension.
- **The spring is the real fast element.** It is compliant, so force builds gradually with
  displacement rather than instantly on contact - this is what makes a 30 Hz loop tolerable at
  all, and it is worth stating plainly rather than treating the loop rate as if it were fine
  on its own.

## Picking B: it is not a free parameter

Simulating the closed loop (tether at 56.2 N/m, 30 Hz, patient pulling at a constant speed)
gives a relationship worth knowing before touching the bench. 56.2 N/m is the bench-measured
stiffness of the whole tether, spring plus cord plus attachments, not the spring alone: hand-pull
the PESOLA through 50-500 gf and back, measuring the distance between a fixed and a moving
reference mark at each level (`scripts/measure_tether_stiffness.py`, fit by `lib.stiffness`). It
replaces the datasheet shortcut this doc used before measuring (600 gf over ~110 mm of printed
scale = 53.5 N/m), which was always going to be a little too stiff, since the real tether adds a
cord in series with the spring. The raw sweep is `data/tether_stiffness.csv`; the worst up/down
gap in it, 6 mm at 400 gf, is the floor on any displacement later inferred from a force reading
(the `ticks_per_metre` measurement, next).

> **steady-state overshoot above the cap = B x pull speed**

That falls straight out of the law: to give way at speed `v`, the law needs an overshoot of
`B*v` to produce it. So B is bounded by how much overshoot the gap between `F_threshold` and
`F_max` allows:

| B | 4 mm/s (slow) | 20 mm/s (brisk) | 50 mm/s (flinch) |
|---|---|---|---|
| 10 | 0.04 N | 0.20 N | 0.50 N |
| 50 | 0.20 N | 1.00 N | 2.50 N |
| 100 | 0.40 N | **2.00 N** | **5.00 N** |

With a 1.96 N cap and a 3.0 N ceiling there is only **1.04 N** of room. B=100, the obvious
"start over-damped" choice, needs 2 N of overshoot just to keep up with a brisk pull: the arm
never catches up, force climbs, and the gate fires. Simulated, that is exactly what happens.

**B in the 10-20 range is the right starting point for this tether**, not 100. Confirm on the
bench, but start there rather than discovering it on a finger.

## The 30 Hz loop cannot absorb a flinch, and no B fixes that

Also from the simulation: at a 50 mm/s yank, the ceiling is reached in ~1.5 s **at every value
of B tried (2 to 20)**. The reason is arithmetic, not tuning. One frame at 30 Hz is 33 ms; at
50 mm/s the finger travels 1.7 mm in that time, which against a 56.2 N/m tether is ~0.10 N
arriving before the controller can see it at all. Sustain that and the gate fires, correctly.

So: **the fast-yank case is handled by the gate releasing, not by the controller tracking it.**
That is the designed behaviour and it should be presented that way rather than as a tuning
failure. It also means the mechanical breakaway and the dead-man switch are not belt-and-braces
extras; they are what covers the regime the control loop structurally cannot.

## What to tune, and against what

None of these can be finalised from first principles; they come from the bench rig (spring +
a weight, no finger) and then from footage of a healthy volunteer:

| Constant | Starts as | Decided by |
|---|---|---|
| `F_threshold` | the clinical cap, 1.96 N (200 gf) | the clinician, per patient |
| `F_max` | ~1.5x threshold | bench: the tension where the gate must have already fired |
| `DEADBAND` | a few times the gauge reader's noise | the `std_g` the reader reports at rest |
| `B` | **10-20**, from the table above, not 100 | bench: lowest overshoot with no oscillation |
| `V_MAX` | >= the fastest pull to be tracked | not the binding constraint; B is |
| `MAX_TRAVEL` | the rig's usable stroke | bench: how far the arm can go before fouling anything |
| `MAX_DROPOUTS` | 2 | real footage: the dropout-length distribution |
| `MAX_AGE` | ~3 frame intervals | measured frame-interval jitter |

One more thing the simulation surfaced: during a sustained slow pull the arm walks a long way
(>120 mm following a 4 mm/s pull for 30 s). `MAX_TRAVEL` is what ends that, and it fires as a
RELEASE. For a real session that is probably the right outcome, but it means the travel limit
is a routine end-of-exercise condition, not only an error path, and the clinician-facing side
should present it that way.

Tune `B` and `V_MAX` on the bench before any finger is involved, and treat the first volunteer
session as a test of the safety gate, not of the control quality.

## Deliberately not in scope yet

- **Prediction or extrapolation through dropouts.** Tempting, since a short gap could be
  bridged. Not now: a predicted force is indistinguishable from a measured one once it is a
  float in a variable, and the failure mode is pulling harder on a finger based on a guess.
  Revisit only with recorded session data showing what the force actually does during real
  dropouts, and only as an input to the *gate* (how long to tolerate), never as a substitute
  force reading.
- **Smoothing beyond what `ReadingFilter` already does.** It is already a Kalman filter with
  an outlier gate; adding a second filter on top mostly adds lag.
- **Servo current as a cross-check.** There is no signal there on this arm; see the result doc.
