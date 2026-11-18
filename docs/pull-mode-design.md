# Admittance controller: two modes, two risk profiles

> **Vocabulary note, added when this was brought onto `main`.** This document predates
> [docs/session-orchestration.md](./session-orchestration.md) and calls the two exercises
> **YIELD** and **PULL**. The session layer now calls them **`active`** and **`passive`**, after
> the range-of-motion vocabulary a hand therapist uses, and reserves `YIELD` and `PULL` for the
> two control *laws*, since a passive repetition uses both. Read YIELD here as `active` and PULL
> as `passive`; nothing else about the content changes.
>
> It is on `main` because it holds the design for the device-driven half, which the implemented
> code currently handles more weakly than this doc specifies. Three places where this is ahead of
> `session/` and `control/`, all unimplemented:
>
> 1. **The inverted monotonicity guard.** `session/modes.py` says the guard in
>    `control.loop.ControlLoop._check_yield_direction` simply does not apply while the arm pulls,
>    and that the range-of-motion target plus the cap replace it. This doc has the better answer:
>    invert it. While pulling, the force *must rise*, and a commanded motion that does not raise
>    it means slack, a detached hook, or tissue that gave way. That is a real safety property the
>    code does not yet have.
> 2. **Three stop conditions, not one.** `session.protocol.Block.rom_target_m` stops a pull at a
>    distance. This doc stops at the first of force, angle, **or the local slope `dF/dtheta`**,
>    which is what a clinician's "end feel" captures by touch. Force alone lets tissue that gives
>    abruptly be driven to an injurious position at constant load; distance alone exposes a stiff
>    finger to a large force first.
> 3. **Unprompted rising force during a hold.** `Phase.HOLD` currently runs `Law.STILL` and checks
>    nothing. This doc releases tension if the force rises with no commanded motion, because that
>    is the patient contracting against the stretch.
>
> Reconciling the three into `control/` and `session/` is the next piece of work on the
> device-driven path, and it should be done before any passive-mode session runs on a finger.


The physical model is one spring between two points. `P` is where the PESOLA's hook grips the finger - the patient moves it. `G` is where arm B's gripper holds the other end - the robot moves it. The scale reads the tension along the line between them, and nothing else:

```
T = k * max(0, d - L0)              d = |G - P|
u_hat = (P - G) / |P - G|            direction the robot yields along
```

Two consequences follow directly. The controller never needs to know where anything is in space - it only needs `T`, so if the hand shifts a little, `d` changes, `T` changes, and the force loop absorbs it without remapping anything. But the PESOLA gives a scalar: `|T|`, never `u_hat`. Direction has to come from somewhere else, and it is the central vulnerability of the design (see "The direction problem" below).

**Units** (see contributing.md "Units"): every `F`, `T`, `F_threshold`, `F_target`, `F_max`, `F_home` in this doc and in the controller is in **newtons**. Grams only ever appear where a human reads the PESOLA directly, e.g. setting up a known weight or reading the clinician's displayed cap; anything the controller computes with is already past the `lib/units.py` boundary.

## Mode is chosen by the operator, not inferred

Before torque is enabled on arm B, the clinician/operator selects a mode and its parameters. This never auto-detects, for two reasons: the two modes have opposite risk profiles (see below), so a misclassification would silently put the system in the one that doesn't self-limit; and knowing whether a flexed finger is a contracture needing a stretch, or just how the patient's hand happens to rest, is clinical information the camera does not have.

```
operator selects:
  mode: YIELD | PULL
  parameters for that mode
  -> confirm before torque is enabled
```

The servo torque cap should also depend on mode (lower in PULL than in YIELD), written to the register only once the mode is confirmed - so a bug that boots with default config boots into the more conservative mode, not the more dangerous one.

## Mode YIELD - tendon injury

Finger relatively extended, moving away from the robot (actively, by the patient). Arm B anchors until the force threshold, then yields.

```python
e = F_meas - F_threshold
if e < delta:
    v = 0                          # the arm is the anchor
else:
    v = clip(e / B, 0, v_max)      # yield proportional to the excess
    move(G, +v * u_hat)            # toward P: tension goes DOWN

# monotonicity guard: motion must LOWER the force
if v > 0 and F_meas_has_not_dropped_in(N_ms):
    release_tension(); stop()
```

**The force the patient generates self-limits the failure.** If the control loop is wrong, the patient simply stops pulling and the force disappears. This is the mode described in the README and in the force-calibration doc.

## Mode PULL - contracture

Finger starts flexed/closed. The robot pulls to open it to a target, the finger moves passively. **This inverts who has authority over the force** - it is not the same system with a flipped sign, it is a device of a different risk class. The patient cannot end the interaction by relaxing, because relaxing doesn't reduce a force the patient isn't generating.

### Returning authority to the patient

Three mechanisms, all three, not a choice of one:

1. **Dead-man switch** held in the patient's free hand. The robot pulls only while it is pressed; releasing it stops and releases tension immediately. Cheapest and strongest mitigation - it literally gives back the authority the mode removed.
2. **Mechanical breakaway** at the finger attachment (e.g. a magnetic coupling that releases above a set force). A physical limit no software bug can exceed.
3. **Servo torque register cap**, set lower than in YIELD mode, written before the mode is enabled.

### Two sub-modes, not interchangeable

Hand therapy already separates these, and the distinction is exactly force control vs. position control:

| | Hold force (creep) | Hold position (stress relaxation) |
|---|---|---|
| Clinical equivalent | Dynamic splint | Static progressive splint |
| What's held constant | `F = F_target` | `theta = theta_target` |
| What drifts | Position advances as tissue lengthens | Force decays as tissue relaxes |
| Typical protocol | 6-8 continuous h/day | 3 x 30 min/day |
| Failure direction | Robot **keeps advancing** on the finger | Force can only **go down** |

**Use hold-position.** Its failure mode is benign by construction - force only decays. Hold-force's failure mode is the robot continuing to advance to maintain load, which is exactly the behaviour a bug must not have. The 3x30min protocol also matches a supervised session; the 6-8h protocol corresponds to a splint, not a robot.

```python
# PULL state machine
if any(F_meas >= F_target, theta >= theta_max, dF_dtheta > stiffness_max,
       dead_man_released, t > t_max):
    goto(HOLD)                         # first one reached wins
else:
    v = clip((F_target - F_meas) / B, 0, v_max_pull)
    move(G, -v * u_hat)                # away from P: tension goes UP

# inverted monotonicity guard: motion must RAISE the force
if v > 0 and F_meas_has_not_risen_in(N_ms):
    stop()                             # slack, detached, or tissue gave way

# during HOLD (position-held): rising force with no commanded motion
if mode == HOLD and F_meas_rising_unprompted:
    release_tension()                  # patient is contracting against the stretch
```

PULL needs a state YIELD doesn't: a slow, low-force **approach phase** that takes up slack before loading, to avoid arriving at the finger with a jerk. Detecting that first soft contact needs forces in the 10-30 g (0.1-0.3 N) range - below the servo current dead zone - so this phase (and the equivalent "return home" in YIELD, see below) depends on the camera, not current.

### Where the stop threshold comes from

Neither ceiling alone is enough. Force-only lets a finger whose tissue gives abruptly get driven to an injurious position at constant load. Angle-only exposes a stiff finger to a large force before reaching the target. A third signal is better than either: the local slope `dF/dtheta` is what a clinician's "end feel" already captures by touch - soft end-feel (edema, early collagen cross-linking) gives under stretch, hard end-feel (mature collagen) barely moves. That judgment is literally the slope of the force-angle curve, which the robot measures in numbers and reproduces between sessions, unlike palpation. If the slope spikes, that's the hard end-feel - stop regardless of whether `F_max` was reached.

Per-session procedure to set numbers (a protocol to run, not fixed constants):

1. Measure today's baseline: bring the finger to a low reference force (~50 g, ~0.49 N), record the angle reached → `theta_ref`, which changes day to day.
2. Sweep slowly, recording F vs. theta, until the first ceiling. That gives today's stiffness/end-feel.
3. Set the target as an increment, not an absolute: `theta_target = theta_ref + delta`, small delta. This is the static-progressive-splint principle - stepwise angle increases, not force increases.
4. Hold position for the prescribed time; force decays on its own.
5. Progress `delta` between sessions based on how `theta_ref` evolves, not on what the patient tolerated that day.

Defining the target as an increment over today's baseline makes the system self-adjusting: a stiffer day lowers `theta_ref` and the target follows it down. A fixed absolute target set last week does not have that property, and that is exactly how overshoot happens.

**Honesty for the pitch**: contracture resolution is governed by total end range time (TERT) - published recommendations are 6-12 h/day, one study reports a mean of 20.6 h/day. A 30-minute robot session is a small fraction of that. The robot does not replace the splint; it adds quantified measurement, controlled progression between sessions, and remote supervision.

**Do not test PULL on a human finger** until the force limit is repeatable against a spring and a weight, the dead-man switch actually cuts power, and the mechanical breakaway is in place. For the hackathon demo, a foam finger or elastic band stands in for a stiff finger and still produces the stiffness curve, which is the interesting deliverable.

## The direction problem

The scale gives magnitude, never direction. As long as the assumed `u_hat` has positive projection onto the true direction, yielding along it still lowers the force even if it isn't optimal; past 90 degrees the sign flips and every correction makes it worse - positive feedback.

**The single guard that catches most of this**: if commanded motion is supposed to reduce (YIELD) or raise (PULL) the force and it measurably does not within `N` ms, release and stop. One condition catches wrong direction, a hand that moved, a detached hook, a flipped encoder sign, and a miscalibrated kinematic model at once.

This is also what arm A is actually for: not "holding the hand" as a gesture, but **keeping the direction assumption valid**. It's a safety component wearing a support costume - which also means it doesn't need to be a robot gripping anything; a rigid cradle with a soft strap does the same job with less risk, since a gripper squeezing a freshly operated hand is risk without a corresponding benefit.

### Two gaps the monotonicity guard alone doesn't close, and how we close them

Three cases from "does it still work if the hand moves a little":

| Case | Covered by the guard above? |
|---|---|
| Hand shifts a few mm | Yes - the force loop absorbs it, see the model above |
| Hand moves away from the robot (YIELD's normal case) | Yes - that's what the mode is for |
| **Hand moves toward the robot (slack)** | Partially - safe, but the exercise goes inert and nobody notices |
| **Hand rotates or shifts laterally a lot** | Not covered - `u_hat` actually changes |
| **Arm B travels far while yielding** | Partially - `u_hat` drifts during its own long excursion |

**Slack.** Not dangerous (T -> 0, the robot sits still) but silent. Add an explicit `SLACK` state: if `T` stays below ~20 g (~0.2 N) past a timeout, flag it in telemetry and run a gentle **take-up**: creep at a velocity well under `v_max` toward a small baseline tension (20-30 g, 0.2-0.3 N), never further. A patient still retreating can always outpace the take-up, so it never turns into chasing - it only keeps the exercise live when the patient has simply stopped engaging it.

**Rotation / large lateral shift, and `u_hat` drift during a long yield.** One added component fixes both: track `P` by vision every cycle instead of assuming it once at setup, and recompute `u_hat = (P - G) / |P - G|` each cycle with `G` from forward kinematics (always available from the encoders). That's the same projection used to discriminate "released" from "detached" below, applied continuously instead of once.

When vision of `P` isn't available or its confidence is low, two extra layers:

- **Consistency check, not just sign.** Compare the force-model's predicted `dF/dt` under the commanded `v * u_hat` against the observed `dF/dt`. A large mismatch - even with the correct sign - flags the 45-90 degree grey zone before it degrades into an outright sign flip.
- **Yield budget.** Cap cumulative travel of arm B before forcing a re-anchor pause. This bounds `u_hat` drift, and it's needed anyway: the current-to-force calibration (`docs/force-calibration.md`) is itself pose-dependent, so a large excursion invalidates it too.

## Distinguishing "the patient let go" from "the hook detached" (YIELD)

Both read as force dropping to zero and call for opposite responses:

| Signal | Patient released | Hook detached |
|---|---|---|
| Shape of the drop | Gradual, hundreds of ms | Step, tens of ms |
| Final force | May retain some residual tension | Exactly slack |
| Finger (tracking) | Returns toward neutral | Stays where it was |
| Hook (camera) | Still on the finger | Not there |
| Correct response | `RETURN` | `STOP` + alert |

`lib/filters.ReadingFilter` already computes `velocity_g_s` (the rate of force change) alongside the smoothed reading - that derivative is exactly the "shape of the drop" signal above, combined with the finger's own tracked position from the hand-tracking roadmap item.

**"Home" should be defined by force, not by an absolute pose.** Home = "the position where tension equals `F_home`," with `F_home` small (20-30 g, 0.2-0.3 N), not a fixed joint configuration. If the hand shifted during the session, an absolute-pose home either pushes into the finger or leaves the spring slack; a tension-defined home self-corrects. Since that range sits below the servo's dead zone, returning also depends on the camera, same as the PULL approach phase above.

## Frequencies: why there are two loops, not one

| Signal / requirement | Frequency | Design consequence |
|---|---|---|
| Voluntary human movement | 0.3-3.5 Hz | What has to be tracked - slow |
| Minimum bandwidth for force control | >=12 Hz | Floor from pHRI literature |
| Recommended sampling | 10x bandwidth | >=120 Hz for decent phase margin |
| Gold standard in physical interaction | 1 kHz | Out of reach over a serial bus to Feetech servos |
| Camera reading the PESOLA | ~30 Hz | **Too slow** for the inner loop |
| Servo current over the serial bus | ~100 Hz | Enough for the inner loop |

Hence two speeds: a fast inner loop on servo current for reacting, and a slow outer loop on the camera that corrects the estimator's bias and produces the gram number shown to the clinician. If the camera drove the loop directly, the system would run at 30 Hz against a >=120 Hz requirement, with a freshly operated finger on the other end.

## What's still not covered

PULL-specific failures that have no YIELD equivalent, because PULL is not "YIELD with the sign flipped":

| Failure | What happens | Mitigation |
|---|---|---|
| Inverted sign in PULL | Robot pulls instead of yielding, looking like nominal behaviour - hard to notice by watching | Mode-specific sign in the monotonicity guard; verify active mode before enabling torque |
| Tissue gives abruptly | At constant force, robot advances suddenly to an injurious position | Angle ceiling in addition to force ceiling; velocity cap |
| Hold-force creep | Robot keeps advancing for minutes to maintain load | Hold position, not force; hard angle ceiling |
| Patient contracts against the stretch | Force rises with no command; pushing through fights an active muscle | Unprompted rise during HOLD -> release |
| Clinical contraindication (recent fracture, infection, acute inflammation) | Stretching is contraindicated and the robot has no way to know | Outside the system - enabled by the clinician, never by default |

See also the general failure table in the README's Safety section, which still applies to both modes (lost serial bus, software hang, power loss, glare, thermal drift).
