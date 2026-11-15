# Session orchestration: modes, reps, and who tells whom

The force loop in `control/` knows how to not hurt a finger for one 33 ms cycle. It has no
idea that a session is ten repetitions on the index finger followed by ten on the middle one,
that the patient has to be told what to do before any of it happens, or that a second arm has
to move the hook in between. That missing layer is what this doc designs, because every part
of the project ends up touching it: the controller, the second arm's policy, and the patient
facing dashboard all have to agree on what state the session is in.

## What already exists

Verified by reading the code, not assumed:

| Where | What it does | Why it is not enough |
|---|---|---|
| `control/safety.py`, `control/admittance.py`, `control/loop.py` | Per cycle force loop: HOLD / YIELD / RELEASE plus the admittance law | Covers one of the two modes below. No notion of a repetition, a mode, a finger, or who asked it to start |
| `dashboard/app/components/GuideHand.tsx` | Cues the patient: 3D hand animates a flex, voice says "Flex" and "Release", rep counter and progress bar | Driven by `performance.now()` on hardcoded constants (`REPS = 5`, `HOLD_S = 3`, `RELEASE_S = 3`). Does not know the finger, the force, or the robot, and never waits for the patient |
| `dashboard/app/lib/curlsim/live.ts` (`LiveAnalyzer`) | Detects phases from the measured signal: `baseline -> rest -> flexion -> hold -> extension`, counts `repsDone`, computes per rep metrics including hysteresis | Observes only, commands nothing, and runs in the browser on simulated or tracked data |
| `dashboard/app/lib/curlsim/types.ts` | `Finger` (five of them), `TestKind` (`full` or `fds`), `Trial` carrying finger, hand, session and trial index | The vocabulary exists, but only inside the dashboard, in TypeScript |
| `data_collection/` (branch `data_collection`) | Records teleoperated episodes for arm 2, task stored as a language string | Its own README: "training and running a policy on the arm are not covered here". There is nothing an orchestrator can call |
| `colab_train/` (branch `training`) | Links to the SmolVLA and ACT Colab notebooks | Those notebooks consume a `LeRobotDataset`; the recordings are Station Parquet. Nobody owns that conversion yet |

Two things fall out of that table. First, **there are already two disagreeing models of a
repetition**: a commanded one on a wall clock in `GuideHand`, and an observed one from the
signal in `LiveAnalyzer`. Second, **nothing in the repo coordinates the two arms**, and
nothing connects a cue shown to the patient to a force actually applied by a motor.

## Three layers, deliberately separate

```
  orchestrator        per block   "ten active reps on the index, then move the hook"
      |
  rep machine         per rep     cue the patient, count what they actually did
      |
  force loop          per cycle   HOLD / YIELD / RELEASE, 30 Hz        (control/, exists)
```

They are separate because they run at different rates and carry different risk. The force
loop is the only part that can hurt someone, it is pure and exhaustively tested, and it must
stay small enough that it keeps being so. Nothing from the layers above may ever be able to
talk it out of releasing: the orchestrator can ask for a repetition, it cannot raise a cap
past the session's ceiling or clear a latched RELEASE. Only a human clears that, which is
already how `SafetyGate.reset` is written.

## The two modes, and what to call them

> **There is an earlier design for the device-driven half**, in
> [docs/pull-mode-design.md](./pull-mode-design.md), and on three points it is ahead both of
> what is written here and of what is implemented: it **inverts** the monotonicity guard rather
> than dropping it (while pulling, the force must rise, and a commanded motion that does not
> raise it means slack, a detached hook or tissue that gave way), it stops a pull on the first
> of force, angle **or** the local `dF/dtheta` slope rather than on distance alone, and it
> releases if the force rises unprompted during a hold, which is the patient contracting
> against the stretch. Reconcile the two before any passive-mode session runs on a finger.

The two exercises differ in **who is the source of motion**, and that is the only difference
that matters to the code.

- **Active mode.** The finger starts relatively extended. The patient flexes it, which
  stretches the spring, so the force rises. Once the force reaches the cap the arm gives way
  so the force stays at that value for the rest of the patient's range. The patient does the
  work; the arm is a controlled brake.
- **Passive mode.** The arm pulls the finger toward extension, at a bounded force, stretching
  it. The patient is moved. Then the patient flexes back to return to the start, and the arm
  releases while they do. The arm does the work on the way out; the patient does it on the way
  back.

**Recommended names: `active` and `passive`.** They come from the vocabulary a hand therapist
already uses, active and passive range of motion (AROM and PROM), so the clinician facing
labels need no translation. The alternatives are worse for specific reasons:

- "tension" is a bad name because *both* modes apply tension and both use the same force cap.
  Nothing about the force distinguishes them.
- "accompany" is a bad name because the arm accompanies in both: it follows the finger in one
  and leads it in the other.
- `yield` and `pull` stay as the names of the **laws**, describing what the arm does in a given
  phase, not as the names of the modes. A passive rep uses both laws, which is exactly why the
  mode cannot be named after one of them.

Nothing in the dashboard has to be renamed to adopt this. There is no mode concept there yet:
`"mode"` in that code means a UI mode (explore mode, hand tracking mode), and `TestKind`
(`full` or `fds`) is an **orthogonal axis**, whether the DIP joint is blocked to isolate the
FDS tendon. A block can be active or passive *and* full or fds. Do not collapse them.

### The symmetry that makes this cheap to build

Both modes are the same two beat repetition, with the beats in opposite order:

| | Work phase | Reset phase |
|---|---|---|
| **active** | patient flexes, arm yields at constant force | arm walks back to start, patient relaxed |
| **passive** | arm pulls toward extension at constant force | patient flexes back, arm releases |
| | *active:* patient moves. *passive:* device moves | *active:* device moves. *passive:* patient moves |

So **the driver flips within a rep, and it flips the other way between modes**. That one fact
is what the state machine and the dashboard both key off, and it means there is one rep
skeleton rather than two state machines.

### The dangerous half: `may_move` is backwards for a device-driven phase

`Decision.may_move` currently returns true only for `Action.YIELD`. For a patient-driven phase
that is exactly right: the arm moves only when the force has gone over the cap.

For a **device-driven** phase it is exactly wrong. There the arm has to advance while the
force is **under** the cap and stop when it reaches it. Reusing `may_move` would give an arm
that sits still until the finger is already at the limit and then starts pulling. This is the
single most likely way to hurt someone with this code, so a device-driven phase gets its own
predicate and never borrows that property.

Three more consequences, all specific to a device-driven phase:

- **It needs a range of motion target from the prescription.** Force alone no longer bounds
  where the arm goes: under the cap, nothing tells it to stop, so it would keep pulling until
  the travel limit fires. Active mode needs no such target because the patient's own range is
  the limit.
- **The monotonicity guard does not exist here.** `ControlLoop._check_yield_direction` assumes
  the arm is giving way and the force should therefore fall. While the arm pulls, a rising
  force is the expected thing. What replaces that guard is the range of motion target plus the
  cap, not a weaker version of it.
- **Dropout tolerance has to drop to zero.** `SafetyLimits.max_dropouts` is 2, which is correct
  for active mode: a missing reading there leaves the arm *static*, which is bad but not
  loading anything. In a device-driven phase a missing reading would leave the arm *pulling*
  with no idea how hard. So advancing requires a fresh, confident reading on the current
  cycle, and a single missing frame stops the advance. This is a deadman, not a tolerance.

## What the force loop already covers, with numbers

Active mode's work phase is the existing law and needs no new control code. "Hold below the
cap, then give way at a rate proportional to the overshoot" is `SafetyGate` returning HOLD and
then YIELD with `yield_velocity`, and holding force constant while the finger keeps moving is
what that law settles into. Three measured facts about it that the layers above have to
respect.

### How constant is constant

The force settles at `F_threshold + B x finger speed`, so the error is bounded by the speed.
Measured by `control.simulate` at `B = 10`:

| Finger speed | Force above the cap | In grams |
|---|---|---|
| 4 mm/s | +0.056 N | +6 gf |
| 10 mm/s | +0.100 N | +10 gf |
| 20 mm/s | +0.200 N | +20 gf |

At therapy speeds that sits at the PESOLA's own graduation, 5 to 10 gf, so the force is
constant to the resolution of the instrument measuring it. This is also why no integral term
belongs here: it would buy nothing measurable and would trade a provably passive loop for one
that can wind up next to a healing finger.

### Pre-tension decides how much of the range is actually exercised

Reaching 1.96 N (200 gf) on the PESOLA's ~54 N/m spring takes **36 mm of stretch**. If a rep
starts with the tether slack, the patient's first 36 mm of flexion all goes into stretching
the spring, on a rising force ramp, and only whatever range is left runs at constant force.
Simulated: a 60 mm fingertip excursion gives 36 mm of ramp and 23 mm of accompanying.

So the `ARMED` phase is not a formality. It has to take the slack up to just under the cap,
and how close it gets is what decides how much of the patient's range is the exercise rather
than the run-up to it. This is the strongest argument yet for measuring the real spring rate
on the bench instead of trusting the 54 N/m figure.

### The travel budget as written cannot fit the exercise

At constant force the stretch is constant, so **the arm displaces one for one with the
finger**. One full-range rep is therefore tens of millimetres of arm travel: 23 mm for a 60 mm
excursion, 61 mm for a 100 mm one.

`SafetyLimits.max_travel_m` defaults to 0.05 m. A single rep can consume the whole budget, and
ten reps are impossible under it. The fix is to split the two jobs one limit is currently
doing at once:

- a **per-rep** travel allowance, reset at each `REP_DONE`, which catches a runaway inside a
  rep, which is what the current limit is actually good at;
- a separate **absolute** bound on where the arm may ever be relative to the session start,
  which catches drift accumulating across a set.

Sizing the per-rep number needs the fingertip excursion along the tether for the prescribed
range of motion, which is a bench measurement, not a guess.

## Layer 1: the rep machine, arm 1

One skeleton, both modes. Per repetition. Commands the cue, and counts what the patient
actually did rather than what the clock says they should have done.

```
IDLE
  ARMED       take the slack up to just under the cap, confirm the tether direction
  CUEING      instruction shown and spoken, waiting for the phase's driver to start
  RAMP        force rising toward the cap, arm holding                      (gate: HOLD)
  WORK        at the cap, the work phase's law running
  HOLD        at end of range, counting out the hold time
  RESET       the reset phase's law running, back to the rep's start position
  REP_DONE    count it, log the rep, reset the per-rep travel, back to CUEING
  SET_DONE    the block's rep count is reached, report up
  ABORTED     the gate latched RELEASE, or a human stopped it
```

The mode fills in two slots and nothing else: **which law runs** in `WORK` and `RESET`, and
**who drives** each of those phases. That is the whole of the difference.

| Phase | active | passive |
|---|---|---|
| `WORK` | `yield`, driver = patient | `pull`, driver = device |
| `RESET` | `pull` back to start, driver = device | `yield`, driver = patient |

`RAMP` and `WORK` are separate states even though the force loop handles the transition
between them on its own, because they are different things clinically: `RAMP` is the 36 mm of
spring stretch, `WORK` is the part that is the exercise. The session log and the force against
displacement curve both want them apart.

**Command and confirm**, rather than one or the other. When the phase's driver is the patient,
the transition out of `CUEING` is them starting to move, not a timer expiring, and `REP_DONE`
means the measured signal actually completed the movement. That is what makes a rep count
honest and what detects a patient who is not moving at all, and it dissolves the two
disagreeing rep models: the `GuideHand` timeline becomes a cue, `LiveAnalyzer`'s detector
becomes the counter, and they stop being rivals. That detection logic wants to exist on the
Python side, driven by force from the gauge reader and by flexion from `HandFrame.angles` and
`HandFrame.tip_wrist_mm`, which are the Python side equivalent of the `x` signal the analyzer
already consumes.

## Layer 2: the orchestrator

Per block. One per session, owning both arms. A block is one finger under one mode for a given
number of reps.

```
SESSION_IDLE   load and validate the protocol
for each block:
  POSITIONING  arm 2 moves the hook to this block's finger
  POSITION_OK  confirmed hooked
  EXERCISING   arm 1 runs the rep machine for this block
  BLOCK_DONE   arm 1 reports the reps it completed
SESSION_DONE / SESSION_ABORTED
```

The rep machine counts to ten with the patient, reports up, the orchestrator decides the
finger is finished, arm 2 moves the hook, arm 2 reports the task complete, and the
orchestrator hands arm 1 the next block.

**The interlock is the point, not the sequencing.** Arm 1 and arm 2 are never active at the
same time, and the orchestrator must have arm 1 released and at zero tension before it hands
control to arm 2. A hook being moved while something is still pulling on a finger is the worst
state this machine has. It is one mutual exclusion rule, it is cheap to enforce, and it is easy
to test exhaustively, which is exactly the shape of thing that belongs in a pure module next to
`control/safety.py`.

**Arm 2 sits behind one interface**, with three implementations: the trained policy, a replayed
teleoperated trajectory, or a human moving the hook by hand and confirming on the dashboard.
The orchestrator only ever needs "move the hook to finger F, tell me when you are done or when
you failed". This matters for the hackathon specifically: the policy does not exist yet, and
with a manual implementation behind the same interface the whole session still runs end to end.
Nothing above arm 2 should know which implementation is in use.

## Does arm 1 need a VLA? No, and the protocol is why

The things that need deciding during a session are which finger, which mode, how many reps,
what cap, how long to hold, how far to stretch, and what to say to the patient. All of it is
known before the session starts. There is nothing to learn: it is a prescription, so it belongs
in the repo as **data that the state machine reads**, not as weights.

```python
Block(finger="index", mode="active",  reps=10, cap_n=1.96, hold_s=3.0)
Block(finger="index", mode="passive", reps=10, cap_n=1.96, hold_s=3.0, rom_target_mm=25.0)
```

Ten reps per finger is a number in a file. It is not a thing a policy is trained to contemplate,
and writing it as data is what lets a clinician change it without retraining anything and what
lets the sequencing be unit tested. Note that `rom_target_mm` only appears on the passive block:
the prescription carries exactly the extra field that the device-driven phase needs.

A learned policy would be worse at this job on every axis that matters here. It cannot be given
a force guarantee, where the gate can. The README already argues why chunked policies such as
ACT and Diffusion Policy are wrong next to a post surgical finger, since they execute a chunk
open loop and cannot react to contact mid chunk; a VLA inherits that and adds latency. And arm 1
is the arm touching the patient, so it is the one place in the system where an unverifiable
controller is least acceptable.

Learning does belong in this project, in four places, none of them arm 1's motion: reading the
gauge, hand tracking, per patient cap adaptation within a session, and arm 2's hook moving
policy.

## The instruction table

Today the vocabulary is two strings, `"Flex"` and `"Release"`, inline in `GuideHand.tsx`. Three
consumers need them, the voice, the 3D guide hand, and the session log, so they have to come
from one keyed table or they will drift apart. Keyed by mode and phase, with the finger name
substituted from the block:

| Mode | Phase | Driver | Instruction |
|---|---|---|---|
| active | CUEING | patient | Start flexing your index finger, slowly |
| active | RAMP | patient | Keep flexing |
| active | WORK | patient | Keep going, the arm is following you |
| active | HOLD | patient | Hold it there |
| active | RESET | device | Relax. The arm will bring your finger back |
| passive | CUEING | device | Relax your index finger and let the arm move it |
| passive | RAMP | device | Stay relaxed |
| passive | WORK | device | Stay relaxed, let it stretch |
| passive | HOLD | device | Hold here, stay relaxed |
| passive | RESET | patient | Now flex your finger back to the start |
| any | ABORTED | any | Stop and relax your hand |

Three of these carry more weight than they look. The `active`/`WORK` cue is the first moment
the patient feels the arm move *with* them instead of resisting, and being told it is supposed
to happen is the difference between that feeling like the exercise working and feeling like the
device letting go. Both `RESET` rows announce a change of driver, which is the moment a patient
is most likely to fight the machine. And every device-driven row says "relax", because a
patient tensing against an arm that is pulling is the one thing the force cap cannot distinguish
from a stiff joint.

## Where the dashboard sits, and how it stays mode-agnostic

The dashboard is a view and a command surface. It is not the brain. The browser can be closed,
and the patient facing screen must not be the thing deciding when to stop pulling on a finger.
So the session state lives in Python, the dashboard subscribes to a stream of it, and it sends
back a small set of commands: start, stop, next block, change the cap.

The abstraction is that **the dashboard never learns which law is running.** It renders one
snapshot shape, identical for both modes:

```json
{
  "block":  { "finger": "index", "mode": "passive", "reps": 10, "cap_n": 1.96 },
  "rep":    { "index": 3, "of": 10 },
  "phase":  "WORK",
  "driver": "device",
  "instruction": { "key": "passive.work", "text": "Stay relaxed, let it stretch" },
  "force_n": 1.94,
  "progress": 0.62,
  "travel_mm": 23.4
}
```

`driver` is the field that does the work. It is the only thing the UI needs to decide whether
the guide hand is a **demand** ("do this now") or a **preview** ("this is about to happen to
you"), which is a different animation and a different colour but the same panel. Everything
else renders the same way in both modes, so adding a third mode later changes the instruction
table and nothing in the dashboard. Note `driver` sits next to `phase`, not inside `block`,
because it flips within a rep.

That also fixes the alignment problem. Right now `GuideHand` invents its own timeline, so what
the patient hears and what the patient sees cannot be aligned with what the arm is doing, by
construction. Once it renders the snapshot instead of running a timer, the instruction, the
animation and the force are the same state by definition. Laurence's component is most of the
way there: what has to go is the `performance.now()` loop and the hardcoded `REPS` and
`HOLD_S`, and what arrives in their place is `phase`, `driver` and `progress` off the wire.

## How to organise the repo for this

**No restructure.** The layout is already by feature, pure decision code is already separated
from I/O, and that is the property worth protecting. Add one package and extend one:

```
control/            the per-cycle laws and the gate. Pure, exhaustively tested
  safety.py         + mode-aware: a device-driven phase gets its own predicate, dropouts 0
  admittance.py     + the pull law for device-driven phases, alongside yield_velocity
  loop.py           + the return/advance path, with its own direction check
  simulate.py       + a passive-mode scenario, to size rom_target before touching a finger

session/            NEW. The sequencing. Pure except for server.py
  modes.py          the mode registry: which law and which driver per phase. One table
  protocol.py       the prescription as data: Block, load and validate
  instructions.py   the (mode, phase) -> instruction table, one source for voice, 3D and log
  reps.py           the one rep skeleton, parameterised by mode
  orchestrator.py   the session machine and the arm 1 / arm 2 interlock
  arm2.py           the hook moving interface: policy, replay, or manual
  snapshot.py       the dashboard-facing state shape, serialised in one place
  server.py         the state stream and command endpoint. The only I/O in the package

tests/control/      + the device-driven cases, added rather than editing the yield ones
tests/session/      NEW
```

Three rules make that efficient rather than just bigger:

1. **`session/` is not part of `control/`.** The force loop runs at 30 Hz and must stay small
   enough to stay exhaustively tested; sequencing runs per rep and per block and has no business
   inside that loop. Same reasoning that already keeps `safety.py` pure and separate.
2. **The mode is a record in `modes.py`, never a branch scattered through the code.** One table
   says, per mode and phase, which law and which driver. `reps.py` reads it. That is what keeps
   a third mode from being a refactor, and it is what makes the mode table and the instruction
   table verifiable against each other in a test.
3. **One I/O module per package.** `server.py` here, `loop.py` in `control/`. Everything that
   decides anything stays testable without a robot, a camera or a browser, which is the rule the
   rest of the repo already follows.

Two things not to do. Do not reorganise the repo by model or by feature importance a day before
a demo. And do not let the dashboard own session state because it is the fastest thing to wire
up.

## Open questions

1. **The real spring rate and the fingertip excursion along the tether.** Both are bench
   measurements, and between them they set the pre-tension target, the per-rep travel allowance,
   `rom_target_mm`, and how much of the range runs at constant force. Every number in this doc
   that came from the 54 N/m estimate moves when these are measured.
2. **How far should a passive stretch go, and who decides.** `rom_target_mm` is in the
   prescription above, but a clinician setting a distance in millimetres is an odd unit for a
   joint. It may want to be an angle off the hand tracker instead, which is available but adds a
   dependency between arm 1's control and the camera that active mode does not have.
3. **How does the orchestrator know arm 2 succeeded?** A vision check, the policy's own success
   signal, or the clinician confirming on the dashboard. Defaulting to clinician confirmation is
   the safe and demonstrable choice, and it keeps the manual fallback honest.
4. **Who authors the protocol**, a file in the repo or the clinician on the dashboard. The data
   model is the same either way, so this can be decided late.
5. **The dataset format gap** between Station Parquet and what the SmolVLA and ACT notebooks
   expect. It belongs to arm 2 and it has no owner yet.
