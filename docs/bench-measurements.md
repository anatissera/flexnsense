# Bench measurements: how to take them

Three numbers have to be measured on the real rig before the controller can drive the arm.
This is the procedure, written so that whoever has the rig in front of them can take them
without having followed how the code got here.

Every number lands in `config/calibration/bench.json`. The scripts write it for you; you do
not edit it by hand. A `null` in that file means **not measured yet**, and the code refuses
to run on a missing number rather than substituting a default, so an unfinished calibration
is a safe state to leave behind.

Record what you did in `config/calibration/bench.md` as you go, including anything that went
wrong or felt wrong. A number with no note about how it was taken is nearly useless later.

## Read this first

| Measurement | Needs | Powered arm | How long |
|---|---|---|---|
| 1. Tether stiffness | PESOLA, a ruler, a fixed anchor | No | 15 min |
| 2. Fingertip excursion | A camera and a hand | No | 20 min per finger |
| 3. `ticks_per_metre`, `yield_sign` | Station, the arm locked in the session pose, a ruler | **Yes** | 30 min |

1 and 2 are independent of each other and of 3, so they can be done in any order, or by
different people. 3 wants 1 done first: it prints a predicted force change per move, which is
a useful cross-check, and it can only do that once the stiffness is known.

Two things about 3 that are worth knowing before starting rather than discovering:

- `--motor <id>` is the joint that moves along the tether axis **for the pose the rig is in**.
  The script cannot infer it, and the repo genuinely does not record it: `docs/elrobot.md` has
  a section titled "What we have not yet verified" which says that beyond "ids 1 to 7 are the
  arm joints in the manual's assembly order, 8 is the gripper", the mapping has to be confirmed
  per unit and must not be assumed. There is a safe way to find it, below, but it needs doing
  at the bench before the script is any use. The results are only valid for that pose.
- The arm is under tension throughout. The script returns it to baseline between moves, and
  also on a crash or a Ctrl-C, but it is a real arm pulling on a real spring and it wants
  someone paying attention rather than someone reading prompts.

---

## 1. Tether stiffness

**What it is.** How hard the tether pulls back per millimetre you stretch it, in newtons per
metre. Not the spring alone: the whole chain of spring, cord, hooks and attachments, which is
what a finger actually pulls against and is always softer than the spring by itself.

**Why it matters.** It decides how much of a patient's finger range is spent stretching the
tether before any force is reached, and it is the constant the controller's damping is chosen
against.

**Already measured once: 56.2 N/m.** If you are re-measuring, you should land near that. A
result far from it means something is set up differently, not that the tether changed.

### Rig

1. Anchor **one PESOLA hook to something that will not move**: a table leg, a chair leg, a
   clamp. Not the robot. Using the gripper as the anchor would drag a powered arm into a
   measurement that otherwise involves no robot at all, and there is no reason to.
2. Pick **two reference marks** to measure between, one on the anchored side and one on the
   hook that moves. Any two marks will do as long as neither moves during the sweep; the fit
   absorbs where you chose them. A strip of tape with a pen line on it works.
3. You will pull the free hook by hand. No weights are needed: the PESOLA is the force
   reference.

### Run it

```bash
.venv/bin/python scripts/measure_tether_stiffness.py
```

It asks you, for each force level going up and then the same levels coming back down:

```
pull to 50 gf, distance between the marks in mm (blank = skip, q = end sweep):
```

Pull until the needle sits on that number, **hold it there**, and measure the distance between
your two marks. Type the number in millimetres.

- Blank skips that one level and carries on.
- `q` ends that direction early.
- Anything that is not a number re-asks the same question. Nothing is lost.

Do **both** directions. The down sweep is not redundant: the difference between up and down is
the tether's hysteresis, which is a number the project needs in its own right.

### Check the result before you trust it

The script prints `k` in N/m. **Sanity-check it against roughly 53.5 N/m**, which is what the
PESOLA's printed scale alone implies. You should get something a little stiffer, since the cord
adds to it.

> **The one mistake that fails silently:** the prompt asks for millimetres. If you read the
> ruler in centimetres and type that, the script fits a stiffness about ten times too high and
> says nothing. It cannot tell. This already happened once. If `k` comes out in the hundreds of
> N/m, you typed centimetres.

Also look at `r2` and the worst residual. The existing measurement got `r2 = 0.995` with a worst
residual of 3.2 mm, which is about a ruler's own precision. A much worse fit means the marks
moved or the needle was not settled when you read it.

### What it writes

`tether_n_per_m` and `displacement_uncertainty_m` into `config/calibration/bench.json`, and the
raw sweep into `data/tether_stiffness.csv`. To re-fit an existing sweep without measuring again:

```bash
.venv/bin/python scripts/measure_tether_stiffness.py --fit data/tether_stiffness.csv
```

---

## 2. Fingertip excursion

**What it is.** How far a fingertip travels along the tether axis as the finger goes from fully
extended to fully flexed, in metres, per finger.

**Why it matters.** It sets how far the arm is allowed to travel during one repetition, and it
bounds how far a passive stretch may pull. Fingers differ, so this is measured per finger.

**Not measured yet for any finger.** The middle finger is the most useful one to start with,
being the longest.

### Rig

A camera the hand-tracking code can open, and a hand in **the pose the session will actually
run in**. The pose matters: the same finger projected onto a different tether axis gives a
different number.

No robot, no tether, no tension. You are only being filmed.

### Run it, in two steps

The order matters and the first step is easy to get wrong.

```bash
.venv/bin/python -m hand_tracking run --jsonl data/excursion-middle.jsonl
```

**The `--jsonl` flag is required.** Without it the command still runs and still writes a file,
but that file has joint angles only and none of the 3D coordinates this measurement needs. The
fitting step will not be able to read it.

While it records, move the finger **slowly and deliberately** from full extension to full
flexion and back. Slowly is the whole technique: a quick flick gives the tracker too few frames
to fit a direction through.

Stop the recording, then fit it:

```bash
.venv/bin/python scripts/measure_tip_excursion.py data/excursion-middle.jsonl --finger middle
```

`--finger` is one of `thumb`, `index`, `middle`, `ring`, `pinky`. Omit it to fit all five from
one recording, if you moved all five.

> Note the finger is called `pinky` here, because that is what the hand tracker calls it. The
> clinical side of the codebase calls the same finger `little`. Both spellings are accepted
> wherever it matters; this is not a mistake.

### Check the result before you trust it

The script prints a number of millimetres, and **next to it, a flag if the recording was no
good**:

- `<-- not a clean pull` means it could not fit a clear direction of motion. Usually: moved too
  fast, or the hand drifted while flexing.
- `<-- no real motion` means under 2 mm of travel. Usually: the wrong finger moved, or the
  tracker lost the hand.

**Read for those two strings specifically.** A plausible-looking number prints either way, and
it is easy to see the number and miss the warning beside it. A flagged finger is deliberately
**not** written to the calibration file, so if nothing was written, that is why.

Re-record rather than accepting a flagged result.

### What it writes

`excursion_m["<finger>"]` into `config/calibration/bench.json`, one entry per good recording.

---

## 3. `ticks_per_metre` and `yield_sign`

**What it is.** How many servo position ticks correspond to one metre of movement along the
tether axis (`ticks_per_metre`), and which direction of tick change *reduces* tension
(`yield_sign`, either +1 or -1).

**Why it matters.** Without it the controller cannot convert a decision in metres into a command
in ticks, so nothing about the arm can run. Getting `yield_sign` backwards means the arm tightens
the tether when it means to release it.

**Both are pose-dependent.** They are only valid for the arm pose they were measured in. If the
session pose changes, measure again.

### This measurement could not be taken on this arm, and the reason is not fixable

Every procedure below moves **one** joint and measures how far the attachment point travelled.
That assumes some joint moves the gripper along the tether. On the ElRobot, in every pose tried
on the bench, none does: a revolute joint moves the gripper along an arc, and the best-aligned
joint at the session pose reaches only about 0.4 of the tether's direction, with the rest of its
travel sideways.

It failed three times on 2026-11-15, and the failures looked different enough to hide the common
cause:

- motor 4, camera method, large steps: the tether went slack partway, `dx = dF/k` underestimated
  the travel, and the fit came out at 38524 ticks/m against a geometric prediction of ~3900;
- motor 4, smaller steps: moving further in one direction first slackened the tether and then
  tensioned it again, because the arc crossed the tether's axis. `fit_ticks_per_metre` refused
  the data, correctly, with "moved the opposite way from the rest";
- motor 1: the same refusal. Motor 1 rotates the base, so it swings the gripper sideways across
  the tether rather than along it.

No amount of re-measuring fixes this, because the quantity being measured is not well defined for
a single joint. What replaces it is `scripts/run_session.py --coordinated`, which moves three
joints together and derives the equivalent conversion, the tether's direction, and `yield_sign`
from the URDF and the live pose. See the "no single joint follows the tether" section of that
script's docstring.

`yield_sign` **is** measured and committed: -1, from the camera run above. A direction is a much
weaker thing to ask of bad magnitude data than a scale factor is, and the URDF geometry agrees
with it independently (`tests/lib/test_kinematics.py`), which is the only reason it is trusted.

What `--coordinated` still does not know is backlash, so it is a starting point, not a
calibration. If a way is found to measure travel along the tether under coordinated motion, it
belongs here.

### Prerequisites, none of which the script can check

1. Station running, and the norma-core Python client set up (`SETUP.md`).
2. `FOLLOWER_BUS_SERIAL` set, or passed as `--bus-serial`.
3. The arm locked in the session pose: `.venv/bin/python scripts/lock_arm.py --motors <id>`.
4. The PESOLA anchored between the gripper and a fixed point, under light pre-tension.
5. **No finger anywhere near the rig**, and nothing else that can be damaged by a pull.
6. You know **which motor moves along the tether axis**, for `--motor`. Do not guess it.

### Finding the right `--motor`, safely

If nobody present already knows the id, identify it before running anything, and do it with the
**tether deliberately slack**:

1. Rig the tether but leave it visibly loose, so no motor can load it.
2. In Station's web UI, jog one motor at a time, a small amount.
3. Watch which motor's rotation visibly moves the attachment point *along the pull direction*,
   rather than sideways or in an arc that does not change the tether's length.

Nothing is under tension while the tether is slack, so a wrong guess here costs nothing, which
is exactly why this is the step to get wrong rather than the measurement. Write the id you found
into `config/calibration/bench.md` along with a description of the pose, because the next person
will need both and neither is recorded anywhere else.

### Why a ruler, and not the camera

An earlier version of these instructions said to infer the displacement from the force change,
as `dx = dF / k`, so that no ruler had to touch the arm. **That method is wrong here and must
not be used.** The tether's measured hysteresis is 6 mm, and any distance inferred from a force
reading inherits that as a floor: 40 % error over a 15 mm move, and still 10 % over 60 mm, which
is most of the tether's usable travel. A ruler reading is good to about a millimetre, so 3 % over
30 mm. The ruler wins by an order of magnitude.

So: mark the gripper against a fixed reference and measure the mark with a ruler. Move as far as
the pose safely allows, because the ruler's fixed error divides down over a longer move.

### Run it

```bash
.venv/bin/python scripts/measure_ticks_per_metre.py --motor <id> --ticks 30,60,90,-30,-60,-90
```

Before it connects to anything, it asks you to type a literal phrase:

```
Type 'no finger attached' to confirm there is none, anywhere near the rig:
```

It has to match exactly. There is no flag to skip it, on purpose. It is a different phrase from
the per-move `yes` prompts so that it cannot be answered out of habit.

Then it reads the arm's baseline and asks for a baseline ruler reading and a baseline force. For
each tick delta it previews the force change it expects, asks `yes` to proceed (anything else
skips that delta), moves, settles, asks for the new ruler and force readings, and **returns to
baseline before the next one**.

Including both positive and negative deltas is what determines `yield_sign`.

### If something goes wrong

The script returns the arm to baseline on a crash or a Ctrl-C. If it ever reports that it could
**not** return to baseline, the arm is sitting at an unknown position under tension and the fix
is physical: release torque or cut power, by hand, now.

### What it writes

`ticks_per_metre` and `yield_sign` into `config/calibration/bench.json`, plus the pose
description you pass as `--pose`.

---

## When you are done

Check what is still missing:

```bash
.venv/bin/python -c "
import json, pathlib
p = pathlib.Path('config/calibration/bench.json')
d = json.loads(p.read_text()) if p.is_file() else {}
missing = [k for k in ('tether_n_per_m','ticks_per_metre','yield_sign') if d.get(k) is None]
if not d.get('excursion_m'): missing.append('excursion_m')
print('missing:', missing or 'nothing, calibration is complete')
"
```

Then note in `config/calibration/bench.md` what you measured, what you skipped, and anything
that looked odd. The next person reads that file before they read any of the code.

## What still cannot be tested without the arm

For completeness, so nobody thinks the list above is the whole validation. These need a powered
arm and a person watching, and none of them is scripted yet:

- that yielding actually loosens the tether in the real rig, confirming `yield_sign` in practice
  rather than on paper;
- that covering the camera mid-pull stops the arm advancing **on the first frame**, which is the
  single most important behaviour to confirm by hand (`session/reps.py`, the device-driven
  deadman);
- that the two-arm interlock refuses a hook move while the tether is still loaded
  (`session/interlock.py`);
- a first finger session at a deliberately low cap, 50 gf or so, in each exercise mode.

Details of what each one is checking: [docs/session-orchestration.md](./session-orchestration.md).
