<div align="center">

<img src="docs/images/logo.svg" alt="Flex'N Sense" width="420">

# Force-Limited Hand Rehabilitation

**Two robot arms move a patient's finger against a force cap a clinician sets remotely.**<br>
For recovery after surgery, injury, stroke, or age-related loss of hand strength.

[![Watch the presentation](https://img.shields.io/badge/▶_Watch_the_presentation-0b7285?style=for-the-badge)](https://flexnsense.vercel.app)
[![Try the dashboard](https://img.shields.io/badge/◉_Try_the_dashboard-2f9e44?style=for-the-badge)](https://flexnsense-dashboard.vercel.app)

[![tests](https://github.com/anatissera/flexnsense/actions/workflows/tests.yml/badge.svg)](https://github.com/anatissera/flexnsense/actions/workflows/tests.yml)
[![NormaCore ElRobot](https://img.shields.io/badge/arm-NormaCore_ElRobot_7%2B1_DOF-0b7285)](./docs/elrobot.md)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](./SETUP.md)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-blue)](./LICENSE)
[![WiLoR: CC BY-NC-ND](https://img.shields.io/badge/WiLoR-CC_BY--NC--ND-orange)](./docs/licences.md)

</div>

![The finger pulls a spring scale held by the arm. A camera reads the scale in grams. On reaching the 200 g cap the arm starts moving with the finger, so the reading stays at 200 g.](docs/images/tether.gif)

The patient pulls, a camera reads the force off a PESOLA spring scale thirty times a second,
and on reaching the prescribed cap the arm starts **moving with the finger**, at whatever speed
holds the reading there. It is not giving way and it is not letting go: the finger keeps working
against a force the clinician chose, for as far as it can travel. What the patient feels is set
by the control law, not by how hard they pull.

That is `active` mode, one of two. In `cpm` the arm drives the whole stroke and the finger goes
along, for a patient who cannot flex yet.

## Why this is hard

A healing finger tolerates very little. Dynamic finger splints in hand therapy work in the
**0 to 450 gf (0 to 4.4 N)** range, and past that you are not treating a tendon, you are
damaging one. So the force has to be known continuously and bounded by something that keeps
working when the software does not.

The kit has **no force sensor**. The obvious substitute, servo current, was measured and ruled
out: the 345:1 gearbox leaves a ~1 mA noise floor regardless of applied force
([docs/force-calibration-result.md](./docs/force-calibration-result.md)). That negative result
is why force sensing is entirely on the camera, and why a lost or low-confidence frame has no
fallback to fall back on, only "release and stop".

This is deliberately **not** a learned policy on the arm that touches the patient. ACT and
Diffusion Policy execute action chunks open-loop and cannot react to contact forces mid-chunk,
which is the exact failure a device pulling on a post-surgical finger must not have.

## How it works

### Reading force from a camera

Two red dots on the PESOLA's tube are fixed references. Each frame the reader finds three red
spots in a row (dot, marker, dot); the marker's position *between* the dots is a fraction that
does not change when the scale moves, turns, or comes closer, and calibration turns that
fraction into grams. When a dot is covered it reports a reason instead of a number.

![Gauge reader: find the red spots, the outer two are the dots and the middle one the marker, the marker's fraction between the dots becomes grams](docs/images/gauge-reader.svg)

Details, calibration and failure modes: [docs/scale-reader.md](./docs/scale-reader.md).

### The control law

Admittance control with a deadband, clinically *assist-as-needed*. The gauge reads grams, which
is what the PESOLA is graduated in and what the clinician sets; everything past that reading is
in newtons.

Every frame resolves to one of three outcomes. Below the cap the arm **holds**, because it is
the anchor and the patient is meant to work against it. Above the cap it **moves away**, at a
speed proportional to how far over it is, `v = min((F - cap) / B, v_max)`. That proportionality
is the whole trick: the further the reading drifts past the cap the faster the arm retreats, so
it settles at exactly the speed the finger is advancing and the reading sits on the cap. With no
reading, a stale one, or one past the hard ceiling, it **releases and stops**. There is no fourth
branch and no guessed force.

`arm1/control/safety.py` is the gate, `arm1/control/admittance.py` the law, and
`arm1/control/loop.py` the only part that touches hardware. The first two are pure and
exhaustively tested. Why `B` is bounded by the gap between the cap and the ceiling:
[docs/admittance-design.md](./docs/admittance-design.md).

### Three joints, not one

A revolute joint moves the gripper along an **arc**, and the tether does not bend, so whatever
leaves the line becomes side-load on the finger or slack in the tether. One full-range
repetition needs 72 to 88 mm of travel. Asked for 90 mm along the tether from the saved home
pose, no single joint can deliver it:

| Joints | Distance covered | Deviation from the line |
|---|---|---|
| Best single joint (motor 4) | 21.8 mm of 90 | 38.1 mm |
| Motors 1, 2 and 4 together | 89.9 mm of 90 | 0.9 mm |

The joints with low deviation in isolation (5 and 6) score that way because they barely move
along the tether at all. Coordination is not a refinement here, it is what makes the stroke
possible.

So the step is solved, not chosen. For a desired tether displacement **d**, the joint step is
the least-squares solution of **J q̇ = d**, where **J** is the 3 x n jacobian of tool position
with respect to the pull joints at the current pose:

**q̇ = J⁺ d**,  with **J⁺ = Jᵀ(J Jᵀ + λ²I)⁻¹**

The damping term λ is what keeps the step finite near a singularity, where an undamped inverse
would ask a joint for an enormous move to buy a millimetre. Two joints are not enough for the
same reason the table shows: **J** is then rank 2, the reachable directions collapse to a plane,
and any stroke off that plane simply misses. `CoordinatedAxis` refuses fewer than three joints,
and refuses a pose that cannot follow the pull to the end.

All of it is read from NormaCore's URDF at the saved home pose, so no simulator and no bench
are involved.

### Two exercise modes

| Mode | Work beat | Reset beat |
|---|---|---|
| `active` | the patient flexes, the arm moves with them to hold the force at the cap | the arm walks back to where the rep began |
| `cpm` | the arm takes the finger out to the stretch target | the arm brings it back; the patient is passive throughout |

`cpm` is the only mode where the driver does not flip within a repetition, which is the point of
it: a repetition that waits on nobody is the same motion every time, so it can run faster and
more smoothly than an arm chasing a signal. It also means a lost force reading stops *both*
halves of the stroke, not one.

Both modes count honestly. A repetition is counted only once the force actually reached the cap,
so a patient who moved a little and came back shows as an **abandoned attempt**, not a rep.

### Safety

Force limits on the patient are enforced **independently of the control loop**:

- a current cap written to the servo registers by Station's own `station.yaml`, in force even if
  every Python process dies;
- software position limits on total travel;
- a vision watchdog: on lost detection, low confidence or dropped frames, **release and stop**,
  never hold;
- a per-cycle deadman on any phase where the arm moves the finger: one missing frame stops the
  advance;
- a low velocity ceiling throughout, and a physical e-stop within reach;
- the spring itself as a mechanical fuse.

**A missing force reading is never evidence of a safe force.** That single rule is why the gate
refuses to guess, why the deadman stops a blind pull, and why the two-machine lease expires on
silence rather than persisting.

This is a research prototype. **Nothing here has touched a patient**, and the bench results,
the known faults and what is still untested on hardware are in
[docs/status.md](./docs/status.md).

## The two arms are trained completely differently

This is the part worth understanding, because the two arms sit at opposite ends of the
"learned versus specified" question, on purpose.

### Arm 1, the tether: no learning at all

Fixed control law, hand-tuned constants, exhaustively tested gate. Nothing about it is learned,
because the thing it does (bound a force on a healing finger) is a specification, not a pattern
to be inferred from data. The constants live in
[`config/session-tuning.json`](./config/session-tuning.json), each one annotated in
[`arm1/session/tuning.py`](./arm1/session/tuning.py) with the bench failure it is paying for.

### Arm 2, the hook mover: behaviour cloning from teleoperation

Moving the scale's hook from one finger to the next is fiddly, contact-rich, and has no clean
analytic solution. So it is learned, from demonstrations:

1. **Record.** A human teleoperates arm 2 through the move with a leader arm, while Station
   records joints and three cameras. [`arm2/data_collection/`](./arm2/data_collection/)
2. **Augment.** WiLoR runs on the side camera and writes 21 hand keypoints into every frame, so
   the policy sees where the fingers actually are, not just where the arm is.
3. **Train.** ACT on the resulting LeRobot dataset, one multi-task model covering all three
   finger-to-finger moves, selected by a `--task` phrase.
   [`arm2/training/`](./arm2/training/)
4. **Run.** The policy drives the arm through Station, with its own goal-rate limits, abort rule
   and stalled-motor hold. It does **not** implement this project's safety layer and must not be
   used near a patient's hand.

**226 teleoperated episodes**, publicly available:
[JustoDario/flexnsense_skill1](https://huggingface.co/datasets/JustoDario/flexnsense_skill1).

A policy cannot confirm its own success, so it never reports one. A person watching the arm
presses a button, `stop_and_home.py` stops the policy and homes the arm, and its **exit code**
decides what the session is told. That is the whole reason `PolicyHookMover` raises instead of
returning a plausible outcome: a hook reported as moved when it has not moved means arm 1 starts
pulling on a finger the hook is not attached to.

### Both arms in simulation

The bench is rebuilt in NVIDIA Isaac Sim, with the scale as a real physical object rather than
a scripted constraint, and both arms are trained together in Isaac Lab across thousands of
tables at once. **Simulation only**: none of these policies has moved the real arm.
[`isaac-sim/`](./isaac-sim/)

![Isaac Sim: the pull arm giving way, seen from the bench, the overhead camera that reads the scale, and the camera on the move arm. The force is printed on the frame.](docs/images/isaac-sim.gif)

The force on the frame is the simulator's own, so it shows directly whether a policy stayed
under the 1.96 N limit, which a video of the arm alone cannot.

![Isaac Lab: one table before training, with the share of tables over the force limit and the share with the loop on its target finger printed on the frame](docs/images/isaac-lab.gif)

Before training, 47.6% of tables are over the force limit and none has the loop on its target
finger. Those two numbers are the reward, and watching them is how a training run is judged.

## Running a session

A session is one command against the real arm, naming the finger, the cap in grams and the
repetitions. It takes **no tuning flags**: the bench-proven profile in
`config/session-tuning.json` supplies them, and the gate's limits are printed on every run. A
clinician's prescription can be passed as a file instead of flags, and `cpm` has its own entry
point because it walks between two saved poses rather than chasing a force. Every command, with
its arguments: [scripts/README.md](./scripts/README.md).

### Across two machines

Arm 2 and its cameras run on one computer, arm 1 and the force camera on another. **Arm 1's
machine is the brain**, and that is not a preference: every safety decision is a function of the
force reading, so putting the brain on the machine that owns the force camera keeps all of them
off the network.

Custody is a **lease with a deadline**, not a flag. A flag set by "you may move" stays set when
the link dies and the far arm keeps going with nobody able to stop it; a lease has to be renewed,
so silence means stop. It is also timed on the receiver's clock, since two machines' monotonic
clocks have no relation to each other. Four message types, no shell commands, and why:
[docs/two-machine-session.md](./docs/two-machine-session.md).

## The dashboard

![The dashboard, with each region labelled: the tendon legend, the exercise wizard and the chat down the left; the digital twin, a 3D hand with its flexor tendons tinted by estimated recovery, in the middle; the patient's record, the live displacement and force traces, and the per-finger recovery index down the right](docs/images/dashboard.gif)

The digital twin in the middle is a 3D MyoHand running in MuJoCo WebAssembly, driven by hand
tracking on the laptop camera, and the chat answers questions about what is on screen in any
language. The traces on the right are the same displacement and force the control loop sees.

The same build runs with nothing attached and beside the rig, and the difference is one
environment variable. With it set nothing is probed and **every panel carries a badge saying
where its numbers come from**, because a simulated force that looks like a measured one is the
one thing a clinical demonstration must not do. When a real session is running, a snapshot
stream replaces the simulation and the panel is badged `live session`.
[docs/dashboard-demo.md](./docs/dashboard-demo.md)

## Setup

A virtual environment and the requirements file are the whole of it, and **the entire test
suite runs with no camera, no robot and no Station**. The arm itself, Station, the motor ids
and `.env` are in [SETUP.md](./SETUP.md). There is **no LeRobot support for the ElRobot**, so
every path to the arm goes through NormaCore's Station: [docs/elrobot.md](./docs/elrobot.md).

## Repository layout

```
arm1/             the tether arm: inverse kinematics and a fixed control law, nothing learned
  lib/            shared helpers, no device knowledge: camera, .env, Kalman filter, geometry, kinematics, Station client
  control/        the admittance controller: safety gate, the law, the hardware loop, simulation
  session/        prescriptions, exercise modes, the rep machine, the interlock, the two-machine protocol
  gauge_reader/   read the PESOLA scale through a camera
  hand_tracking/  finger flexion angles (MediaPipe on CPU, WiLoR for precision capture)
arm2/             the hook mover: teleoperated recordings and the policy trained on them
  data_collection/  record episodes through Station, with hand keypoints in every frame
  training/         Station Parquet to LeRobot, ACT training, inference on the arm
scripts/          grouped by what they do to the rig (scripts/README.md)
  session/        run an exercise with a finger in the hook
  arm/            move or configure the arm, nothing attached
  measure/        produce a number the repo then depends on
  preview/        render or inspect, nothing moves
config/           committed: joint calibration, bench measurements, tuning profile, example protocols
dashboard/        the Next.js clinician dashboard (own package.json)
isaac-sim/        both arms in Isaac Sim and Isaac Lab (simulation only)
sim/              URDF-level checks that need no simulator
tests/            one directory per package above
docs/             guides, design documents, the contributor guide and the pitch deck
```

`arm1/` and `arm2/` is the division the two halves of this project actually have: the tether
arm is solved with inverse kinematics and a control law nobody trained, and the hook mover is a
policy cloned from teleoperation. They share `lib/` and Station, and nothing else.


## Configuration, and which machine holds what

| File | Holds | Committed |
|---|---|---|
| `config/session-tuning.json` | the machine's own tuning: damping, speeds, camera gating | yes, it describes the shared rig |
| `config/protocols/*.json` | a clinician's prescription: finger, mode, reps, cap, hold | yes, it is shared clinical intent |
| `config/calibration/*.json` | joint ranges, home poses, bench constants | yes, measured not chosen |
| `config/scale_calib.json` | where *this* camera sees the scale | no, per machine |
| `.env` | Station host, bus serials, camera index | no, per machine |

The dividing question for tuning versus prescription is "would a different patient change this
number?" If yes it is a prescription, if no it is tuning.

Three values in the tuning profile are deliberately **looser** than what the code shipped with:
the hard ceiling, the blind-frame count, and the yield-direction check being off. They sit in
their own `safety` block and are printed on every run, because a loosened limit nobody is shown
on the way past stops being a decision and becomes an accident.

Which physical arm is which, bus serials, voltages:
[docs/hardware-setup.md](./docs/hardware-setup.md).

## Models and data

| Repo | What it holds |
|---|---|
| [JustoDario/flexnsense_skill1](https://huggingface.co/datasets/JustoDario/flexnsense_skill1) | 226 teleoperated episodes on the real arm 2: three finger-to-finger tasks, three cameras, hand keypoints per frame |
| [JustoDario/flexnsense_act_skill1_rightindex2middle](https://huggingface.co/JustoDario/flexnsense_act_skill1_rightindex2middle) | The first ACT policy, the one that ran on the arm |
| [JustoDario/flexnsense_smolvla](https://huggingface.co/JustoDario/flexnsense_smolvla) | SmolVLA fine-tuned on the same episodes. Untested on the arm |
| [peresjav/flexnsense-sim-policies](https://huggingface.co/peresjav/flexnsense-sim-policies) | The Isaac Lab policies, training code and videos. Simulation only |
| [peresjav/lerobot-2026](https://huggingface.co/datasets/peresjav/lerobot-2026) | 512 simulated episodes per policy, Parquet, no images |

## Documentation

| Document | Covers |
|---|---|
| [docs/status.md](./docs/status.md) | What has run on the bench, what has not, and the known faults |
| [SETUP.md](./SETUP.md) | Python, Station, motors, `.env` |
| [scripts/README.md](./scripts/README.md) | Every command, grouped by what it does to the rig |
| [docs/elrobot.md](./docs/elrobot.md) | The arm and Station: why not LeRobot, what is verified |
| [docs/hardware-setup.md](./docs/hardware-setup.md) | Which arm is which, bus serials, who runs what |
| [docs/bench-measurements.md](./docs/bench-measurements.md) | How to take every bench number this repo depends on |
| [docs/admittance-design.md](./docs/admittance-design.md) | The control law, choosing `B`, what the simulation established |
| [docs/session-orchestration.md](./docs/session-orchestration.md) | Modes, the rep machine, the two-arm interlock |
| [docs/two-machine-session.md](./docs/two-machine-session.md) | The socket protocol, the lease, the four rules a network must not break |
| [docs/scale-reader.md](./docs/scale-reader.md) | The gauge reader: calibration, precision, troubleshooting |
| [docs/hand-tracking.md](./docs/hand-tracking.md) | MediaPipe and WiLoR, the flexion angle, the VRAM question |
| [docs/force-calibration-result.md](./docs/force-calibration-result.md) | Why servo current cannot stand in for a force sensor |
| [docs/dashboard-demo.md](./docs/dashboard-demo.md) | The two dashboard builds, what is real in each |
| [docs/licences.md](./docs/licences.md) | Every dependency's licence, and the one that would block a product |
| [docs/contributing.md](./docs/contributing.md) | Git workflow, units, rules for contributors and coding agents |

`main` is protected: work on a branch and open a PR.

## Team

[Ana Tissera](https://github.com/anatissera) ·
[Laurence Liang](https://www.linkedin.com/in/laurence-liang-innovator) ·
[Justo Valverde](https://www.linkedin.com/in/justodariovalverde) ·
[Javier Peres](https://www.linkedin.com/in/peresjavier) ·
[Guillaume Bonifas](https://www.linkedin.com/in/guillaumebonifas) ·
[Damien Frechou](https://www.linkedin.com/in/damien-frechou)

Thanks to the organizers at the Edelweiss Innovation & Intelligence Institute, HerCode, AI Team,
EuroTech Federation and TUM-XR, and to the hardware partners.
