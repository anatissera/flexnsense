# Roadmap

One branch per subtask, one PR into `main`. Order matters where noted.

## 0. ElRobot via NormaCore Station, not LeRobot - `feature/elrobot-support`

**Also blocks everything else that touches the arm.** There is no LeRobot support for the
ElRobot; every script talks to it through NormaCore's own binary, Station, via `lib/station.py`.
See [docs/elrobot.md](./docs/elrobot.md).

## 1. Force calibration - `feature/force-calibration`

**Blocks everything else.** Establish whether servo current can stand in for a force sensor.

- Log current from arm B's servos (via Station's `st3215/inference` queue) while known forces are applied along the therapy axis.
- Sweep up and back down to expose hysteresis from the 345:1 gearbox.
- Fit grams = f(current − baseline); report the dead zone and the noise floor.

Guide: [docs/force-calibration.md](./docs/force-calibration.md).

**Exit criterion:** either a usable current→grams curve with a dead zone well under 200 g, or a documented negative result that moves force sensing entirely onto the camera.

## 2. Gauge reader - `feature/gauge-reader`

Read the PESOLA's red indicator from a fixed camera and output grams + a confidence value.

- Rectify using fixed reference points on the tube (ArUco markers or the two scale endpoints), then project the red blob's sub-pixel centroid onto the scale axis - the scale is linear, so two references are enough. Do not try to detect every tick.
- Handle specular glare off the aluminium: the highlight can saturate to white and swallow the red.
- Temporal filter (EMA or Kalman); a jittery force reading drives a jittery robot against a finger.
- Emit confidence explicitly: blob area out of range, multiple red blobs, or nothing found → low confidence, which the watchdog consumes.

## 3. Admittance controller + safety layer - `feature/admittance-control`

The control loop from the README, plus the safety layer that does **not** depend on it: servo-level torque cap, software position limits, vision watchdog that releases tension, velocity ceiling, e-stop.

Force comes from the camera alone: item 1 came back negative, there is no current signal on this arm ([docs/force-calibration-result.md](./docs/force-calibration-result.md)).

Written, tested in software, **not yet run against the arm**: `control/safety.py` (the gate: HOLD / YIELD / RELEASE), `control/admittance.py` (the law), `control/loop.py` (the wiring to the camera and Station), `control/simulate.py` (closed-loop simulation for choosing constants). Design and what the simulation established about B: [docs/admittance-design.md](./docs/admittance-design.md).

Still to do: measure `ticks_per_metre` and the yield direction on the bench, then a bench run against a spring and a weight before any finger.

## 4. Simulation rig - `feature/sim-rig`

MuJoCo model plus a linear spring (F = k·x, k measurable from the PESOLA in minutes) and a scripted "patient finger" trajectory. NormaCore ships URDF/STEP/STL for the ElRobot (`hardware/elrobot/simulation` in their repo), but it hasn't been checked for use here yet.

Purpose is **de-risking the controller without a human in the loop** - oscillation, force overshoot, behaviour when the force reading freezes or spikes. Not for generating training data.

Can run in parallel with 1–2.

## 5. Hand tracking / ROM - `feature/hand-tracking`

Finger range of motion during the session. MediaPipe Hand Landmarker on CPU is the safe default, done: `hand_tracking/` reports one flexion angle per finger, see [docs/hand-tracking.md](./docs/hand-tracking.md). WiLoR is still worth adding alongside it, and it outputs MANO joint rotations, which is the input format a musculoskeletal model would later want, something a bare per-joint angle does not give.

The 4 GB question is now answered: WiLoR in fp16 peaks at 1.69 GB of VRAM on the RTX 3050 and runs at 11 FPS on 640x480 with two hands once its pyrender overlay is dropped (the dashboard draws its own hand, so that overlay, 73% of the cost, is not needed). Benchmark and the full argument in [docs/hand-tracking.md](./docs/hand-tracking.md). 11 FPS and a 2.5 GB checkpoint behind a server process mean WiLoR is the precision-capture path, not the live one: MediaPipe stays the default in the dashboard. Both already emit the same `HandFrame` (`hand_tracking/frame.py`, `dashboard/app/lib/handFrame.ts`), so the remaining work is the WiLoR service itself, not the plumbing around it.

Independent of 0–4; camera-only, does not touch the arm.

## 5b. Session orchestration - `feature/session-orchestration`

The layer between the 30 Hz force loop and the clinician: the two exercise modes (`active`,
the patient flexes and the arm yields at constant force; `passive`, the arm pulls the finger
toward extension and the patient flexes back), the per-rep state machine that cues the patient
and counts what they actually did, the session state machine that sequences blocks across
fingers, and the interlock that guarantees arm 1 and arm 2 are never active at the same time.
The exercise prescription is **data** (finger, mode, reps, cap, hold, stretch target), not a
trained policy.

Designed in [docs/session-orchestration.md](./docs/session-orchestration.md). Written so far, all pure and tested without hardware: `session/modes.py` (the two modes as data), `session/protocol.py` (the prescription), `session/instructions.py` (what the patient is told), `session/calibration.py` (the bench constants, as explicit absences), `session/snapshot.py` plus `dashboard/app/lib/sessionSnapshot.ts` (the UI contract), `session/interlock.py` (the two-arm custody rule) and `session/orchestrator.py` (the session machine). Still to write: `session/reps.py` (the per-rep machine) and `session/server.py` (the state stream the dashboard reads).
Active mode's work phase is the existing admittance law; the delta on the control side is the
device-driven phase, where `Decision.may_move` is backwards, dropout tolerance has to go to
zero, and a range-of-motion target replaces the monotonicity guard. That doc also inventories
what already exists in the dashboard (`GuideHand`, `LiveAnalyzer`) and lists the open
questions, the unmeasured spring rate being the one that moves the most numbers.

Depends on 3, whose `control/` package the rep machine drives. Blocks 6 and 7, and it is the
contract arm 2's policy has to meet.

## 6. Session metrics - `feature/session-metrics`

Record force against finger displacement across the session and derive the joint stiffness curve. This is the clinical deliverable.

Depends on 3 and 5.

## 7. Remote clinician dashboard - `feature/remote-dashboard`

Live force, position, ROM and session metrics; the clinician sets the threshold and can stop the session.

Depends on 6.
