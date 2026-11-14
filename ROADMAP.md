# Roadmap

One branch per subtask, one PR into `main`. Order matters where noted.

## 1. Force calibration - `feature/force-calibration`

**Blocks everything else.** Establish whether servo current can stand in for a force sensor.

- Log `Present_Current` / `Present_Load` from arm B's servos while known forces are applied along the therapy axis.
- Sweep up and back down to expose hysteresis from the 1/345 gearbox.
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

Depends on 1 or 2 for a force signal.

## 4. Simulation rig - `feature/sim-rig`

MuJoCo SO-101 model (`TheRobotStudio/SO-ARM100` → `Simulation/SO101`) plus a linear spring (F = k·x, k measurable from the PESOLA in minutes) and a scripted "patient finger" trajectory.

Purpose is **de-risking the controller without a human in the loop** - oscillation, force overshoot, behaviour when the force reading freezes or spikes. Not for generating training data.

Can run in parallel with 1–2.

## 5. Hand tracking / ROM - `feature/hand-tracking`

Finger range of motion during the session. MediaPipe Hand Landmarker on CPU is the safe default, done: `hand_tracking/` reports one flexion angle per finger, see [docs/hand-tracking.md](./docs/hand-tracking.md). WiLoR is still worth adding alongside it if the 4 GB RTX 3050 holds it in fp16 (verify with `nvidia-smi` before committing to it) - it outputs MANO joint rotations, which is the input format a musculoskeletal model would later want, something a bare per-joint angle does not give.

Independent of 1–4.

## 6. Session metrics - `feature/session-metrics`

Record force against finger displacement across the session and derive the joint stiffness curve. This is the clinical deliverable.

Depends on 3 and 5.

## 7. Remote clinician dashboard - `feature/remote-dashboard`

Live force, position, ROM and session metrics; the clinician sets the threshold and can stop the session.

Depends on 6.
