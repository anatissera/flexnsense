# Scripts

Grouped by what they do to the rig, because that is the thing worth knowing before running one.

```
session/   run an exercise on a patient's finger
arm/       move or configure the arm, with nothing attached
measure/   produce a number this repo then depends on
preview/   render or inspect, nothing moves
lerobot/   NOT our hardware: parked SO-101 scripts, kept in case the project ever ports
```

## session/ moves an arm with a finger in the hook

| | |
|---|---|
| `run_session.py` | the admittance controller: the patient pulls, the arm yields at the cap |
| `run_cpm.py` | continuous passive motion: the arm walks the finger between two saved poses |
| `try_remote_hook.py` | one hook request to the other machine's actuator, over the socket |

Every one of these asks for a typed confirmation first and prints the gate's limits before
anything moves.

## arm/ moves the arm with nothing attached

| | |
|---|---|
| `go_home.py` | walk to one of the two named poses, in paced steps |
| `set_home_pose.py` | save the pose the arm is in now as `home` or `cpm` |
| `find_home_pose.py` | search the workspace for the pose that pulls straightest |
| `jog_coordinated.py` | walk the gripper along the tether's line and back, three joints together |
| `jog_one_motor.py` | nudge exactly one motor by a small tick delta, then put it back |
| `lock_arm.py` | lock every joint where it stands, or release them |
| `which_motor.py` | which joints move the gripper along the tether, from the URDF and the live pose |
| `calibrate.py` | check Station's connection and calibration state for a bus |
| `restore_calibration.py` | write a saved joint-range calibration back to the servos |

## measure/ produces numbers the rest of the repo depends on

The procedures are in [docs/bench-measurements.md](../docs/bench-measurements.md). The outputs
land in `config/calibration/` and are committed, because they describe the shared rig rather
than one laptop.

| | |
|---|---|
| `measure_ticks_per_metre.py` | ticks per metre and the yield sign, with a ruler against the arm |
| `measure_ticks_with_camera.py` | the same, using the camera instead of a ruler |
| `measure_tether_stiffness.py` | the tether's stiffness, with the PESOLA and a ruler. No arm |
| `measure_tip_excursion.py` | how far each fingertip travels, from a hand-tracking recording |
| `log_force_calibration.py` | servo current against known applied force. The experiment that came back negative |
| `check_camera_vs_station.py` | does opening the camera stall Station's position stream |

## preview/ renders, and moves nothing

| | |
|---|---|
| `simulate_pull.py` | one joint's arc against a coordinated straight pull, side by side |
| `simulate_session_video.py` | the controller responding to a pull, as a video, from the real control code |
| `flexnsense_viewer.py` | browse the teleoperation dataset arm 2 was trained on |

## A note on imports

`scripts/` is not a package, so these do not import each other. Shared code lives in `lib/`,
`control/` and `session/`; a script that needs another script's logic is a sign that logic
belongs in one of those three instead.
