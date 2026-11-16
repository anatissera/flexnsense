# Technology: what was used and why

## The pieces

| Piece | What it is | What it did here |
|---|---|---|
| **Nebius cloud workstation** | A rented computer with an NVIDIA RTX PRO 6000 graphics card (96 GB), in eu-south1. About $1.80 an hour while running | Runs the simulator and the training. This Mac cannot run Isaac Sim |
| **NVIDIA Isaac Sim 6.2** | A robot simulator. It contains PhysX (the physics engine: gravity, joints, motors, collisions, friction) and RTX (the renderer that makes the pictures) | Simulates the arms and the scale, and renders the videos |
| **NVIDIA Isaac Lab** (version `main@727e4e43a`) | A framework on top of Isaac Sim for robot learning. Its key trick: thousands of copies of the same scene simulated together on the graphics card | The scenes, the training tasks, 4,096 arms in parallel |
| **rsl_rl 5.5** | A reinforcement-learning library from ETH Zurich's Robotic Systems Lab, the one Isaac Lab uses by default | The learning algorithm (PPO) |
| **PyTorch 2.11** | The neural-network library underneath | The policy network |
| **NormaCore ElRobot model** | The robot maker's own description of the arm: `elrobot_follower.urdf` plus 19 mesh files, from their public repo (commit `6b494f7`, Apache-2.0 licence) | The simulated arm is built from it: link lengths, joint axes, joint limits, masses, motor effort and speed limits |
| **ONNX** | A portable file format for a trained network | The exported policy (`policy.onnx`) can run on any computer without Isaac |

## "Did you use technology from the robot?"

Yes, in one precise sense: **the simulated arm is NormaCore's own model of the ElRobot.** The geometry, the 8 motors
(7 arm joints and the gripper), the joint limits, the masses, and the motor limits (2.94 N·m, 4.71 rad/s) all come from
their URDF. The gripper's finger sizes were checked against their meshes and their gripper sheet.

What was **not** taken from the real robot:

- **No recorded data from the real arm was used for this training.** The team's 43 real recordings (NormaCore Parquet
  episodes) trained the two earlier imitation policies (ACT and SmolVLA). The two policies here learned only from
  simulated trial and error.
- **Servo behaviour is assumed.** How stiffly an ST3215 holds its position was not measured; the simulation uses
  40 N·m/rad and 2 N·m·s/rad.
- **The spring scale's stiffness is assumed** (5.886 N over 0.10 m).

## Two kinds of robot learning, and which one this is

| | Imitation learning | Reinforcement learning (used here) |
|---|---|---|
| Learns from | Recordings of a person doing the task | Its own trial and error, scored by a reward |
| Needs | Demonstrations (the team recorded 43) | A simulator and a reward formula |
| The team's earlier policies | ACT, SmolVLA (camera images in, joint targets out) | |
| The policies in this folder | | Pull arm, move arm (numbers in, joint targets out) |

Reinforcement learning was chosen here because the pull task is about **force**, and the simulator knows the force
exactly at every instant, for thousands of different pulls. Demonstrations cannot give that.

## The episode format (for collecting data later)

If simulated episodes are recorded for imitation learning, they must use NormaCore's Parquet layout, which is what
their trainer reads. One row per frame, six columns: `episode_start_ns`, `global_frame_id`,
`timestamp_ns_since_episode_start`, `joints` (per motor: `range_min`, `range_max`, `position`, `position_norm`, `goal`,
`goal_norm`, `current_ma`, `velocity`), `images` (one JPEG per camera), `task` (text). Their trainer uses only
`position_norm`, `goal_norm`, the images and the task text. Source: `docs/datasets/export-parquet/datasets-export.md`
in NormaCore's repo. No such recorder has been written yet.
