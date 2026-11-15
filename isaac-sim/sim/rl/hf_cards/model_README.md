---
library_name: rsl_rl
pipeline_tag: reinforcement-learning
tags:
  - robotics
  - simulation
  - isaac-lab
  - ppo
  - elrobot
  - rehabilitation
---

# Flex'N Sense: two arm policies trained in simulation (ElRobot, Isaac Lab)

**Made for a hackathon only.** These policies were trained by team Flex'N Sense during the Leukerbad Robotics
Hackathon (November 2026) as a demonstration. They were trained and measured in simulation only. They have not been
run on a real robot, they are not a medical device, and they must not be used to control a robot near a person.

## The two policies

| Folder | Task | What goes in | What comes out |
|---|---|---|---|
| `policies/pull_v1` | Hold the top of a spring scale; give way when the finger pulls harder than 1.96 N (200 gf); come back when the pull stops | 29 numbers: arm joints (angle, speed), distance to targets, the force now and one step ago, the gripper's place relative to its start, the last action | 5 numbers: changes to the targets of motors 1, 2, 3, 4, 6 |
| `policies/move_v1` | Take the scale near its loop, lift it over a finger, set the loop on the next finger, let go | 44 numbers: arm joints, gripper command, where the gripper is and points, the grasp point, the loop and the target finger, held / placed flags, the last action | 6 numbers: the same five motors, plus open / close |

Each folder has `policy.onnx` (with `policy.onnx.data`), `policy.pt` (TorchScript), `checkpoint.pt` (to continue
training), `params/` (the exact settings) and `summary.json` (the measurement).

## How they were trained

- **Arm:** NormaCore's ElRobot, simulated from their own URDF model (commit `6b494f7`, Apache-2.0).
- **Simulator:** NVIDIA Isaac Sim 6.2 with Isaac Lab `main@727e4e43a`, 120 physics steps a second, 60 decisions a second.
- **Method:** reinforcement learning, PPO from `rsl_rl` 5.5.1, 4,096 simulated arms in parallel on one NVIDIA RTX PRO
  6000. No recording from a real arm was used.
- **Pull:** 600 iterations, 59 million steps, 7.7 minutes. The scale is a spring whose stiffness, and the finger's
  pull (distance, speed, hold time, direction), vary per episode.
- **Move:** six runs, about 470 million steps, about 80 minutes. The first five runs each exposed a flaw in the reward
  or in the task's rules; they were fixed before the final run. The grip in this task follows rules, not contact
  physics.

## Measured in simulation (1,024 arms each)

| Pull arm | Trained policy | Arm held still |
|---|---|---|
| Peak force, average | 1.60 N | 3.61 N |
| Peak force, worst pull | 2.52 N | 7.32 N |
| Pulls more than 10 % over the 1.96 N limit | 1.1 % | 85 % |

| Move arm | |
|---|---|
| Loop placed within 8 mm of the target finger | 100 % |
| Dropped / hit a finger | 0 % / 0 % |
| Time to place, average | 0.40 s (full motor speed: too fast for use next to a hand) |

## What is assumed

Servo stiffness and damping (40 N·m/rad, 2 N·m·s/rad), the spring (5.886 N over 0.10 m) and the gearbox's apparent
mass on the jaws are assumptions, not measurements. The force the pull policy reads is the true simulated force plus
noise; on a robot it would have to come from a camera reading of the scale. The move policy is told where the scale
and the fingers are.

## Files

`policies/` the two policies. `media/` videos of the training, including a film that goes from 1 arm to 1,024.
`code/` the Isaac Lab tasks and scripts that produced them. Recorded episodes of both policies are in the companion
dataset repo.

## Credits

The arm model is NormaCore's (Apache-2.0). The simulation uses NVIDIA Isaac Sim and Isaac Lab, and `rsl_rl` from the
Robotic Systems Lab at ETH Zurich.
