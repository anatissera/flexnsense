---
pretty_name: Flex'N Sense simulated episodes (ElRobot, Isaac Lab)
task_categories:
  - robotics
tags:
  - simulation
  - isaac-lab
  - reinforcement-learning
  - elrobot
  - rehabilitation
size_categories:
  - 100K<n<1M
configs:
  - config_name: pull_arm
    data_files: data/pull_arm.parquet
  - config_name: move_arm
    data_files: data/move_arm.parquet
---

# Flex'N Sense: simulated episodes of two trained arm policies

**Made for a hackathon only.** This dataset was produced by team Flex'N Sense during the Leukerbad Robotics Hackathon
(November 2026) as a demonstration. It is simulation data from policies trained in a few minutes to hours. It is not
validated on a real robot, it is not a medical device or clinical data, and it must not be used to control a robot
near a person.

## What this is

1,024 episodes of two policies acting **in simulation** (NVIDIA Isaac Lab), 512 episodes each, 20 rows a second:

| Config | Episodes | Seconds each | What the arm does |
|---|---|---|---|
| `pull_arm` | 512 | 10 | Holds the top of a spring scale. A finger pulls on the other end. When the pull exceeds 1.96 N (200 gf) the arm gives way; when the pull stops it comes back |
| `move_arm` | 512 | 8 (shorter if the episode failed) | Takes the scale near its loop, lifts it over a finger, sets the loop on the neighbouring finger, lets go |

The arm is the NormaCore ElRobot (7 arm motors and a gripper), simulated from NormaCore's own URDF model.

There are **no recordings of a real arm and no camera images** in this dataset.

## How it was made

1. **The arm.** NormaCore's `elrobot_follower.urdf` and meshes (their public repository, commit `6b494f7`, Apache-2.0)
   were converted for NVIDIA Isaac Sim 6.2. Link lengths, joint limits, masses and the motor limits (2.94 N·m,
   4.71 rad/s) come from that file. Servo stiffness and damping are assumed (40 N·m/rad, 2 N·m·s/rad).
2. **The tasks.** Two reinforcement-learning tasks were written in Isaac Lab (`main@727e4e43a`):
   - *Pull.* The scale is a spring (5.886 N over an assumed 0.10 m, varied by 30 % per episode) between the gripper and
     a finger point that moves away, holds and returns. Pull distance, speed, hold time, direction and the arm's
     start posture vary per episode. The reward penalises force above 1.96 N, drifting from its place while the force
     is low, and moving off the scale's line.
   - *Move.* The scale's ring end is fixed where the other arm holds it. The grip follows rules, not contact physics:
     jaws closing within 15 mm of the grasp point take the tube, the loop then follows the gripper, opening within
     8 mm of the target finger places it, opening elsewhere drops it, and a loop carried lower than 2 cm between two
     fingers counts as hitting a finger.
3. **The training.** PPO from the `rsl_rl` library (5.5.1), 4,096 simulated arms in parallel on one NVIDIA RTX PRO
   6000. Policy: a small network (three layers), numbers in, joint-target changes out, 60 decisions a second.
   - Pull: 600 iterations, 59 million simulated steps, 7.7 minutes.
   - Move: six runs, about 470 million steps, about 80 minutes. The first five runs each exposed a flaw in the reward
     or the rules (for example the policy learned to swipe the loop straight through a finger, and later to grab the
     tube off-centre so that a flaw in the rule moved the loop for it). Those were fixed before the final run.
4. **These episodes.** The two final policies were run once on 512 fresh simulated arms each (seed 2026), and the
   state was written 20 times a second.

Measured in simulation on 1,024 arms each: the pull policy keeps the average peak force at 1.60 N where a rigid arm
gives 3.61 N, with 1.1 % of pulls more than 10 % over the limit (85 % for the rigid arm). The move policy places the
loop within 8 mm in 100 % of episodes, with none dropped and none hitting a finger. It does so in about 0.4 s, at the
full motor speed the model allows, which is far too fast for use next to a hand.

## Columns

One row per frame. The first six columns follow NormaCore's Parquet episode layout
(`docs/datasets/export-parquet/datasets-export.md` in their repository), so tools written for their recordings can
read the joint data.

| Column | Type | Meaning |
|---|---|---|
| `episode_start_ns` | uint64 | The episode's id (a made-up clock time; one value per episode) |
| `global_frame_id` | binary | Frame counter, 8 bytes, little-endian |
| `timestamp_ns_since_episode_start` | uint64 | Time within the episode, steps of 50 ms |
| `joints` | list of 8 structs | Motors 1 to 8: `range_min`, `range_max`, `position`, `goal` (servo ticks), `position_norm`, `goal_norm` (0 to 1 within the range), `current_ma`, `velocity` |
| `images` | list | **Empty.** No camera was rendered |
| `task` | string | The task in words |
| `sim_*` | float32 | The simulator's own signals (below) |

`pull_arm` extras: `sim_force_n` (true spring force), `sim_finger_pull_m` (how far the finger has pulled),
`sim_gripper_along_m`, `sim_gripper_side_m`, `sim_gripper_up_m` (the gripper relative to where it started).

`move_arm` extras: `sim_loop_x_m`, `sim_loop_y_m`, `sim_loop_z_m`, `sim_target_x_m`, `sim_target_y_m`,
`sim_target_z_m` (in the bench's frame), `sim_held` (1 while the tube is held), `sim_placed` (1 once the loop is on the
target finger).

**What is synthetic in the joint columns**, because there is no real servo here: `position`, `goal`, `range_min` and
`range_max` are ticks computed from the simulated angle (4,096 per turn, 2,048 at the URDF's zero angle), not ticks
from a calibrated arm. `current_ma` and `velocity` are always 0. Motor 8 (the gripper) is computed from the jaw
opening; in the pull episodes the jaws stay closed.

## Load it

```python
from datasets import load_dataset
pull = load_dataset("<this repo>", "pull_arm", split="train")
print(pull[0]["joints"][0], pull[0]["sim_force_n"])
```

## Limits

- Simulation only, with assumed servo and spring values. Nothing here was checked against the real arm.
- No images, so it cannot train a camera-based policy as it is.
- The move episodes rely on a rule-based grip and on knowing where the scale and fingers are.

## Credits and licence

The arm model is NormaCore's (Apache-2.0). The simulation uses NVIDIA Isaac Sim and Isaac Lab, and `rsl_rl` from the
Robotic Systems Lab at ETH Zurich. The episodes were generated by team Flex'N Sense for the Leukerbad Robotics
Hackathon 2026.
