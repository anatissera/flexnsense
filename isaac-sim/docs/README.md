# Process: how the simulation and the training were done

Written 2026-11-15. This folder explains the whole route, from "an arm model on a cloud machine" to "a trained policy",
in the order it happened. Everything is simulation; no real arm was moved by any of it.

| File | What it answers |
|---|---|
| [01-technology.md](01-technology.md) | Which tools were used, what each one is for, and what comes from the real robot |
| [02-steps.md](02-steps.md) | What was done, step by step, with the script for each step |
| [03-training.md](03-training.md) | How the training works: what the policy sees and does, the rewards, the numbers, what failed |
| [04-problems-and-fixes.md](04-problems-and-fixes.md) | The things that went wrong and how each was found |
| [05-files.md](05-files.md) | Every folder and script, and what it is for |
| [06-limits-and-next.md](06-limits-and-next.md) | What is assumed, what is not done, what to do next |

How to get the policies, upload them and put one on the robot is in [../policies/README.md](../policies/README.md).

## The short version

1. A GPU workstation was rented at Nebius and NVIDIA Isaac Sim + Isaac Lab were installed on it.
2. NormaCore's own 3D model of the ElRobot (their URDF file and meshes) was converted for the simulator.
3. A scene like the bench was built: two follower arms, the table, the spring scale, the hand.
4. The scale's tube was made a physical object, and the move arm was made to really pinch and carry it.
5. Two training tasks were written, one per arm, and trained with 4,096 simulated arms at once.
6. Both policies work in simulation, each measured on 1,024 arms. The move arm took six attempts; its grip is a rule there, not physics.
