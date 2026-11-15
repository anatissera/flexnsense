# Flex'N Sense in simulation: training the two arms in NVIDIA Isaac Lab

This folder holds everything we used to simulate the Flex'N Sense bench and to train one policy per arm with
thousands of simulated arms at once: the code, the settings, and the notes on how it was done. The trained policies
and the videos are on Hugging Face (links below).

**Everything here is simulation.** None of these policies has moved the real arm yet. What is assumed, what works and
what does not is stated plainly below and in [docs/06-limits-and-next.md](docs/06-limits-and-next.md).

## Status: read this first

- **Nothing here has run on the real arm.** The policies were trained and measured in simulation only.
- **Test the pull policy.** It is the one worth trying on the robot, carefully, with the dry run first (below).
- **The move policy is known to be wrong for real use.** In its training task the scale is not a physical object: the
  gripper never closes around it, the loop follows the gripper by a rule, the whole move takes about 0.4 s, and it
  starts already hovering over the scale. It is being redone on the physical scene (`sim/08_physical_scale.py`), starting
  from the arm's resting posture. Do not run it on the robot.

## Get the policies and the recorded episodes

They are on Hugging Face, not in git:

- Policies: https://huggingface.co/peresjav/flexnsense-sim-policies (ONNX, TorchScript, checkpoints, videos)
- Recorded simulation episodes: https://huggingface.co/datasets/peresjav/lerobot-2026 (Parquet, no camera images)

```bash
pip install -U huggingface_hub
# from the repository root: puts the files where this folder's scripts expect them (isaac-sim/policies/pull_v1/...)
hf download peresjav/flexnsense-sim-policies --include "policies/*" --local-dir isaac-sim
# optional: the videos, and the recorded episodes
hf download peresjav/flexnsense-sim-policies --include "media/*" --local-dir isaac-sim
hf download peresjav/lerobot-2026 --repo-type dataset --local-dir isaac-sim/dataset
```

Both repos are public, so no login is needed. The videos include two films that go from 1 arm to 1,024.

## Try the pull policy on the real arm

The policy is a small network: 29 numbers in, 5 numbers out, 60 times a second. `policies/run_pull_on_robot.py` is the
bridge to Station. **It has never been run on hardware.** Go through these steps in order, with one person at the
keyboard and a hand on the power.

1. **Measure the joint mapping.** The policy thinks in radians from the URDF's zero pose (upright column, level
   forearm); the servos think in ticks (4,096 per turn). For motors 1 to 7, read the tick value at that pose from
   Station and note the sign (negative where the motor turns the other way from the URDF axis). Write
   `isaac-sim/policies/joint_map.json`:

   ```json
   {"1": {"zero_tick": 2048, "ticks_per_rad": 651.9}, "2": {"zero_tick": 2048, "ticks_per_rad": -651.9}}
   ```

   651.9 is 4096 / (2 pi). The numbers above are placeholders, not measurements: check every motor by moving it by
   hand and watching the angle the script prints.
2. **Start Station** (see `SETUP.md`). Leave its current limits in `station.yaml` as they are.
3. **Start the scale reading:** `.venv/bin/python -m gauge_reader.server` (WebSocket `/force`, port 8766). This opens
   the camera.
4. **Put the arm in the start posture** used in training (motors 1 to 7, radians): 0, 0.04, 1.057, 0.072, 0, -1.169, 0,
   jaws closed on the scale's ring, the scale level and in line with the arm.
5. **Dry run. Nothing is sent to the arm:**

   ```bash
   pip install onnxruntime websockets
   .venv/bin/python isaac-sim/policies/run_pull_on_robot.py \
       --policy isaac-sim/policies/pull_v1/policy.onnx \
       --joint_map isaac-sim/policies/joint_map.json --force_url ws://localhost:8766/force
   ```

   Pull on the scale by hand. "force" must follow the scale, "along" must be about zero at rest, and the actions must
   stay small while the force is low and point the same way each time the force passes 1.96 N (200 gf).
6. **Only then add `--send`.** The script stops if the scale reading is missing for 0.25 s, limits every joint to
   0.012 rad per step, and stops if a target strays 0.9 rad from the start posture. Station's own current limits stay
   in force underneath.

What will most likely differ on the real arm: the servos are stiffer and slower than the assumed ones, the camera
reading arrives at about 30 Hz with a delay, and the real spring constant is not the assumed one. If the dry run looks
wrong, that is the first place to look: measure those three, put them in `sim/rl/flex_tasks/pull_env_cfg.py`, and
retrain (8 minutes).

## Results

### Pull arm (`Flex-Pull-v0`)

Trained in 7.7 minutes (600 iterations, 59 million simulated steps). Measured on 1,024 pulls it had not seen, the same
pulls with and without the policy:

| | Trained policy | Arm held still |
|---|---|---|
| Peak force, average | 1.60 N (163 gf) | 3.61 N (368 gf) |
| Peak force, worst pull | 2.52 N (257 gf) | 7.32 N (747 gf) |
| Pulls more than 10 % over the 1.96 N limit | 1.1 % | 85 % |
| Time above 110 % of the limit, average | 0.003 s | 3.1 s |
| How far the arm gives way, average | 44 mm | 5 mm |

### Move arm (`Flex-Move-v0`)

Six training runs, about 80 minutes and 470 million simulated steps in total; the first five each exposed a flaw in
the reward or in the task's rules (the record is in [docs/03-training.md](docs/03-training.md)). Measured on 1,024 arms
at the real placing tolerance of 8 mm:

| | |
|---|---|
| Loop placed on the target finger | 100 % |
| Dropped | 0 % |
| Hit a finger on the way | 0 % |
| Highest lift during the carry, average | 46 mm |
| Time from start to placed, average | 0.40 s |

Two cautions. The grip is a rule in this task, not physics (see the table below). And the policy is fast: it uses the
full motor speed the arm's model allows, which is not what you want next to a hand; a real version needs a speed limit.

## What is real and what is assumed

| Real | Assumed or simplified |
|---|---|
| The arm is NormaCore's own model of the ElRobot (their URDF and meshes, Apache-2.0): geometry, joint limits, masses, motor effort and speed limits | Servo stiffness and damping (40 N·m/rad, 2 N·m·s/rad): not measured |
| The force limit, 1.96 N (200 gf), is the one the project uses | The spring scale is a spring of 5.886 N over 0.10 m, varied by 30 % per pull: the real constant is not measured |
| The physical pinch of the scale by the jaws is simulated with contact and friction in `sim/08_physical_scale.py` | In the move training task the grip follows rules (close near the tube: taken; open near the target finger: placed), so that thousands of arms can learn quickly |
| | The policies read numbers, not camera images. On the robot the force would come from the camera reading of the scale, and the move policy would need the positions of the scale and the fingers |

No recording from the real arm was used for this training. The policies learned by trial and error in the simulator
(reinforcement learning, PPO).

## What is in this folder

```
README.md               this file
.env.example            the cloud settings to fill in (copy to .env, which is never committed)
docs/                   how it was done: technology, steps, training, problems and fixes, file map, limits
sim/                    scripts that run on the GPU workstation
  01 ... 09             import the arm, build the bench, the physical scale, the corrected arm model
  rl/                   the training: tasks, settings, start/stop, measuring, filming, upload
    flex_tasks/         Flex-Pull-v0 and Flex-Move-v0
tools/                  URDF check, forward kinematics, a chart script
cloud/nebius/           create / start / stop / copy files to the workstation
assets/elrobot/         NormaCore's ElRobot model (Apache-2.0, see LICENSE and SOURCE.txt there)
policies/               the bridge script for the real arm, the measurements and the training settings
                        (the policy files themselves are downloaded from Hugging Face, see above)
```

Start with [docs/README.md](docs/README.md).

## Run it yourself

You need a machine with an NVIDIA RTX graphics card and NVIDIA Isaac Lab. We rented one at Nebius; the scripts in
`cloud/nebius/` create it and install everything. It costs about $1.80 an hour while it runs.

```bash
cp .env.example .env            # fill in your Nebius project, network and subnet; accept the Omniverse licence
bash cloud/nebius/create.sh     # creates the workstation and installs Isaac Sim and Isaac Lab
bash cloud/nebius/sync.sh up    # copies sim/ and the arm model to it

# on the workstation
source ~/isaac-env.sh && cd ~/IsaacLab
bash ~/flexnsense_sim/01_import_elrobot.sh                 # NormaCore's URDF -> simulator model
bash ~/flexnsense_sim/rl/train.sh Flex-Pull-v0             # 4,096 arms, about 8 minutes
tail -f ~/train_Flex-Pull-v0.log

# measure it, film it
PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/eval_pull.py \
    --checkpoint ~/IsaacLab/logs/rsl_rl/flex_pull/<date>/model_599.pt
PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/film_movie.py \
    --before <model_0.pt> --after <model_599.pt>

bash cloud/nebius/stop.sh       # stop paying
```

Versions used: Isaac Sim 6.2, Isaac Lab `main@727e4e43a`, rsl_rl 5.5.1, PyTorch 2.11.

## The policies

`policies/README.md` has more detail on what each policy reads and writes. `policies/move_v1/NOTES.md` records where
the move policy ended and why it should not be used as it is.

## Licences

Our code in this folder follows the repository's licence. `assets/elrobot/` is NormaCore's work under Apache-2.0; its
licence file and source note are kept next to it, and the files are unmodified.
