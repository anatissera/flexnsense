# Flex'N Sense in simulation: training the two arms in NVIDIA Isaac Lab

This folder holds everything we used to simulate the Flex'N Sense bench and to train one policy per arm with
thousands of simulated arms at once: the code, the settings, and the notes on how it was done. The trained policies
and the videos are on Hugging Face (links below).

**Everything here is simulation.** None of these policies has moved the real arm yet. What is assumed, what works and
what does not is stated plainly below and in [docs/06-limits-and-next.md](docs/06-limits-and-next.md).

## Status: read this first

- **Nothing here has run on the real arm.** The policies were trained and measured in simulation only.
- **The policy to test is `policies/pull_bench_v1`** (the pull arm, trained on the full bench at the pace of the real
  control loop). It is in this folder, ready to run with `robot_training/inference`. Go through the steps below in
  order; the first three need no robot.
- **The move policy is known to be wrong for real use** and is not prepared for the robot: in its training task the
  scale is not a physical object, the whole move takes about 0.4 s, and it starts already hovering over the scale.
  For moving the scale between fingers, the policies in `robot_training/` (trained on real recordings) are the ones
  to use.

## Test the pull policy with `robot_training/inference`

The policy plugs into the control loop of branch `feature/robot-training` without changing it: it takes the eight
`position_norm` values of a Station frame and returns `goal_norm`, 10 times a second, and that loop's Station code
and tick-level safety (`GoalSafety`) do the rest. Station's camera pictures are not used; the force comes from
`gauge_reader` reading the scale's own camera inside the same process.

**What you need:** the environment of `robot_training/setup_env.sh`. It already has everything: torch, numpy,
OpenCV and norma-core's `station_py`. No extra package. Run from a checkout that has both this folder and
`robot_training/` (merge one branch into the other, or pass `--robot-training <path to robot_training/inference>`).
On Linux, `python -m gauge_reader cameras` lists the cameras once pull request #45 is merged; until then pass the
camera's number.

**If you already ran `run_pull_on_robot.py` with `pull_v1`:** that was the first policy (60 decisions a second, one
exact servo model, no margin). Use `pull_bench_v1` with `run_pull_local.py` instead. The first policy trained the
same way as `pull_bench_v1` but without the margin went over the force limit in the self-test below.

### 1. Unit tests (no robot)

```bash
.venv/bin/python -m pytest -q tests/isaac_sim
```

### 2. Self-test: the whole chain against a stand-in arm and scale (no robot, no camera, no Station)

```bash
source ~/robot_env/env.sh
python isaac-sim/policies/selftest_pull.py --policy isaac-sim/policies/pull_bench_v1 -v
```

It runs 30 pulls twice, with the policy and with the arm held still, through `GoalSafety`, and prints the peak forces
and `RESULT: OK` or `NOT OK`. Our run of it (Linux, 40 pulls): average peak force 1.39 N with the
policy against 3.23 N held still, worst pull 1.95 N, none more than 10 % over the 1.96 N limit, no safety aborts.
With the stand-in motors made faster or slower (`--lag 0.05` to `--lag 0.5`) the average stays between 1.24 and
1.40 N; the worst single pull rises to 2.77 N with the slowest motors. The stand-in arm is a simple model, not the
real one: a pass here means the code path is right, not that the real arm will behave the same.

### 3. Which way does each motor turn? (the one thing to check by hand)

The policy thinks in radians from the URDF's zero pose. A Station frame gives each joint as a fraction of its
calibrated range, and the follower's calibrated ranges match the URDF's joint limits (motor 6: 2,048 ticks is
3.1416 rad, the URDF's span exactly; the others within 1 %, motor 1 within 5 %). So no tick measurement is needed:

    angle = lower limit + position_norm x (upper limit - lower limit)

What is **not verified** is the direction: whether `position_norm` going up means the URDF angle going up, for each of
motors 1 to 7. The default assumes yes for all seven. To check, run

```bash
python isaac-sim/policies/check_directions.py
```

It prints, for each motor, where the model expects the jaws to go when that motor's `position_norm` rises a little
from the start posture (so many millimetres forward, left or up). With torque off, nudge each joint so that its
`position_norm` rises in Station's web page and compare.

If a motor goes the other way, write `isaac-sim/policies/pull_bench_v1/joint_directions.json` with -1 for it, for
example `[1, -1, 1, 1, 1, 1, 1]`. A quick sanity check that needs no table: put the arm in the start posture below and
run step 5 (dry run); the line it prints must show "along" near 0 mm and stay near 0 while nobody pulls. A wrong
direction shows up as "along" jumping as soon as the arm is touched.

The check was done against the calibration of bus `5B8E114979` (`config/calibration/`). `robot_training/inference`
defaults to bus `5B3D050113`: pass the right `--bus-serial`, and if that arm's ranges differ, the mapping needs
checking for it too.

### 4. Set up the bench

1. Station running, the follower with torque, the leader without (as in `robot_training/README.md`, section 4.1).
2. The scale's calibration on this machine: `config/scale_calib.json` (it is per machine and not in git; make it with
   `python -m gauge_reader calibrate`). The script opens the scale's camera itself (`--force-camera`).
3. The arm in the start posture used in training, jaws closed on the scale's ring, the scale level and in line with
   the arm. As `position_norm` for motors 1 to 7 (with all directions +1): 0.50, 0.51, 0.80, 0.52, 0.45, 0.07, 0.54.
   The policy tolerates about 0.1 rad around it; further away it has not been trained.

### 5. Dry run, then 30 ticks, then free

```bash
source ~/robot_env/env.sh
python isaac-sim/policies/run_pull_local.py --policy isaac-sim/policies/pull_bench_v1 \
    --force-camera <N> --bus-serial <FOLLOWER SERIAL> --camera-order side,gripper,table --dry-run
python isaac-sim/policies/run_pull_local.py --policy isaac-sim/policies/pull_bench_v1 \
    --force-camera <N> --bus-serial <FOLLOWER SERIAL> --camera-order side,gripper,table --max-ticks 30
```

`--force-camera` is the camera that looks at the scale. `--camera-order` only stops the loop from trying to recognise
Station's cameras, which the policy does not use; the loop still needs Station to publish its three pictures. Before
anything moves, the script loads the calibration, opens the scale camera and waits for a first reading; if any of that
fails it stops there and prints the real cause (missing calibration file, camera that will not open, scale not found
in the picture). In the dry run nothing is sent. Pull on the scale by hand and watch the line
printed every second: "force" must follow the scale, "along" must be about 0 mm at rest, the action must stay near 0
while the force is low and go the same way every time the force passes about 1.6 N (160 gf).

What protects the arm: `GoalSafety` from `robot_training` (goal change limited per tick, abort if a goal is far from
the position), a target that never runs more than 0.6 rad ahead of the arm, and a hard stop, with the reason printed,
when the scale reading is missing for 0.3 s. On that stop the arm **keeps holding** its last goal, as that loop does; it does not release. The
limits in `robot_training/README.md` section 4.0 apply here as well: do not use it with a patient's hand.

## Get the earlier policies, the videos and the recorded episodes

They are on Hugging Face, not in git:

- https://huggingface.co/peresjav/flexnsense-sim-policies (the first two policies, videos)
- https://huggingface.co/datasets/peresjav/lerobot-2026 (recorded simulation episodes, Parquet, no camera images)

```bash
pip install -U huggingface_hub
hf download peresjav/flexnsense-sim-policies --local-dir isaac-sim/hf
hf download peresjav/lerobot-2026 --repo-type dataset --local-dir isaac-sim/hf/dataset
```

`pull_bench_v1` is newer than that upload and lives in this folder (130 kB).

## Results

### Pull arm on the full bench (`Flex-PullBench-v0`): the one prepared for the robot

The scene is the bench as built in `sim/08_physical_scale.py`: NVIDIA's lab table from the Isaac asset library, the
arm clamped to it, the hand, the pulling finger and the scale's tube. The policy decides 10 times a second, as the
real control loop does, reads the force the way the camera gives it (steps of 5 gf, noise, 3 % of frames lost), and
is trained to stay 20 % under the limit with a different servo for every arm (stiffness 15 to 80 N·m/rad, damping
1 to 4, and a lag in following its goal). 600 iterations, 19 minutes. Measured on 1,024 benches it had not seen:

| | Trained policy | Arm held still |
|---|---|---|
| Peak force, average | 1.59 N (162 gf) | 3.39 N (345 gf) |
| Peak force, worst pull | 3.47 N (353 gf) | 7.17 N (731 gf) |
| Pulls more than 10 % over the 1.96 N limit | 2.5 % | 83 % |
| Time above 110 % of the limit, average | 0.009 s | 2.9 s |
| How far the arm gives way, average | 45 mm | 7 mm |

The worst pull (3.47 N) comes from the softest, slowest servos in the varied set. A first version without the margin
and the servo variation held 1.93 N in the simulator but went over the limit in the self-test (2.07 N on average,
48 % of pulls more than 10 % over), which is why it was retrained and why the self-test is step 2 above.

In this task the hand, the finger and the tube are bodies that are shown and placed each instant, not simulated by
contact: the scale's pull on the arm is the spring force along the tube, which is what a light tube held at both ends
transmits. Contact matters for the move task (pinching the tube), not for this one.

### Pull arm, first version (`Flex-Pull-v0`, 60 decisions a second, arm only)

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
policies/               pull_bench_v1 (the policy to test), the runner for robot_training's control loop
                        (sim_pull_runner.py, run_pull_local.py), the no-robot self-test (selftest_pull.py),
                        check_directions.py, and the measurements and settings of the earlier policies.
                        run_pull_on_robot.py is the first bridge script, kept for whoever already uses it
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
