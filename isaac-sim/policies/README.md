# Simulation-trained policies: what they are, how to get them, how to put them on the robot

Written 2026-11-15. Everything here was trained and measured **in simulation only** (Isaac Lab on the Nebius
workstation). Nothing in this folder has moved the real arm yet.

## 1. What was trained

Two policies, one per follower arm, each trained with 4,096 simulated arms in parallel (PPO, `rsl_rl`).

| | Pull arm (`Flex-Pull-v0`) | Move arm (`Flex-Move-v0`) |
|---|---|---|
| Job | Hold the scale's ring. When the finger pulls harder than 1.96 N (200 gf), give way; come back when the pull stops | Take the scale near its loop, lift it over the finger, set the loop on the next finger, let go |
| Sees | 29 numbers: arm joints (angle, speed), distance to its targets, the scale's force now and one step ago, where the gripper is relative to its start, its last action | 44 numbers: arm joints, gripper command, where the gripper is and points, where the grasp point, the loop and the target finger are, held / placed flags, last action |
| Does | Changes the targets of motors 1, 2, 3, 4, 6 (60 times a second) | Same five motors, plus open / close |
| Training time | 7.7 minutes (600 iterations, 59 million steps) | About 80 minutes over six runs (see `move_v1/NOTES.md`) |
| Code | `sim/rl/flex_tasks/pull_env.py`, `pull_env_cfg.py` | `sim/rl/flex_tasks/move_env.py`, `move_env_cfg.py` |

Motors 5 and 7 (the two roll motors) are held still in both, as in the team's real demonstrations.

### Pull arm: measured result

1,024 simulated pulls, the same pulls with and without the policy (`pull_v1/summary.json`):

| | Trained policy | Arm held still |
|---|---|---|
| Peak force, average | 1.60 N (163 gf) | 3.61 N (368 gf) |
| Peak force, worst pull | 2.52 N (257 gf) | 7.32 N (747 gf) |
| Pulls that went more than 10 % over the limit | 1.1 % | 85 % |
| Time above 110 % of the limit, average | 0.003 s | 3.1 s |
| How far the arm gives way, average | 44 mm | 5 mm |
| Drift from its place while nobody pulls | 4 mm | 5 mm |

Videos (in `../videos/`): `09_pull_policy_trained_vs_hold.mp4` (policy, then the same pull with the arm held still),
`10_pull_policy_training_progress.mp4` (the policy after 0, 100, 200 and 599 training iterations).

### What is simplified in the simulation (read before trusting the numbers)

- **The scale is a spring, not the PESOLA.** 5.886 N over an assumed 0.10 m of travel, varied by 30 % per pull. The real
  spring constant has not been measured.
- **Servo stiffness and damping are assumed** (40 N·m/rad, 2 N·m·s/rad). They were not measured on the ST3215.
- **The force the pull policy sees is the true spring force plus noise, one step late.** On the robot it must come
  from the camera reading of the scale (`gauge_reader`), which is slower and can drop out.
- **The move task uses rules for the grip**, not physics: jaws closing within 15 mm of the grasp point take the tube,
  letting go within 8 mm of the target finger places the loop, a low loop between two fingers counts as hitting a
  finger. The physical pinch is shown separately in `sim/08_physical_scale.py` (videos 07 and 08).
- **The move policy is told where the scale, the loop and the fingers are.** On the robot those positions would have
  to come from a camera. That part does not exist yet.

## 2. How to get the files (pull)

### What the download puts in this folder

```
policies/pull_bench_v1/                              the policy prepared for the robot (in git, 130 kB): policy.pt,
                                                     policy.onnx, flex_sim.json, settings, measurement
policies/pull_v1/policy.onnx (+ policy.onnx.data)   the first pull policy (60 decisions a second), for onnxruntime
policies/pull_v1/policy.pt                           the same, TorchScript
policies/pull_v1/2026-11-15_19-02-34/model_599.pt    the training checkpoint (to continue training)
policies/pull_v1/2026-11-15_19-02-34/params/         the exact training settings (env.yaml, agent.yaml)
policies/pull_v1/summary.json, trace.csv             the measurement above
policies/move_v1/...                                 the same for the move arm
```

### From Hugging Face

```bash
pip install -U huggingface_hub
hf download peresjav/flexnsense-sim-policies --local-dir flexnsense-sim-policies
```

The recorded simulation episodes are a separate dataset repo:

```bash
hf download peresjav/lerobot-2026 --repo-type dataset --local-dir lerobot-2026
```

Both repos are public.

### The team's real recordings and earlier policies (for reference)

```bash
hf download JustoDario/flexnsense_skill1 --repo-type dataset --local-dir data/flexnsense_skill1   # gated: needs access
hf download JustoDario/flexnsense_act_skill1_rightindex2middle --local-dir policies_real/act
hf download JustoDario/flexnsense_smolvla --local-dir policies_real/smolvla
```

Those two policies are imitation policies trained on the real recordings (NormaCore Parquet episodes). They are
loaded with the team's own scripts (branch `feature/robot-training`). They are a different kind of policy from the two here.

## 3. How to put the pull policy on the robot

The steps are in the folder's main [README](../README.md), section "Test the pull policy with
`robot_training/inference`". In short: the policy to use is `pull_bench_v1` in this folder; `sim_pull_runner.py`
turns it into a runner for the control loop of branch `feature/robot-training`; `run_pull_local.py` starts that loop
with it; `selftest_pull.py` tests the whole chain without a robot; `check_directions.py` prints what each motor
should do, to check the motor directions by hand. `run_pull_on_robot.py` is the first bridge script (for `pull_v1`, straight to
Station); it is kept as it was for whoever already uses it.

## 4. How to upload to Hugging Face (push), from the cloud

The upload runs on the workstation, so it does not use this Mac's connection. The token stays in a file there.

```bash
# 1. you, once: put your token in a file on the workstation (type it yourself; it is not stored anywhere else)
ssh flex@<workstation ip> 'umask 077; cat > ~/.hf_token'      # paste the token, Enter, then Ctrl-D

# 2. upload (creates two PUBLIC repos: the model repo peresjav/flexnsense-sim-policies and the dataset repo peresjav/lerobot-2026)
ssh flex@<workstation ip> '~/pq/bin/python ~/flexnsense_sim/rl/push_hf.py'

# 3. afterwards, remove the token from the workstation
ssh flex@<workstation ip> 'rm ~/.hf_token'
```

What gets uploaded, from `~/flexnsense-sim/deliver/` on the workstation: to the model repo, both policies (ONNX,
TorchScript, checkpoint, settings), their measurements, the task code and the videos; to the dataset repo, the recorded
simulation episodes as Parquet with their description.

## 5. How to train again

```bash
bash cloud/nebius/start.sh && bash cloud/nebius/sync.sh up      # the workstation bills about $1.80 an hour while it runs
ssh flex@<ip> 'bash ~/flexnsense_sim/rl/train.sh Flex-Pull-v0'  # log: ~/train_Flex-Pull-v0.log
ssh flex@<ip> 'bash ~/flexnsense_sim/rl/stop.sh'                # run this alone, not in a line with other commands
# measure and film (on the workstation):
#   PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/eval_pull.py --checkpoint <model.pt> [--video]
bash cloud/nebius/stop.sh
```

Checkpoints land in `~/IsaacLab/logs/rsl_rl/flex_pull/<date>/` and `flex_move/<date>/` on the workstation.
