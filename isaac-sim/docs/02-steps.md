# Steps, in order

Every script named here is in this folder's parent (`isaac-sim/`). Scripts under `sim/` run on the
workstation; `cloud/nebius/sync.sh up` copies them there.

## A. The machine

| Step | Script | Result |
|---|---|---|
| Create the workstation | `cloud/nebius/create.sh` | VM `flexnsense-ws`, user `flex` |
| Install Isaac Sim and Isaac Lab | `cloud/nebius/setup_vm.sh` | `~/IsaacLab`, `source ~/isaac-env.sh` |
| Start, stop, status, copy files | `cloud/nebius/start.sh`, `stop.sh`, `status.sh`, `sync.sh` | |

## B. The arm

| Step | Script | Result |
|---|---|---|
| Check NormaCore's URDF | `tools/check_urdf.py` | 8 revolute joints, limits and masses as expected |
| Convert it for the simulator | `sim/01_import_elrobot.sh` | `usd/`: one simple collision shape per part |
| Convert it with finer collision shapes | `sim/01b_import_decomposed.sh` | `usd_decomposed/`: shapes that follow the parts |
| Repair the finer model's gripper joints | `sim/09_fix_gripper_usd.py` | `elrobot_follower_fixed.usda` (see 04-problems) |
| Move every joint, render | `sim/02_move_joints.py` | The arm imports and moves |

## C. The bench, scripted (no learning)

| Step | Script | Result |
|---|---|---|
| First force loop | `sim/03_yield_loop.py` | Holding still: 4.47 N peak. Giving way: about 2.1 N, against a 1.96 N limit |
| Bench layout, one arm | `sim/05_bench_scene.py` | Gripper low and level, jaws on the scale's ring |
| Two arms, drawn scale | `sim/07_two_arms.py` | Pull arm and move arm, three camera views |
| Physical scale, real grip | `sim/08_physical_scale.py` | The tube is a body with collisions; the move arm pinches and carries it by friction |

Videos of these: `videos/01` to `videos/08`.

## D. Training (learning)

| Step | Script | Result |
|---|---|---|
| The pull task | `sim/rl/flex_tasks/pull_env.py`, `pull_env_cfg.py` | Task id `Flex-Pull-v0` |
| The move task | `sim/rl/flex_tasks/move_env.py`, `move_env_cfg.py` | Task id `Flex-Move-v0` |
| Learning settings | `sim/rl/flex_tasks/agents.py` | PPO, network sizes, learning rate |
| Start a training | `sim/rl/train.sh <task>` | Log `~/train_<task>.log`, checkpoints in `~/IsaacLab/logs/rsl_rl/` |
| Stop a training | `sim/rl/stop.sh` | |
| Measure a pull policy | `sim/rl/eval_pull.py --checkpoint <file>` | `summary.json`, `trace.csv`, exported `policy.onnx` |
| Film a pull policy | `sim/rl/eval_pull.py --video`, `sim/rl/film_training.sh <run>` | Videos 09 and 10 |
| Measure / film a move policy | `sim/rl/eval_move.py` | |
| Upload to Hugging Face | `sim/rl/push_hf.py` | Private model repo |

## E. Towards the robot

| Step | File | State |
|---|---|---|
| The pull task on the full bench, at the real loop's pace | `sim/rl/flex_tasks/pull_env_cfg.py` (`PullBenchEnvCfg`), `sim/10_make_hand_usd.py` | Task id `Flex-PullBench-v0`, policy `policies/pull_bench_v1` |
| Runner for the team's control loop | `policies/sim_pull_runner.py`, `policies/run_pull_local.py` | Written and unit-tested; never run on hardware |
| Test of the whole chain without a robot | `policies/selftest_pull.py` | Passes (average peak 1.39 N against 3.23 N held still) |
| Guide | `policies/README.md` | |
