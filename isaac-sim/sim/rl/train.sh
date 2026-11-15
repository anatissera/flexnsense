#!/usr/bin/env bash
# Train a Flex'N Sense task in the background on the workstation.  bash ~/flexnsense_sim/rl/train.sh Flex-Pull-v0 [extra args]
# Log: ~/train_<task>.log   Checkpoints: ~/IsaacLab/logs/rsl_rl/<experiment>/<date>/model_*.pt
source ~/isaac-env.sh; cd ~/IsaacLab
task=$1; shift
PYTHONPATH=~/flexnsense_sim/rl nohup uv run --extra isaacsim python ~/flexnsense_sim/rl/launch.py train \
  --rl_library rsl_rl --task "$task" --num_envs 4096 "$@" > ~/"train_$task.log" 2>&1 &
echo "started $task (pid $!)"
