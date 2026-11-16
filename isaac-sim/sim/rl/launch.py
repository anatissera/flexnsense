"""Train or play the Flex'N Sense tasks with Isaac Lab's own command line.

    source ~/isaac-env.sh && cd ~/IsaacLab
    PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/launch.py train \\
        --rl_library rsl_rl --task Flex-Pull-v0 --num_envs 4096 --headless
    PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/launch.py play \\
        --rl_library rsl_rl --task Flex-Pull-v0 --num_envs 16 --checkpoint <model.pt>
"""

import sys

import flex_tasks  # noqa: F401  (registers the task ids)

from isaaclab_rl.entrypoints import run_play_cli, run_train_cli

if len(sys.argv) < 2 or sys.argv[1] not in ("play", "train"):
    sys.exit("usage: launch.py play|train <isaaclab play/train arguments>")
sys.exit((run_play_cli if sys.argv[1] == "play" else run_train_cli)(sys.argv[2:]))
