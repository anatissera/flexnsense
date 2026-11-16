"""Flex'N Sense training tasks. String entry points only, so nothing heavy is imported before the simulator starts.

  Flex-Pull-v0   the pull arm: hold the scale's ring; when the finger pulls harder than the limit, give way.
  Flex-Pull10-v0 the same pull task at 10 decisions a second, the pace of the team's real control loop.
  Flex-PullBench-v0  the pull task on the full bench: NVIDIA's table, the hand, the scale, a camera-like force reading.
  Flex-Team-v0   both arms on one wooden table, both learning at the same time, each its own policy (two agents).
  Flex-Duo-v0    both arms on one wooden table: the pull arm runs its policy, the move arm learns next to it.
  Flex-Move-v0   the move arm: take the scale near its loop and move the loop from one finger to the next.
"""

import gymnasium as gym

gym.register(
    id="Flex-Pull-v0",
    entry_point="flex_tasks.pull_env:PullEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "flex_tasks.pull_env_cfg:PullEnvCfg",
        "rsl_rl_cfg_entry_point": "flex_tasks.agents:PullPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Flex-Move-v0",
    entry_point="flex_tasks.move_env:MoveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "flex_tasks.move_env_cfg:MoveEnvCfg",
        "rsl_rl_cfg_entry_point": "flex_tasks.agents:MovePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Flex-Pull10-v0",
    entry_point="flex_tasks.pull_env:PullEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "flex_tasks.pull_env_cfg:Pull10EnvCfg",
        "rsl_rl_cfg_entry_point": "flex_tasks.agents:Pull10PPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Flex-PullBench-v0",
    entry_point="flex_tasks.pull_env:PullEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "flex_tasks.pull_env_cfg:PullBenchEnvCfg",
        "rsl_rl_cfg_entry_point": "flex_tasks.agents:PullBenchPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Flex-Duo-v0",
    entry_point="flex_tasks.duo_env:DuoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "flex_tasks.duo_env_cfg:DuoEnvCfg",
        "rsl_rl_cfg_entry_point": "flex_tasks.agents:DuoPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Flex-Team-v0",
    entry_point="flex_tasks.team_env:TeamEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "flex_tasks.team_env_cfg:TeamEnvCfg",
        "skrl_ippo_cfg_entry_point": "flex_tasks:team_ippo.yaml",
        "skrl_cfg_entry_point": "flex_tasks:team_ippo.yaml",
    },
)

gym.register(
    id="Flex-Phys-v0",
    entry_point="flex_tasks.phys_env:PhysEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "flex_tasks.phys_env_cfg:PhysEnvCfg",
        "rsl_rl_cfg_entry_point": "flex_tasks.agents:PhysPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
