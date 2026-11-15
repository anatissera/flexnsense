"""Record episodes of a trained policy acting in simulation (numbers only, no pictures), one episode per arm.

    source ~/isaac-env.sh && cd ~/IsaacLab
    PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/record_rollouts.py \\
        --task pull --checkpoint <model.pt> --episodes 512 --out ~/flexnsense-sim/dataset/pull.npz
Then turn the .npz into NormaCore-style Parquet with rollouts_to_parquet.py (plain Python, no simulator needed).

Recorded 20 times a second: the eight motors' angles and targets (motor 8, the gripper, is taken from the jaw
opening), the joint limits, and task signals (pull: the force and the finger's pull; move: where the loop is, whether
the tube is held, whether the loop has been placed).
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["pull", "move"], required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--episodes", type=int, default=512)
parser.add_argument("--out", required=True)
add_launcher_args(parser)
args_cli = parser.parse_args()

import os

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

EVERY = 3  # the policy acts 60 times a second; one row every third step = 20 rows a second
JAW_PER_RAD = 0.0115  # m of jaw travel per radian of the gripper motor (NormaCore's URDF)


def main():
    if args_cli.task == "pull":
        from flex_tasks.agents import PullPPORunnerCfg as AgentCfg
        from flex_tasks.pull_env_cfg import PullEnvCfg as EnvCfg
    else:
        from flex_tasks.agents import MovePPORunnerCfg as AgentCfg
        from flex_tasks.move_env_cfg import MoveEnvCfg as EnvCfg
    env_cfg, agent_cfg = EnvCfg(), AgentCfg()
    env_cfg.scene.num_envs = args_cli.episodes
    env_cfg.seed = 2026
    if args_cli.task == "move":
        env_cfg.place_within_start = env_cfg.place_within
    with launch_simulation(env_cfg, args_cli):
        from isaaclab.utils import to_dict

        if args_cli.task == "pull":
            from flex_tasks.pull_env import PullEnv as Env
        else:
            from flex_tasks.move_env import MoveEnv as Env
        env = Env(env_cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
        runner.load(os.path.expanduser(args_cli.checkpoint))
        policy = runner.get_inference_policy(device=env.device)
        robot = env.robot
        arm = robot.find_joints("rev_motor_0[1-7]", preserve_order=True)[0]
        jaw = robot.find_joints("rev_motor_08_2")[0][0]  # runs 0 (open) to +25.5 mm (shut)
        lim = robot.data.joint_pos_limits.torch[0]
        lo = torch.cat([lim[arm, 0], torch.zeros(1, device=env.device)])
        hi = torch.cat([lim[arm, 1], lim[jaw, 1:2] / JAW_PER_RAD])
        torch.manual_seed(2026)
        with torch.inference_mode():
            env.reset()
            obs = wrapped.get_observations()
        cols = {k: [] for k in ("pos", "goal", "vel", "extra")}
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        valid = []
        steps = int(env.max_episode_length) - 2
        for i in range(steps):
            with torch.inference_mode():
                obs, _, _, _ = wrapped.step(policy(obs))
            if args_cli.task == "move":
                alive &= ~(env.dropped | env.snagged)  # such an episode ends there; later rows belong to a new one
            if i % EVERY:
                continue
            q, qd = robot.data.joint_pos.torch, robot.data.joint_vel.torch
            cols["pos"].append(torch.cat([q[:, arm], q[:, jaw : jaw + 1] / JAW_PER_RAD], dim=-1).cpu().numpy())
            cols["goal"].append(torch.cat([env.q_tgt[:, arm], env.q_tgt[:, jaw : jaw + 1] / JAW_PER_RAD], dim=-1).cpu().numpy())
            cols["vel"].append(torch.cat([qd[:, arm], qd[:, jaw : jaw + 1] / JAW_PER_RAD], dim=-1).cpu().numpy())
            if args_cli.task == "pull":
                along, side, height, _ = env._rel()
                ex = torch.stack([env.force, env._pull(), along, side, height], dim=-1)
            else:
                ex = torch.cat([env.loop, env.dst, env.held.float().unsqueeze(-1), env.placed.float().unsqueeze(-1)], dim=-1)
            cols["extra"].append(ex.cpu().numpy())
            valid.append(alive.cpu().numpy().copy())
        extra_names = (["force_n", "finger_pull_m", "gripper_along_m", "gripper_side_m", "gripper_up_m"] if args_cli.task == "pull"
                       else ["loop_x_m", "loop_y_m", "loop_z_m", "target_x_m", "target_y_m", "target_z_m", "held", "placed"])
        out = os.path.expanduser(args_cli.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(
            out, pos=np.stack(cols["pos"], 1), goal=np.stack(cols["goal"], 1), vel=np.stack(cols["vel"], 1),
            extra=np.stack(cols["extra"], 1), valid=np.stack(valid, 1), lo=lo.cpu().numpy(), hi=hi.cpu().numpy(),
            extra_names=np.array(extra_names), dt=np.array(env.step_dt * EVERY), task=np.array(args_cli.task),
        )
        print("[REC] wrote", out, "episodes", env.num_envs, "rows per episode", len(valid), flush=True)
        env.close()


if __name__ == "__main__":
    main()
