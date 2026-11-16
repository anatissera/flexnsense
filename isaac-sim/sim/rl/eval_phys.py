"""Measure a physical-contact policy on many tables it has not seen, every episode starting from the table.

    PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/eval_phys.py --checkpoint <model.pt> --out <dir>
Writes <out>/summary.json and the stand-alone policy (policy.pt, policy.onnx).
"""
import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--out", default="~/flexnsense-sim/runs/phys/eval")
add_launcher_args(parser)
args_cli = parser.parse_args()

import json
import os

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from flex_tasks.agents import PhysPPORunnerCfg
from flex_tasks.phys_env_cfg import PhysEnvCfg

OUT = os.path.expanduser(args_cli.out)


def main():
    env_cfg = PhysEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 1611
    env_cfg.start_on_index_share = 0.0
    agent_cfg = PhysPPORunnerCfg()
    with launch_simulation(env_cfg, args_cli):
        from isaaclab.utils import to_dict

        from flex_tasks.phys_env import PhysEnv

        env = PhysEnv(env_cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
        runner.load(os.path.expanduser(args_cli.checkpoint))
        policy = runner.get_inference_policy(device=env.device)
        os.makedirs(OUT, exist_ok=True)
        try:
            runner.export_policy_to_jit(path=OUT, filename="policy.pt")
            runner.export_policy_to_onnx(path=OUT, filename="policy.onnx")
        except Exception as err:
            print("[EVAL] export failed:", err, flush=True)
        n, dev = env.num_envs, env.device
        z = lambda: torch.zeros(n, device=dev)
        far, peak1, peak2, speed, lift, t_index, t_middle, t_all = z(), z(), z(), z(), z(), z() - 1, z() - 1, z() - 1
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        with torch.inference_mode():
            env.reset()
            obs = wrapped.get_observations()
            for i in range(int(env.max_episode_length) - 2):
                obs, _, _, _ = wrapped.step(policy(obs))
                alive &= ~((env.episode_length_buf <= 1) & (i > 3))  # a table whose episode ended early is out
                t = (i + 1) * env.step_dt
                st = env.stage.float() + env.finished.float()
                t_index = torch.where(alive & (st >= 1) & (t_index < 0), torch.full_like(t_index, t), t_index)
                t_middle = torch.where(alive & (st >= 3) & (t_middle < 0), torch.full_like(t_middle, t), t_middle)
                t_all = torch.where(alive & (st >= 4) & (t_all < 0), torch.full_like(t_all, t), t_all)
                far = torch.where(alive, torch.maximum(far, st), far)
                peak1 = torch.where(alive, torch.maximum(peak1, env.peak[:, 0]), peak1)
                peak2 = torch.where(alive, torch.maximum(peak2, env.peak[:, 1]), peak2)
                speed = torch.where(alive, torch.maximum(speed, env.top_speed), speed)
                lift = torch.where(alive, torch.maximum(lift, env.lifted), lift)
        m = lambda v: round(float(v.mean()), 4)
        mm = lambda v, ok: round(float(v[ok].mean()), 3) if bool(ok.any()) else None
        done = far >= 4
        summary = {
            "tables": n, "every_episode_starts_with_the_loop_on_the_table": True,
            "share_loop_put_on_index_finger": m((far >= 1).float()),
            "share_loop_then_moved_to_middle_finger": m((far >= 3).float()),
            "share_whole_sequence_with_both_pulls": m(done.float()),
            "share_out_of_time": m((~alive).float()),
            "seconds_until_on_index_mean": mm(t_index, t_index > 0),
            "seconds_until_on_middle_mean": mm(t_middle, t_middle > 0),
            "seconds_whole_sequence_mean": mm(t_all, t_all > 0),
            "first_pull_top_reading_N_mean": mm(peak1, far >= 2), "first_pull_top_reading_N_highest": round(float(peak1.max()), 3),
            "second_pull_top_reading_N_mean": mm(peak2, done), "second_pull_top_reading_N_highest": round(float(peak2.max()), 3),
            "pull_target_N": env_cfg.pull_to,
            "move_arm_fastest_gripper_speed_m_per_s_mean": m(speed), "loop_highest_point_m_mean": m(lift),
        }
        print("[EVAL]", json.dumps(summary), flush=True)
        json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
        env.close()


if __name__ == "__main__":
    main()
