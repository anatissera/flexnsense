"""Run the two jointly trained policies on tables they have not seen and count what happens (no pictures).

    source ~/isaac-env.sh && cd ~/IsaacLab
    PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/eval_team.py \\
        --checkpoint ~/IsaacLab/logs/skrl/flex_team/<run>/checkpoints/best_agent.pt --out <folder>
One episode per table, at the real 8 mm placing tolerance. Writes <out>/summary.json.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--out", default="~/flexnsense-sim/runs/team/eval")
add_launcher_args(parser)
args_cli = parser.parse_args()

import json
import os

import torch
import yaml

import flex_tasks
from flex_tasks.team_env_cfg import TeamEnvCfg


def main():
    env_cfg = TeamEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 2026
    env_cfg.place_within_start = env_cfg.place_within
    with launch_simulation(env_cfg, args_cli):
        from skrl.utils.runner.torch import Runner

        from isaaclab_rl.skrl import SkrlVecEnvWrapper

        from flex_tasks.team_env import TeamEnv

        env = TeamEnv(env_cfg)
        dev, n = env.device, env.num_envs
        wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
        agent_cfg = yaml.safe_load(open(os.path.join(os.path.dirname(flex_tasks.__file__), "team_ippo.yaml")))
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        runner = Runner(wrapped, agent_cfg)
        runner.agent.load(os.path.expanduser(args_cli.checkpoint))
        runner.agent.enable_training_mode(False, apply_to_models=True)
        obs, _ = wrapped.reset()
        states = wrapped.state()
        z = lambda: torch.zeros(n, device=dev)
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        placed, dropped, snagged, late = z(), z(), z(), z()
        t_place, lift, peak1, peak2, over, sleep_end, move_speed = z(), z(), z(), z(), z(), z(), z()
        from_table = env.free.clone()
        for i in range(int(env.max_episode_length) - 2):
            with torch.inference_mode():
                out = runner.agent.act(obs, states, timestep=0, timesteps=0)
                actions = {a: out[-1][a].get("mean_actions", out[0][a]) for a in env.cfg.possible_agents}
                tip_before = env._tips()[1].clone()
                obs, _, _, _, _ = wrapped.step(actions)
                states = wrapped.state()
            ended = alive & ((env.ev_drop > 0) | (env.ev_snag > 0) | (env.ev_late > 0))
            dropped += (alive & (env.ev_drop > 0)).float()
            snagged += (alive & (env.ev_snag > 0)).float()
            late += (alive & (env.ev_late > 0)).float()
            a = alive.float()
            placed = torch.where(alive, env.placed.float(), placed)
            t_place = torch.where(alive & env.placed, env.t_place, t_place)
            lift = torch.where(alive, torch.maximum(lift, env.lifted), lift)
            peak1 = torch.where(alive, torch.maximum(peak1, env.peak1), peak1)
            peak2 = torch.where(alive, torch.maximum(peak2, env.peak2), peak2)
            over += a * (env.force > 1.1 * env_cfg.cap).float() * env.step_dt
            speed = (env._tips()[1] - tip_before).norm(dim=-1) / env.step_dt
            if i < 3:
                speed = torch.zeros_like(speed)
            move_speed = torch.where(alive, torch.maximum(move_speed, speed), move_speed)
            sleep_end = torch.where(alive, env._sleep_error(), sleep_end)
            alive &= ~ended
        m = lambda v, sel=None: round(float((v if sel is None else v[sel]).mean()), 4) if (sel is None or bool(sel.any())) else None
        ok = placed > 0
        summary = {
            "tables": n,
            "place_tolerance_m": env_cfg.place_within,
            "share_loop_placed": m(placed),
            "share_placed_when_loop_started_on_a_finger": m(placed, ~from_table),
            "share_placed_when_loop_started_on_the_table": m(placed, from_table),
            "share_dropped": m(dropped), "share_hit_a_finger": m(snagged), "share_too_late": m(late),
            "seconds_from_end_of_pull_to_placed_mean": m(t_place, ok),
            "seconds_to_placed_slowest": round(float(t_place[ok].max()), 3) if bool(ok.any()) else None,
            "highest_lift_m_mean": m(lift, ok),
            "move_arm_fastest_gripper_speed_m_per_s_mean": m(move_speed),
            "move_arm_fastest_gripper_speed_m_per_s_worst": round(float(move_speed.max()), 3),
            "move_arm_away_from_rest_at_end_rad_mean": m(sleep_end, ok),
            "limit_N": env_cfg.cap,
            "first_pull_peak_N_mean": m(peak1, ~from_table), "first_pull_peak_N_worst": round(float(peak1.max()), 3),
            "second_pull_peak_N_mean": m(peak2, ok), "second_pull_peak_N_worst": round(float(peak2.max()), 3),
            "share_of_tables_with_a_pull_over_110pct_limit": m(((peak1 > 1.1 * env_cfg.cap) | (peak2 > 1.1 * env_cfg.cap)).float()),
            "seconds_above_110pct_limit_mean": m(over),
        }
        out_dir = os.path.expanduser(args_cli.out)
        os.makedirs(out_dir, exist_ok=True)
        json.dump(summary, open(f"{out_dir}/summary.json", "w"), indent=2)
        print("[EVAL]", json.dumps(summary), flush=True)
        env.close()


if __name__ == "__main__":
    main()
