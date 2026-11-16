"""Record episodes of the two jointly trained policies working together (numbers only), one episode per table.

    source ~/isaac-env.sh && cd ~/IsaacLab
    PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/record_team.py \\
        --checkpoint <agent.pt> --episodes 1024 --out ~/flexnsense-sim/dataset/team
Writes <out>_pull.npz and <out>_move.npz (same episodes, one file per arm) and <out>_report.json.
Only complete, clean episodes are kept: the loop was placed, nothing was dropped, no finger was hit, and no pull went
more than 10 % over the force limit. The report says how many were left out and why.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--episodes", type=int, default=1024)
parser.add_argument("--out", required=True)
add_launcher_args(parser)
args_cli = parser.parse_args()

import json
import os

import numpy as np
import torch
import yaml

import flex_tasks
from flex_tasks.team_env_cfg import TeamEnvCfg

EVERY = 1  # the policies act 10 times a second; one row per decision
JAW_PER_RAD = 0.0115


def main():
    env_cfg = TeamEnvCfg()
    env_cfg.scene.num_envs = args_cli.episodes
    env_cfg.seed = 1611
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
        arm = env.arm_ids
        jaw = int(env.jaw_ids[1])  # runs 0 (open) to +25.5 mm (shut)
        lim = env.limits
        lo = torch.cat([lim[arm, 0], torch.zeros(1, device=dev)]).cpu().numpy()
        hi = torch.cat([lim[arm, 1], lim[jaw, 1:2] / JAW_PER_RAD]).cpu().numpy()
        obs, _ = wrapped.reset()
        states = wrapped.state()
        from_table = env.free.clone()
        rec = {k: [] for k in ("pa", "ga", "pb", "gb", "xa", "xb")}
        peak = torch.zeros(n, device=dev)
        failed = torch.zeros(n, dtype=torch.bool, device=dev)
        why = {"dropped": 0, "hit_a_finger": 0, "too_late": 0}
        for i in range(int(env.max_episode_length) - 2):
            with torch.inference_mode():
                out = runner.agent.act(obs, states, timestep=0, timesteps=0)
                actions = {a: out[-1][a].get("mean_actions", out[0][a]) for a in env.cfg.possible_agents}
                obs, _, _, _, _ = wrapped.step(actions)
                states = wrapped.state()
            for key, ev in (("dropped", env.ev_drop), ("hit_a_finger", env.ev_snag), ("too_late", env.ev_late)):
                why[key] += int(((ev > 0) & ~failed).sum())
            failed |= (env.ev_drop > 0) | (env.ev_snag > 0) | (env.ev_late > 0)
            peak = torch.maximum(peak, env.force * (~failed).float())
            qa, qb = env.A.data.joint_pos.torch, env.B.data.joint_pos.torch
            pack = lambda q, t: (torch.cat([q[:, arm], q[:, jaw : jaw + 1] / JAW_PER_RAD], -1).cpu().numpy(),
                                 torch.cat([t[:, arm], t[:, jaw : jaw + 1] / JAW_PER_RAD], -1).cpu().numpy())
            pa, ga = pack(qa, env.qA_tgt)
            pb, gb = pack(qb, env.qB_tgt)
            ph = env._phase().float().unsqueeze(-1)
            xa = torch.cat([env.force.unsqueeze(-1), env.force_obs.unsqueeze(-1), env.seen.float().unsqueeze(-1), ph,
                            env.held.float().unsqueeze(-1)], -1)
            dst, _ = env._finger(1 - env.src)
            xb = torch.cat([env.loop, dst, env.held.float().unsqueeze(-1), env.placed.float().unsqueeze(-1), ph,
                            env.force_obs.unsqueeze(-1)], -1)
            for k, v in (("pa", pa), ("ga", ga), ("pb", pb), ("gb", gb), ("xa", xa.cpu().numpy()), ("xb", xb.cpu().numpy())):
                rec[k].append(v)
        over = peak > 1.1 * env_cfg.cap
        keep = (env.placed & ~failed & ~over).cpu().numpy()
        steps = len(rec["pa"])
        valid = np.repeat(keep[:, None], steps, axis=1)
        out = os.path.expanduser(args_cli.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        common = dict(valid=valid, lo=lo, hi=hi, dt=np.array(env.step_dt * EVERY))
        np.savez_compressed(out + "_pull.npz", pos=np.stack(rec["pa"], 1), goal=np.stack(rec["ga"], 1), vel=np.zeros((n, steps, 8), np.float32),
                            extra=np.stack(rec["xa"], 1), task=np.array("team_pull"),
                            extra_names=np.array(["force_n", "force_read_by_camera_n", "marker_seen", "phase", "scale_held_by_move_arm"]), **common)
        np.savez_compressed(out + "_move.npz", pos=np.stack(rec["pb"], 1), goal=np.stack(rec["gb"], 1), vel=np.zeros((n, steps, 8), np.float32),
                            extra=np.stack(rec["xb"], 1), task=np.array("team_move"),
                            extra_names=np.array(["loop_x_m", "loop_y_m", "loop_z_m", "target_x_m", "target_y_m", "target_z_m", "held",
                                                  "placed", "phase", "force_read_by_camera_n"]), **common)
        report = {"episodes_run": n, "episodes_kept": int(keep.sum()), "rows_per_episode": steps, "rows_per_second": round(1 / (env.step_dt * EVERY)),
                  "left_out": {**why, "not_placed_by_the_end": int((~env.placed & ~failed).sum()),
                               "a_pull_more_than_10pct_over_the_limit": int((over & env.placed & ~failed).sum())},
                  "kept_that_started_with_the_loop_on_the_table": int((keep & from_table.cpu().numpy()).sum()),
                  "kept_that_started_with_the_loop_on_a_finger": int((keep & ~from_table.cpu().numpy()).sum()),
                  "peak_force_in_kept_episodes_N_mean": round(float(peak.cpu().numpy()[keep].mean()), 3) if keep.any() else None}
        json.dump(report, open(out + "_report.json", "w"), indent=2)
        print("[REC]", json.dumps(report), flush=True)
        env.close()


if __name__ == "__main__":
    main()
