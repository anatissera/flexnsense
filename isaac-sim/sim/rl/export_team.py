"""Export the two jointly trained policies (skrl IPPO checkpoint) as stand-alone TorchScript files, and check them.

Each exported file is the whole decision of one arm: it scales the observation the way training did and returns the
action the policy would take without exploration noise. The check runs the trained agent and the exported files on the
same observations and prints the largest difference.

    source ~/isaac-env.sh && cd ~/IsaacLab
    PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/export_team.py \\
        --checkpoint ~/IsaacLab/logs/skrl/flex_team/<run>/checkpoints/best_agent.pt --out ~/flexnsense-sim/stage/isaac-sim/policies/team_v1
Writes <out>/pull/policy.pt, <out>/move/policy.pt and <out>/io_sample.npz
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
add_launcher_args(parser)
args_cli = parser.parse_args()

import os

import numpy as np
import torch
import yaml

import flex_tasks
from flex_tasks.team_env_cfg import TeamEnvCfg


class Exported(torch.nn.Module):
    """Observation scaling, then the policy's layers (ELU between them), as plain tensors."""

    def __init__(self, mean, var, eps: float, clip: float, linears):
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", torch.sqrt(var.clone()) + eps)
        self.clip = float(clip)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(l.in_features, l.out_features) for l in linears])
        for mine, theirs in zip(self.layers, linears):
            mine.load_state_dict(theirs.state_dict())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.clamp((obs - self.mean) / self.std, -self.clip, self.clip)
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < n - 1:
                x = torch.nn.functional.elu(x)
        return x


def main():
    env_cfg = TeamEnvCfg()
    env_cfg.scene.num_envs = 64
    env_cfg.seed = 3
    with launch_simulation(env_cfg, args_cli):
        from skrl.utils.runner.torch import Runner

        from isaaclab_rl.skrl import SkrlVecEnvWrapper

        from flex_tasks.team_env import TeamEnv

        env = TeamEnv(env_cfg)
        wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
        agent_cfg = yaml.safe_load(open(os.path.join(os.path.dirname(flex_tasks.__file__), "team_ippo.yaml")))
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        runner = Runner(wrapped, agent_cfg)
        runner.agent.load(os.path.expanduser(args_cli.checkpoint))
        runner.agent.enable_training_mode(False, apply_to_models=True)
        agent = runner.agent
        out_dir = os.path.expanduser(args_cli.out)
        obs, _ = wrapped.reset()
        states = wrapped.state()
        samples = {}
        for step in range(40):  # some steps into an episode, so the observations are not all the start posture
            with torch.inference_mode():
                res = agent.act(obs, states, timestep=0, timesteps=0)
                actions = {a: res[-1][a].get("mean_actions", res[0][a]) for a in env.cfg.possible_agents}
                if step == 39:
                    samples = {a: (obs[a].clone(), actions[a].clone()) for a in env.cfg.possible_agents}
                obs, _, _, _, _ = wrapped.step(actions)
                states = wrapped.state()
        io = {}
        for name in env.cfg.possible_agents:
            policy = agent.policies[name]
            pre = None
            for attr in ("_observation_preprocessor", "_state_preprocessor"):
                holder = getattr(agent, attr, None)
                cand = holder.get(name) if isinstance(holder, dict) else None
                if cand is not None and hasattr(cand, "running_mean"):
                    pre = cand
                    break
            if pre is None:
                raise RuntimeError(f"no observation scaler found for agent {name}: {[a for a in dir(agent) if 'preproc' in a]}")
            linears = [m for m in policy.modules() if isinstance(m, torch.nn.Linear)]
            o, a = samples[name]
            mod = Exported(pre.running_mean.float(), pre.running_variance.float(), float(pre.epsilon),
                           float(pre.clip_threshold), linears).to("cpu").eval()
            with torch.no_grad():
                got = mod(o.cpu())
            diff = float((got - a.cpu()).abs().max())
            os.makedirs(f"{out_dir}/{name}", exist_ok=True)
            torch.jit.script(mod).save(f"{out_dir}/{name}/policy.pt")
            io[f"{name}_obs"], io[f"{name}_action"] = o.cpu().numpy(), a.cpu().numpy()
            print(f"[EXPORT] {name}: {len(linears)} layers {[l.out_features for l in linears]}, obs {o.shape[1]} -> action {a.shape[1]} | "
                  f"largest difference from the trained agent on {len(o)} samples = {diff:.2e}", flush=True)
        np.savez(f"{out_dir}/io_sample.npz", **io)
        print("[EXPORT] wrote", out_dir, flush=True)
        env.close()


if __name__ == "__main__":
    main()
