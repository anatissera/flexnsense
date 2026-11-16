"""What does a checkpoint of the physical task actually do? Prints distances over one episode for a few tables."""
import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
add_launcher_args(parser)
args_cli = parser.parse_args()

import os

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from flex_tasks.agents import PhysPPORunnerCfg
from flex_tasks.phys_env_cfg import PhysEnvCfg


def main():
    env_cfg = PhysEnvCfg()
    env_cfg.scene.num_envs = 64
    agent_cfg = PhysPPORunnerCfg()
    with launch_simulation(env_cfg, args_cli):
        from isaaclab.utils import to_dict

        from flex_tasks.phys_env import PhysEnv

        env = PhysEnv(env_cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
        runner.load(os.path.expanduser(args_cli.checkpoint))
        policy = runner.get_inference_policy(device=env.device)
        r3 = lambda v: [round(float(x), 3) for x in v]
        with torch.inference_mode():
            env.reset()
            obs = wrapped.get_observations()
            t0 = (env.started_at == 0).nonzero().squeeze(-1)[:3]
            t2 = (env.started_at == 2).nonzero().squeeze(-1)[:2]
            pick = torch.cat([t0, t2])
            for i in range(138):
                act = policy(obs)
                obs, _, _, _ = wrapped.step(act)
                if i % 12 == 0:
                    tip, _, grasp, loop, _, force = env._where()
                    d = (tip - grasp).norm(dim=-1)
                    tgt = env._target()
                    print(f"[POL] t={i/10:4.1f} dist to grasp {r3(d[pick])} | grip {r3(env.grip[pick])} | tip z {r3(tip[pick, 2])} | loop z {r3(loop[pick, 2])} "
                          f"| loop to target {r3((loop - tgt)[pick][:, 0:2].norm(dim=-1))} | act0 {r3(act[pick[0]])}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
