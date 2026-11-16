"""Does the move arm of the physical task answer its commands? Prints joint and gripper motion for fixed actions."""
import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()

import torch

from flex_tasks.phys_env_cfg import PhysEnvCfg


def main():
    env_cfg = PhysEnvCfg()
    env_cfg.scene.num_envs = 8
    with launch_simulation(env_cfg, args_cli):
        from flex_tasks.phys_env import PhysEnv

        env = PhysEnv(env_cfg)
        r3 = lambda v: [round(float(x), 3) for x in v]
        print("[DBG] B joints:", env.B.joint_names, "| act ids", env.act_ids.tolist(), "jaw ids", env.jaw_ids.tolist(), flush=True)
        print("[DBG] B limits (act):", [r3(env.limits[i]) for i in env.act_ids], "| jaws", [r3(env.limits[i]) for i in env.jaw_ids], flush=True)
        with torch.inference_mode():
            for k in range(6):
                env.reset()
                a = torch.zeros(8, 6, device=env.device)
                if k < 5:
                    a[:, k] = 1.0
                else:
                    a[:, 5] = 1.0
                q0 = env.B.data.joint_pos.torch[0].clone()
                t0 = env._where()[0][0].clone()
                rew = 0.0
                for _ in range(25):
                    _, r, _, _, _ = env.step(a)
                    rew += float(r[0])
                q1 = env.B.data.joint_pos.torch[0]
                tip, _, grasp, loop, _, _ = env._where()
                print(f"[DBG] action {k}=+1 for 2.5 s: joints moved {r3((q1 - q0)[env.arm_ids])} jaws {r3((q1 - q0)[env.jaw_ids])} | tip moved "
                      f"{r3(tip[0] - t0)} | tip {r3(tip[0])} grasp {r3(grasp[0])} dist {float((tip[0]-grasp[0]).norm()):.3f} | stage {int(env.stage[0])} reward sum {rew:.2f}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
