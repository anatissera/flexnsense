"""Check the physical-contact bench before training: where things are, whether a pull reads 200 g, and pictures.

    PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/look_phys.py
Writes ~/flexnsense-sim/runs/phys_look/: bench_*.png, top_*.png
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="~/flexnsense-sim/runs/phys_look")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

import os

import torch
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import CameraCfg

from flex_tasks.phys_env_cfg import PhysEnvCfg

OUT = os.path.expanduser(args_cli.out)


def cam(path, w, h):
    return CameraCfg(prim_path=path, update_period=0, height=h, width=w, data_types=["rgb"],
                     renderer_cfg=IsaacRtxRendererCfg(enable_scene_partitioning=False),
                     spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                                                      clipping_range=(0.05, 500.0)))


def main():
    env_cfg = PhysEnvCfg()
    env_cfg.scene.num_envs = 16
    env_cfg.seed = 4
    env_cfg.scene.cam_bench = cam("/World/CamBench", 1280, 720)
    env_cfg.scene.cam_top = cam("/World/CamTop", 960, 720)
    with launch_simulation(env_cfg, args_cli):
        from PIL import Image

        from flex_tasks.phys_env import PhysEnv

        env = PhysEnv(env_cfg)
        dev = env.device
        o = env.scene.env_origins
        T = lambda v: torch.tensor(v, device=dev)
        r3 = lambda v: [round(float(x), 3) for x in v]
        os.makedirs(OUT, exist_ok=True)
        zero = torch.zeros(env.num_envs, 6, device=dev)

        def shot(i, tag):
            e0 = o[i]
            env.scene["cam_bench"].set_world_poses_from_view((e0 + T([-0.50, 0.05, 0.44])).unsqueeze(0), (e0 + T([0.08, 0.36, 0.04])).unsqueeze(0))
            env.scene["cam_top"].set_world_poses_from_view((e0 + T([0.0, 0.415, 0.50])).unsqueeze(0), (e0 + T([0.0, 0.43, 0.0])).unsqueeze(0))
            for _ in range(3):
                env.sim.render()
            for name in ("bench", "top"):
                c = env.scene[f"cam_{name}"]
                c.update(dt=env.step_dt)
                Image.fromarray(c.data.output["rgb"][0, ..., :3].cpu().numpy()).save(f"{OUT}/{name}_{tag}.png")

        with torch.inference_mode():
            env.reset()
            for _ in range(15):
                env.step(zero)
            tip_b, pointing, grasp, loop, along, force = env._where()
            on = (env.started_at == 2).nonzero().squeeze(-1)
            tb = (env.started_at == 0).nonzero().squeeze(-1)
            print("[LOOK] joints A:", env.A.joint_names, flush=True)
            print("[LOOK] fingers index, middle:", r3(env.fingers[0]), r3(env.fingers[1]), flush=True)
            for i in list(tb[:2]) + list(on[:2]):
                i = int(i)
                print(f"[LOOK] table {i} start stage {int(env.started_at[i])}: loop {r3(loop[i])} grasp {r3(grasp[i])} along {r3(along[i])} "
                      f"move tip {r3(tip_b[i])} | around its target {bool(env._around(loop, env._target())[i])} | force {float(force[i]):.2f} N", flush=True)
            shot(int(tb[0]), "on_table")
            shot(int(on[0]), "on_index")
            # a pull on the tables whose loop is on the index finger
            env.pull_mode[on] = 1
            peak = torch.zeros(env.num_envs, device=dev)
            for k in range(90):
                env.step(zero)
                f = env._where()[5]
                peak = torch.maximum(peak, f)
                if k in (20, 35, 50):
                    print(f"[LOOK] pull t={k/10:.1f}s: force {r3(f[on][:4])} N | pull share {r3(env.pull_s[on][:4])} | mode {env.pull_mode[on][:4].tolist()}", flush=True)
                if k == 35:
                    shot(int(on[0]), "pulling")
            loop = env._where()[3]
            print(f"[LOOK] pull peaks {r3(peak[on])} N | after it the loop is still round the index finger: "
                  f"{env._around(loop, env.fingers[0].expand(env.num_envs, 3))[on].tolist()} | force now {r3(env._where()[5][on][:4])}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
