"""Pictures of the two-arm bench before any training: the table, both arms, the hand and the scale as drawn.

    source ~/isaac-env.sh && cd ~/IsaacLab
    PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/look_team.py
Writes ~/flexnsense-sim/runs/team_look/: bench.png, overhead.png (what the camera that reads the scale sees), field.png
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--side", type=int, default=4)
parser.add_argument("--out", default="~/flexnsense-sim/runs/team_look")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

import os

import torch
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import CameraCfg

from flex_tasks.team_env_cfg import TeamEnvCfg

OUT = os.path.expanduser(args_cli.out)


def cam(path, w, h, focal=24.0):
    return CameraCfg(prim_path=path, update_period=0, height=h, width=w, data_types=["rgb"],
                     renderer_cfg=IsaacRtxRendererCfg(enable_scene_partitioning=False),
                     spawn=sim_utils.PinholeCameraCfg(focal_length=focal, focus_distance=400.0, horizontal_aperture=20.955,
                                                      clipping_range=(0.05, 500.0)))


def main():
    env_cfg = TeamEnvCfg()
    env_cfg.scene.num_envs = args_cli.side**2
    env_cfg.seed = 4
    env_cfg.scene.cam_bench = cam("/World/CamBench", 1280, 720)
    env_cfg.scene.cam_top = cam("/World/CamTop", 960, 720)
    env_cfg.scene.cam_field = cam("/World/CamField", 1280, 720)
    with launch_simulation(env_cfg, args_cli):
        from PIL import Image

        from flex_tasks.bench_draw import draw, make_markers
        from flex_tasks.team_env import TeamEnv

        env = TeamEnv(env_cfg)
        dev = env.device
        marks = make_markers()
        o = env.scene.env_origins
        i0 = int((o[:, 0] + o[:, 1]).argmin())
        e0 = o[i0]
        T = lambda v: torch.tensor(v, device=dev)
        env.scene["cam_bench"].set_world_poses_from_view((e0 + T([-0.50, 0.05, 0.44])).unsqueeze(0), (e0 + T([0.08, 0.36, 0.04])).unsqueeze(0))
        env.scene["cam_top"].set_world_poses_from_view((e0 + T([0.0, 0.415, 0.50])).unsqueeze(0), (e0 + T([0.0, 0.43, 0.0])).unsqueeze(0))
        mid = (o.min(dim=0).values + o.max(dim=0).values) / 2
        span = float((o.max(dim=0).values - o.min(dim=0).values).max()) + 2.0
        env.scene["cam_field"].set_world_poses_from_view((mid + T([-0.62, -0.62, 0.50]) * span).unsqueeze(0), (mid + T([0.0, 0.0, 0.1])).unsqueeze(0))
        os.makedirs(OUT, exist_ok=True)
        zero = {"pull": torch.zeros(env.num_envs, 5, device=dev), "move": torch.zeros(env.num_envs, 6, device=dev)}
        with torch.inference_mode():
            env.reset()
            for i in range(22):  # about 2 seconds: a pull is under way on most tables
                env.step(zero)
                draw(env, marks)
        for _ in range(3):
            env.sim.render()
        for name in ("bench", "top", "field"):
            c = env.scene[f"cam_{name}"]
            c.update(dt=env.step_dt)
            Image.fromarray(c.data.output["rgb"][0, ..., :3].cpu().numpy()).save(f"{OUT}/{name}.png")
        print("[LOOK] wrote", OUT, "| force on table 0:", round(float(env.force[i0]), 2), "N | loop on finger", int(env._on_finger()[i0]), flush=True)
        env.close()


if __name__ == "__main__":
    main()
