"""A short film of the two arms learning together: one table, then 4, 16 (before training), then 16, 64, 256, 1024
(after training). Every table has both arms, the hand and the scale as drawn in the single-table film.

    source ~/isaac-env.sh && cd ~/IsaacLab
    PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/film_team_movie.py \\
        --checkpoint ~/IsaacLab/logs/skrl/flex_team/<run>/checkpoints/best_agent.pt
Frames: <out>/frames/ (10 pictures a second of simulated time).
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--side", type=int, default=32, help="tables per side of the field (32 -> 1024 tables)")
parser.add_argument("--out", default="~/flexnsense-sim/runs/team/movie")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

import math
import os

import torch
import yaml
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import CameraCfg

import flex_tasks
from flex_tasks.team_env_cfg import TeamEnvCfg

OUT = os.path.expanduser(args_cli.out)
STAGES = [
    (5.0, 1, "before", "One table, before training: neither arm knows its job"),
    (4.0, 2, "before", "4 tables, before training"),
    (4.0, 4, "before", "16 tables, before training"),
    (9.0, 4, "after", "The same 16 tables after training together"),
    (7.0, 8, "after", "64 tables: one arm limits the force, the other moves the scale"),
    (5.0, 16, "after", "256 tables"),
    (8.0, None, "after", "{n} tables at once ({m} arms; the training used 4,096 tables)"),
]
MOVE_S = 1.6
if os.environ.get("FILM_TEST"):
    STAGES = [(0.6, 1, "after", "test: close"), (0.4, None, "after", "test: {n} tables")]
    MOVE_S = 0.05


def main():
    env_cfg = TeamEnvCfg()
    n = args_cli.side**2
    env_cfg.scene.num_envs = n
    env_cfg.seed = 9
    env_cfg.place_within_start = env_cfg.place_within
    env_cfg.scene.light.spawn = sim_utils.DomeLightCfg(intensity=900.0, color=(0.9, 0.93, 1.0))
    env_cfg.scene.cam = CameraCfg(
        prim_path="/World/FilmCam", update_period=0, height=720, width=1280, data_types=["rgb"],
        renderer_cfg=IsaacRtxRendererCfg(enable_scene_partitioning=False),
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 800.0)),
    )
    with launch_simulation(env_cfg, args_cli):
        from PIL import Image, ImageDraw, ImageFont
        from skrl.utils.runner.torch import Runner

        from isaaclab_rl.skrl import SkrlVecEnvWrapper

        from flex_tasks.bench_draw import draw, make_markers
        from flex_tasks.team_env import TeamEnv

        env = TeamEnv(env_cfg)
        dev = env.device
        wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
        agent_cfg = yaml.safe_load(open(os.path.join(os.path.dirname(flex_tasks.__file__), "team_ippo.yaml")))
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        runner = Runner(wrapped, agent_cfg)  # freshly made: this is what "before training" looks like
        runner.agent.enable_training_mode(False, apply_to_models=True)
        loaded = False
        marks = make_markers()
        cam = env.scene["cam"]
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
            small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except OSError:
            font = small = ImageFont.load_default()
        o = env.scene.env_origins
        gap = env_cfg.scene.env_spacing
        # start on the table at one corner; the camera stands inside the field and looks out towards it
        corner = o[(o[:, 0] + o[:, 1]).argmax()].clone()
        view = torch.tensor([-0.72, -0.42, 0.55], device=dev)

        def framing(k: float):
            aim = corner + torch.tensor([-(k - 1) / 2 * gap + 0.08, -(k - 1) / 2 * gap + 0.38, 0.05], device=dev)
            return aim, 1.05 + 1.02 * (k - 1) * gap

        os.makedirs(f"{OUT}/frames", exist_ok=True)
        obs, _ = wrapped.reset()
        states = wrapped.state()
        frame, k_prev = 0, 1.0
        for seconds, k, which, caption in STAGES:
            k = float(k or args_cli.side)
            if which == "after" and not loaded:
                runner.agent.load(os.path.expanduser(args_cli.checkpoint))
                runner.agent.enable_training_mode(False, apply_to_models=True)
                loaded = True
                obs, _ = wrapped.reset()  # every table starts a new episode with the trained policies
                states = wrapped.state()
            for i in range(int(seconds / env.step_dt)):
                with torch.inference_mode():
                    out = runner.agent.act(obs, states, timestep=0, timesteps=0)
                    actions = {a: out[-1][a].get("mean_actions", out[0][a]) for a in env.cfg.possible_agents}
                    if which == "before":  # an untrained policy acts at random
                        actions = {a: out[0][a] for a in env.cfg.possible_agents}
                    obs, _, _, _, _ = wrapped.step(actions)
                    states = wrapped.state()
                p = min(1.0, i * env.step_dt / MOVE_S)
                p = p * p * (3 - 2 * p)
                k_now = math.exp(math.log(k_prev) + (math.log(k) - math.log(k_prev)) * p)
                aim, dist = framing(k_now)
                cam.set_world_poses_from_view((aim + view * dist).unsqueeze(0), aim.unsqueeze(0))
                draw(env, marks, grow=max(1.0, dist / 6.0))
                im = Image.fromarray(cam.data.output["rgb"][0, ..., :3].cpu().numpy())
                canvas = Image.new("RGB", (im.width, im.height + 110), (14, 20, 31))
                canvas.paste(im, (0, 110))
                dr = ImageDraw.Draw(canvas)
                dr.text((28, 14), caption.format(n=f"{n:,}", m=f"{2 * n:,}"), font=font, fill=(244, 247, 251))
                over = float((env.force > 1.05 * env_cfg.cap).float().mean()) * 100
                placed = float(env.placed.float().mean()) * 100
                dr.text((28, 66), f"force over the 1.96 N limit on {over:4.1f} % of tables     loop on its target finger on "
                        f"{placed:4.1f} %     simulated", font=small, fill=(255, 130, 120) if over > 5 else (150, 230, 190))
                canvas.save(f"{OUT}/frames/frame_{frame:04d}.png")
                frame += 1
            k_prev = k
        print("[FILM] wrote", frame, "frames to", OUT, flush=True)
        env.close()


if __name__ == "__main__":
    main()
