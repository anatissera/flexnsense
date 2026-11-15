"""A short film of the pull-arm training: one arm, then 4, 16 (untrained), then 16, 64, 256, 1024 (trained).

One camera starts close on a single arm and pulls back in stages. Every arm has a ball where the patient's finger is
and a dotted line for the scale; the ball is green while the force is under the limit and red while it is over.

    source ~/isaac-env.sh && cd ~/IsaacLab
    PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/film_movie.py \\
        --before <model_0.pt> --after <model_599.pt>
Frames: ~/flexnsense-sim/runs/rl_pull/movie/frames/  (join them with ffmpeg at 30 pictures a second)
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--before", required=True, help="checkpoint of the untrained policy (model_0.pt)")
parser.add_argument("--after", required=True, help="checkpoint of the trained policy")
parser.add_argument("--side", type=int, default=32, help="arms per side of the field (32 -> 1024 arms)")
parser.add_argument("--out", default="~/flexnsense-sim/runs/rl_pull/movie")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

import math
import os

import torch
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import CameraCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from flex_tasks.agents import PullPPORunnerCfg
from flex_tasks.pull_env_cfg import PullEnvCfg

OUT = os.path.expanduser(args_cli.out)
# (seconds, arms per side in view, which policy, caption)
STAGES = [
    (4.0, 1, "before", "1 arm, before training"),
    (3.5, 2, "before", "4 arms, before training"),
    (3.5, 4, "before", "16 arms, before training (red = over the force limit)"),
    (4.5, 4, "after", "The same 16 arms after 8 minutes of training"),
    (4.0, 8, "after", "64 trained arms: each gives way when pulled too hard"),
    (4.0, 16, "after", "256 trained arms"),
    (6.5, None, "after", "{n} arms at once (the training used 4,096)"),
]
MOVE_S = 1.4  # seconds the camera takes to pull back to the next stage
if os.environ.get("FILM_TEST"):  # a quick look: a few pictures of a close view and of the whole field
    STAGES = [(0.3, 2, "after", "test: close"), (0.3, None, "after", "test: {n} arms")]
    MOVE_S = 0.05


def main():
    env_cfg = PullEnvCfg()
    n = args_cli.side**2
    env_cfg.scene.num_envs = n
    env_cfg.seed = 3
    env_cfg.scene.cam = CameraCfg(
        prim_path="/World/FilmCam", update_period=0, height=720, width=1280, data_types=["rgb"],
        # Isaac Lab normally hides every environment from cameras outside it (so that a policy's camera sees only its
        # own arm). For this film one camera must see them all, so that separation is switched off.
        renderer_cfg=IsaacRtxRendererCfg(enable_scene_partitioning=False),
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 500.0)),
    )
    agent_cfg = PullPPORunnerCfg()
    with launch_simulation(env_cfg, args_cli):
        from PIL import Image, ImageDraw, ImageFont

        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.utils import to_dict

        from flex_tasks.pull_env import PullEnv

        env = PullEnv(env_cfg)
        dev = env.device
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        policies = {}
        for name, ckpt in (("before", args_cli.before), ("after", args_cli.after)):
            runner = OnPolicyRunner(wrapped, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            runner.load(os.path.expanduser(ckpt))
            policies[name] = (runner, runner.get_inference_policy(device=dev))  # keep the runner: the policy lives in it
        cam = env.scene["cam"]
        mat = lambda c: sim_utils.PreviewSurfaceCfg(diffuse_color=c, emissive_color=tuple(0.35 * v for v in c))
        marks = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/film", markers={
            "ok": sim_utils.SphereCfg(radius=0.016, visual_material=mat((0.15, 0.85, 0.45))),
            "over": sim_utils.SphereCfg(radius=0.016, visual_material=mat((1.0, 0.18, 0.12))),
            "scale": sim_utils.SphereCfg(radius=0.006, visual_material=mat((0.82, 0.84, 0.88))),
        }))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
            small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except OSError:
            font = small = ImageFont.load_default()
        o = env.scene.env_origins
        gap = env_cfg.scene.env_spacing
        # The arm the film starts on stands at a corner of the field. The camera is inside the field and looks out
        # towards that corner, so at first there is only that one arm and empty floor behind it; pulling back brings
        # the arms around it into view.
        corner = o[(o[:, 0] + o[:, 1]).argmin()].clone()
        view = torch.tensor([0.66, 0.58, 0.48], device=dev)

        def framing(k: float):
            """Where to look and from how far, to see k x k arms from the corner."""
            aim = corner + torch.tensor([(k - 1) / 2 * gap, (k - 1) / 2 * gap + 0.18, 0.10], device=dev)
            return aim, 1.15 + 1.05 * (k - 1) * gap

        os.makedirs(f"{OUT}/frames", exist_ok=True)
        with torch.inference_mode():
            env.reset()
            obs = wrapped.get_observations()
        frame, k_prev = 0, 1.0
        for seconds, k, which, caption in STAGES:
            k = float(k or args_cli.side)
            policy = policies[which][1]
            steps = int(seconds / env.step_dt)
            for i in range(steps):
                with torch.inference_mode():
                    obs, _, _, _ = wrapped.step(policy(obs))
                if i % 2:
                    continue
                p = min(1.0, i * env.step_dt / MOVE_S)
                p = p * p * (3 - 2 * p)
                k_now = math.exp(math.log(k_prev) + (math.log(k) - math.log(k_prev)) * p)  # pull back evenly on a log scale
                aim, dist = framing(k_now)
                cam.set_world_poses_from_view((aim + view * dist).unsqueeze(0), aim.unsqueeze(0))
                tip, _ = env._tip()
                finger = env.home + env.u * (env_cfg.rest + env._pull().unsqueeze(-1))
                over = env.force > 1.05 * env_cfg.cap
                grow = max(1.0, dist / 6.0)  # bigger balls from far away, so the colours still read
                pts = torch.cat([finger] + [tip + (finger - tip) * j / 6 for j in range(1, 6)])
                idx = torch.cat([over.long(), torch.full((5 * n,), 2, dtype=torch.long, device=dev)])
                scl = torch.cat([torch.full((n, 3), grow, device=dev), torch.full((5 * n, 3), min(grow, 2.0), device=dev)])
                marks.visualize(translations=pts, scales=scl, marker_indices=idx)
                im = Image.fromarray(cam.data.output["rgb"][0, ..., :3].cpu().numpy())
                canvas = Image.new("RGB", (im.width, im.height + 110), (14, 20, 31))
                canvas.paste(im, (0, 110))
                dr = ImageDraw.Draw(canvas)
                dr.text((28, 14), caption.format(n=f"{n:,}"), font=font, fill=(244, 247, 251))
                share = float(over.float().mean()) * 100
                dr.text((28, 66), f"limit 1.96 N (200 gf)   green under it, red over it   over the limit now: {share:4.1f} % of arms   "
                        "simulated", font=small,
                        fill=(255, 130, 120) if share > 5 else (150, 230, 190))
                canvas.save(f"{OUT}/frames/frame_{frame:04d}.png")
                frame += 1
            k_prev = k
        print("[FILM] wrote", frame, "frames to", OUT, flush=True)
        env.close()


if __name__ == "__main__":
    main()
