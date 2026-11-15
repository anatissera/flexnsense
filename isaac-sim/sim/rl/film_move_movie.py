"""A short film of the move-arm training: one arm, then 4, 16 (untrained), then 16, 64, 256, 1024 (trained).

Every arm has its bench drawn: four fingers, the scale (a dotted line from the ring to the loop), the loop as a ball.
The loop is grey while it is on its first finger, yellow while the arm carries it, green once it sits on the target
finger, red if it was dropped or pushed into a finger. Episodes are cut to 3 seconds so that the field keeps moving.

    source ~/isaac-env.sh && cd ~/IsaacLab
    PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/film_move_movie.py \\
        --before <model_0.pt> --after <trained model.pt>
Frames: ~/flexnsense-sim/runs/rl_move/movie/frames/
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--before", required=True)
parser.add_argument("--after", required=True)
parser.add_argument("--side", type=int, default=32, help="arms per side of the field (32 -> 1024 arms)")
parser.add_argument("--out", default="~/flexnsense-sim/runs/rl_move/movie")
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

from flex_tasks.agents import MovePPORunnerCfg
from flex_tasks.move_env_cfg import MoveEnvCfg

OUT = os.path.expanduser(args_cli.out)
STAGES = [
    (4.5, 1, "before", "1 arm, before training"),
    (3.5, 2, "before", "4 arms, before training"),
    (3.5, 4, "before", "16 arms, before training"),
    (5.0, 4, "after", "The same 16 arms after training"),
    (4.5, 8, "after", "64 trained arms: take the scale, lift, set the loop on the next finger"),
    (4.0, 16, "after", "256 trained arms"),
    (6.5, None, "after", "{n} arms at once (the training used 4,096)"),
]
MOVE_S = 1.4
if os.environ.get("FILM_TEST"):
    STAGES = [(1.2, 1, "after", "test: close"), (0.6, None, "after", "test: {n} arms")]
    MOVE_S = 0.05


def main():
    env_cfg = MoveEnvCfg()
    n = args_cli.side**2
    env_cfg.scene.num_envs = n
    env_cfg.seed = 5
    env_cfg.episode_length_s = 3.0
    env_cfg.place_within_start = env_cfg.place_within  # the real 8 mm tolerance throughout
    env_cfg.scene.cam = CameraCfg(
        prim_path="/World/FilmCam", update_period=0, height=720, width=1280, data_types=["rgb"],
        renderer_cfg=IsaacRtxRendererCfg(enable_scene_partitioning=False),  # one camera must see every arm
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 500.0)),
    )
    agent_cfg = MovePPORunnerCfg()
    with launch_simulation(env_cfg, args_cli):
        from PIL import Image, ImageDraw, ImageFont

        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.utils import to_dict

        from flex_tasks.move_env import MoveEnv

        env = MoveEnv(env_cfg)
        dev = env.device
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        policies = {}
        for name, ckpt in (("before", args_cli.before), ("after", args_cli.after)):
            runner = OnPolicyRunner(wrapped, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            runner.load(os.path.expanduser(ckpt))
            policies[name] = (runner, runner.get_inference_policy(device=dev))
        cam = env.scene["cam"]
        mat = lambda c, e=0.3: sim_utils.PreviewSurfaceCfg(diffuse_color=c, emissive_color=tuple(e * v for v in c))
        marks = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/film", markers={
            "finger": sim_utils.CylinderCfg(radius=0.0095, height=0.08, visual_material=mat((0.86, 0.52, 0.38), 0.15)),
            "tube": sim_utils.SphereCfg(radius=0.0075, visual_material=mat((0.30, 0.33, 0.42), 0.1)),
            "loop_wait": sim_utils.SphereCfg(radius=0.014, visual_material=mat((0.55, 0.57, 0.62), 0.1)),
            "loop_carry": sim_utils.SphereCfg(radius=0.014, visual_material=mat((1.0, 0.78, 0.10))),
            "loop_done": sim_utils.SphereCfg(radius=0.014, visual_material=mat((0.15, 0.85, 0.45))),
            "loop_fail": sim_utils.SphereCfg(radius=0.014, visual_material=mat((1.0, 0.18, 0.12))),
            "target": sim_utils.SphereCfg(radius=0.006, visual_material=mat((0.15, 0.85, 0.45))),
            "mat": sim_utils.CuboidCfg(size=(0.30, 0.42, 0.002), visual_material=mat((0.33, 0.27, 0.22), 0.0)),  # a dark mat under each bench
        }))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
            small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except OSError:
            font = small = ImageFont.load_default()
        o = env.scene.env_origins
        gap = env_cfg.scene.env_spacing
        # start on the arm at one corner; the camera is inside the field, in front of the hand, looking out to it
        corner = o[(o[:, 0] + o[:, 1]).argmax()].clone()
        view = torch.tensor([-0.74, -0.40, 0.54], device=dev)

        def framing(k: float):
            aim = corner + torch.tensor([-(k - 1) / 2 * gap + 0.08, -(k - 1) / 2 * gap + 0.42, 0.07], device=dev)
            return aim, 1.05 + 1.05 * (k - 1) * gap

        N_TUBE = 7
        up = torch.tensor([0.0, 0.0, 1.0], device=dev)
        fx = torch.tensor([1.0, 0.0, 0.0], device=dev) * env_cfg.finger_gap
        os.makedirs(f"{OUT}/frames", exist_ok=True)
        with torch.inference_mode():
            env.reset()
            obs = wrapped.get_observations()
        frame, k_prev = 0, 1.0
        done_total, ep_total = 0.0, 0.0
        for seconds, k, which, caption in STAGES:
            k = float(k or args_cli.side)
            policy = policies[which][1]
            for i in range(int(seconds / env.step_dt)):
                with torch.inference_mode():
                    obs, _, _, _ = wrapped.step(policy(obs))
                if i % 2:
                    continue
                p = min(1.0, i * env.step_dt / MOVE_S)
                p = p * p * (3 - 2 * p)
                k_now = math.exp(math.log(k_prev) + (math.log(k) - math.log(k_prev)) * p)
                aim, dist = framing(k_now)
                cam.set_world_poses_from_view((aim + view * dist).unsqueeze(0), aim.unsqueeze(0))
                grow = max(1.0, dist / 5.0)
                ring, loop, dst = env.ring + o, env.loop + o, env.dst + o
                f0 = env.finger0 + o
                fingers = torch.cat([f0 + fx * j for j in range(4)])
                fingers[:, 2] = o[0, 2] + 0.04
                tube = torch.cat([ring + (loop - ring) * (0.08 + 0.80 * j / (N_TUBE - 1)) for j in range(N_TUBE)])
                state = torch.full((n,), 2, dtype=torch.long, device=dev)  # waiting
                state = torch.where(env.held, torch.full_like(state, 3), state)
                state = torch.where(env.placed, torch.full_like(state, 4), state)
                state = torch.where(env.dropped | env.snagged, torch.full_like(state, 5), state)
                mats = o + torch.tensor([0.03, 0.42, 0.001], device=dev)
                pts = torch.cat([fingers, tube, loop, dst + up * 0.052, mats])
                idx = torch.cat([torch.zeros(4 * n, dtype=torch.long, device=dev), torch.ones(N_TUBE * n, dtype=torch.long, device=dev),
                                 state, torch.full((n,), 6, dtype=torch.long, device=dev), torch.full((n,), 7, dtype=torch.long, device=dev)])
                scl = torch.cat([torch.ones(4 * n, 3, device=dev), torch.full((N_TUBE * n, 3), min(grow, 1.6), device=dev),
                                 torch.full((n, 3), grow, device=dev), torch.full((n, 3), min(grow, 2.0), device=dev),
                                 torch.ones(n, 3, device=dev)])
                marks.visualize(translations=pts, scales=scl, marker_indices=idx)
                im = Image.fromarray(cam.data.output["rgb"][0, ..., :3].cpu().numpy())
                canvas = Image.new("RGB", (im.width, im.height + 110), (14, 20, 31))
                canvas.paste(im, (0, 110))
                dr = ImageDraw.Draw(canvas)
                dr.text((28, 14), caption.format(n=f"{n:,}"), font=font, fill=(244, 247, 251))
                share = float(env.placed.float().mean()) * 100
                dr.text((28, 66), f"loop: grey waiting, yellow carried, green on target, red failed    "
                        f"on target now: {share:4.1f} %    simulated", font=small, fill=(150, 230, 190))
                canvas.save(f"{OUT}/frames/frame_{frame:04d}.png")
                frame += 1
            k_prev = k
        print("[FILM] wrote", frame, "frames to", OUT, flush=True)
        env.close()


if __name__ == "__main__":
    main()
