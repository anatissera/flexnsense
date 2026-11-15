"""Measure a trained move-arm policy at the real placing tolerance (8 mm), and optionally film it.

    source ~/isaac-env.sh && cd ~/IsaacLab
    PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/eval_move.py --checkpoint <model.pt>
    ... --video            # 16 arms, films arm 0 for three episodes into <out>/frames/

Writes <out>/summary.json and the exported policy (policy.pt, policy.onnx).
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--video", action="store_true")
parser.add_argument("--out", default="~/flexnsense-sim/runs/rl_move/eval")
add_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
    args_cli.num_envs = 16

import json
import os

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from flex_tasks.agents import MovePPORunnerCfg
from flex_tasks.move_env_cfg import MoveEnvCfg

OUT = os.path.expanduser(args_cli.out)
FINGERS = ["index", "middle", "ring", "little"]


def main():
    env_cfg = MoveEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 11
    env_cfg.place_within_start = env_cfg.place_within  # the real tolerance from the first step
    if args_cli.video:
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        import isaaclab.sim as sim_utils
        from isaaclab.sensors.camera import CameraCfg

        env_cfg.scene.cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Cam", update_period=0, height=540, width=960, data_types=["rgb"],
            renderer_cfg=IsaacRtxRendererCfg(),
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                                             clipping_range=(0.01, 100.0)),
        )
    agent_cfg = MovePPORunnerCfg()
    with launch_simulation(env_cfg, args_cli):
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.utils import to_dict

        import isaaclab.sim as sim_utils
        from flex_tasks.move_env import MoveEnv

        env = MoveEnv(env_cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
        runner.load(os.path.expanduser(args_cli.checkpoint))
        policy = runner.get_inference_policy(device=env.device)
        os.makedirs(OUT, exist_ok=True)
        try:
            runner.export_policy_to_jit(path=OUT, filename="policy.pt")
            runner.export_policy_to_onnx(path=OUT, filename="policy.onnx")
        except Exception as err:
            print("[EVAL] export failed:", err, flush=True)
        dev, n = env.device, env.num_envs
        cam = env.scene["cam"] if args_cli.video else None
        if cam is not None:
            from PIL import Image, ImageDraw, ImageFont

            o = env.scene.env_origins
            cam.set_world_poses_from_view(o + torch.tensor([-0.28, 0.22, 0.34], device=dev),
                                          o + torch.tensor([0.05, 0.47, 0.04], device=dev))
            mat = lambda c, m=0.0: sim_utils.PreviewSurfaceCfg(diffuse_color=c, metallic=m)
            skin = (0.93, 0.72, 0.60)
            marks = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/bench", markers={
                "finger": sim_utils.CylinderCfg(radius=0.0095, height=0.08, visual_material=mat(skin)),
                "tube": sim_utils.SphereCfg(radius=0.0075, visual_material=mat((0.80, 0.82, 0.85), 0.5)),
                "loop": sim_utils.SphereCfg(radius=0.013, visual_material=mat((0.62, 0.64, 0.68), 0.7)),
                "ring": sim_utils.SphereCfg(radius=0.008, visual_material=mat((0.15, 0.15, 0.17))),
                "target": sim_utils.SphereCfg(radius=0.007, visual_material=mat((0.1, 0.8, 0.3))),
            }))
            N_TUBE = 24
            idx = torch.tensor([0] * 4 + [1] * N_TUBE + [2, 3, 4], device=dev)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
                small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
            except OSError:
                font = small = ImageFont.load_default()
        episodes = 3 if args_cli.video else 1
        steps = int(env.max_episode_length) - 2
        frame = 0
        placed = torch.zeros(n, device=dev)
        dropped, t_place, lift, drag = torch.zeros_like(placed), torch.zeros_like(placed), torch.zeros_like(placed), torch.zeros_like(placed)
        for ep in range(episodes):
            torch.manual_seed(11 + ep)
            with torch.inference_mode():
                env.reset()
                obs = wrapped.get_observations()
            was_done = torch.zeros(n, dtype=torch.bool, device=dev)
            for i in range(steps):
                with torch.inference_mode():
                    obs, _, _, _ = wrapped.step(policy(obs))
                if ep == 0:
                    alive = ~was_done
                    placed = torch.where(alive, env.placed.float(), placed)
                    t_place = torch.where(alive & env.placed, env.t_place, t_place)
                    lift = torch.where(alive, torch.maximum(lift, env.lifted), lift)
                    drag += alive.float() * env.drag * env.step_dt
                    dropped = torch.maximum(dropped, (env.ev_drop > 0).float())
                    was_done |= (env.ev_drop > 0) | (env.drag > 0)
                if cam is not None and i % 2 == 0:
                    org = env.scene.env_origins[0]
                    ring, loop, dst = env.ring[0] + org, env.loop[0] + org, env.dst[0] + org
                    f0 = env.finger0[0]  # the four fingers stand in a row, 24 mm apart
                    fingers = torch.stack([f0 + torch.tensor([env_cfg.finger_gap * k, 0.0, 0.0], device=dev) for k in range(4)])
                    fingers[:, 2] = 0.04
                    fingers = fingers + org
                    tube = torch.stack([ring + (loop - ring) * (0.09 + 0.78 * k / (N_TUBE - 1)) for k in range(N_TUBE)])
                    marks.visualize(translations=torch.cat([fingers, tube, loop.unsqueeze(0), ring.unsqueeze(0),
                                                            (dst + torch.tensor([0.0, 0.0, 0.05], device=dev)).unsqueeze(0)]),
                                    marker_indices=idx)
                    im = Image.fromarray(cam.data.output["rgb"][0, ..., :3].cpu().numpy())
                    canvas = Image.new("RGB", (im.width, im.height + 100), (14, 20, 31))
                    canvas.paste(im, (0, 100))
                    dr = ImageDraw.Draw(canvas)
                    state = "placed on the target finger" if bool(env.placed[0]) else ("carrying the scale" if bool(env.held[0]) else "reaching for the scale")
                    dr.text((24, 12), f"Trained move policy: {state}", font=font, fill=(244, 247, 251))
                    dr.text((24, 58), f"loop to target {float((env.loop[0] - env.dst[0]).norm()) * 1000:5.1f} mm    green dot = target finger    "
                            f"simulated, episode {ep + 1}, t = {i * env.step_dt:4.1f} s", font=small, fill=(150, 230, 190))
                    os.makedirs(f"{OUT}/frames", exist_ok=True)
                    canvas.save(f"{OUT}/frames/frame_{frame:04d}.png")
                    frame += 1
        mean = lambda v: round(float(v.mean()), 4)
        ok = placed > 0
        summary = {
            "arms": n,
            "place_tolerance_m": env_cfg.place_within,
            "share_placed": mean(placed),
            "share_dropped": mean(dropped),
            "seconds_to_place_mean": round(float(t_place[ok].mean()), 3) if bool(ok.any()) else None,
            "seconds_to_place_slowest": round(float(t_place[ok].max()), 3) if bool(ok.any()) else None,
            "highest_lift_m_mean": mean(lift),
            "share_that_hit_a_finger": mean((drag > 0).float()),
        }
        print("[EVAL]", json.dumps(summary), flush=True)
        with open(f"{OUT}/summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        print("[EVAL] wrote", OUT, "frames:", frame, flush=True)
        env.close()


if __name__ == "__main__":
    main()
