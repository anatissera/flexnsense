"""Measure a trained pull-arm policy against "just hold still", on the same set of pulls, and optionally film it.

    source ~/isaac-env.sh && cd ~/IsaacLab
    PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/eval_pull.py --checkpoint <model.pt>
    ... --video            # 16 arms, films arm 0 into <out>/frames/

Writes <out>/summary.json, <out>/trace.csv (arm 0, every step, both runs) and the exported policy (policy.pt, policy.onnx).
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--video", action="store_true")
parser.add_argument("--hz10", action="store_true", help="The 10-decisions-a-second variant (Flex-Pull10-v0).")
parser.add_argument("--bench", action="store_true", help="The full bench (Flex-PullBench-v0): table, hand, scale.")
parser.add_argument("--grid", type=int, default=0, help="Film this many arms at once from one camera above the field (e.g. 64).")
parser.add_argument("--pull_back", action="store_true", help="With --grid: start close on a few arms and pull back to show them all.")
parser.add_argument("--title", default=None, help="Caption for the policy's run in the video.")
parser.add_argument("--only_policy", action="store_true", help="Skip the comparison run with the arm held still.")
parser.add_argument("--out", default="~/flexnsense-sim/runs/rl_pull/eval")
add_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
    args_cli.num_envs = args_cli.grid or 16

import csv
import json
import os

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from flex_tasks.agents import PullPPORunnerCfg
from flex_tasks.pull_env_cfg import PullEnvCfg

OUT = os.path.expanduser(args_cli.out)


def main():
    from flex_tasks.pull_env_cfg import Pull10EnvCfg

    from flex_tasks.pull_env_cfg import PullBenchEnvCfg

    env_cfg = PullBenchEnvCfg() if args_cli.bench else (Pull10EnvCfg() if args_cli.hz10 else PullEnvCfg())
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 7
    if args_cli.video:
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        import isaaclab.sim as sim_utils
        from isaaclab.sensors.camera import CameraCfg

        env_cfg.scene.cam = CameraCfg(
            prim_path="/World/FieldCam" if args_cli.grid else "{ENV_REGEX_NS}/Cam", update_period=0,
            height=720 if args_cli.grid else 540, width=1280 if args_cli.grid else 960, data_types=["rgb"],
            renderer_cfg=IsaacRtxRendererCfg(),
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                                             clipping_range=(0.01, 100.0)),
        )
    agent_cfg = PullPPORunnerCfg()
    with launch_simulation(env_cfg, args_cli):
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.utils import to_dict

        import isaaclab.sim as sim_utils
        from flex_tasks.pull_env import PullEnv

        env = PullEnv(env_cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
        runner.load(os.path.expanduser(args_cli.checkpoint))
        policy = runner.get_inference_policy(device=env.device)
        os.makedirs(OUT, exist_ok=True)
        try:
            runner.export_policy_to_jit(path=OUT, filename="policy.pt")
            runner.export_policy_to_onnx(path=OUT, filename="policy.onnx")
        except Exception as err:  # the export is a convenience; the measurement does not depend on it
            print("[EVAL] export failed:", err, flush=True)
        cam = env.scene["cam"] if args_cli.video else None
        if cam is not None:
            from PIL import Image, ImageDraw, ImageFont

            o = env.scene.env_origins
            eye = o + torch.tensor([0.62, -0.10, 0.42] if args_cli.bench else [0.55, -0.35, 0.35], device=env.device)
            look = o + torch.tensor([0.0, 0.36, 0.05] if args_cli.bench else [0.0, 0.32, 0.06], device=env.device)
            if args_cli.grid:  # one camera looking across the whole field of arms
                lo, hi = o.min(dim=0).values, o.max(dim=0).values
                mid, span = (lo + hi) / 2, float((hi - lo).max())
                eye = (mid + torch.tensor([-0.75 * span, -0.75 * span, 0.55 * span], device=env.device)).unsqueeze(0)
                look = (mid + torch.tensor([0.0, 0.0, 0.1], device=env.device)).unsqueeze(0)
            cam.set_world_poses_from_view(eye, look)
            mat = lambda c: sim_utils.PreviewSurfaceCfg(diffuse_color=c)
            marks = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/scale", markers={
                "finger": sim_utils.SphereCfg(radius=0.012, visual_material=mat((0.93, 0.72, 0.60))),
                "scale": sim_utils.SphereCfg(radius=0.005, visual_material=mat((0.8, 0.82, 0.85))),
                "home": sim_utils.SphereCfg(radius=0.006, visual_material=mat((0.1, 0.6, 0.9))),
            }))
            idx = torch.tensor([0] + [1] * 14 + [2], device=env.device)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
                small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
            except OSError:
                font = small = ImageFont.load_default()
        steps = int(env.max_episode_length) - 2
        summary, traces, frame = {}, {}, 0
        for mode in ("policy",) if args_cli.only_policy else ("policy", "hold"):
            torch.manual_seed(7)
            with torch.inference_mode():
                env.reset()
                obs = wrapped.get_observations()
            peak = torch.zeros(env.num_envs, device=env.device)
            over = torch.zeros_like(peak)
            give = torch.zeros_like(peak)
            drift_idle, idle_n = torch.zeros_like(peak), torch.zeros_like(peak)
            rows = []
            for i in range(steps):
                with torch.inference_mode():
                    act = policy(obs) if mode == "policy" else torch.zeros(env.num_envs, 5, device=env.device)
                    if mode == "policy" and i == 30:  # a sample of what goes in and what comes out, to check the exported file
                        import numpy as np

                        o_ = obs["policy"] if not torch.is_tensor(obs) else obs
                        np.savez(f"{OUT}/io_sample.npz", obs=o_[:64].cpu().numpy(), action=act[:64].cpu().numpy())
                    obs, _, _, _ = wrapped.step(act)
                f = env.force
                along, side, height, _ = env._rel()
                peak = torch.maximum(peak, f)
                over += (f > 1.1 * env_cfg.cap).float() * env.step_dt
                give = torch.maximum(give, along)
                idle = (env._pull() < 1e-4).float()  # the finger is not pulling: the arm should be at its place
                drift_idle += idle * along.abs()
                idle_n += idle
                rows.append((round(i * env.step_dt, 4), round(float(f[0]), 4), round(float(env._pull()[0]), 4),
                             round(float(along[0]), 4), round(float(side[0]), 4), round(float(height[0]), 4)))
                if cam is not None and (i % 2 == 0 or args_cli.bench or args_cli.hz10):
                    if args_cli.grid and args_cli.pull_back:
                        p_ = min(1.0, i / (0.8 * steps))
                        p_ = p_ * p_ * (3 - 2 * p_)
                        near_, far_ = 2.5, 0.8 * span  # metres from the point looked at
                        dist_ = near_ * (far_ / near_) ** p_
                        corner = lo + (mid - lo) * (0.25 + 0.75 * p_)  # drift from near a corner to the middle of the field
                        aim = corner + torch.tensor([0.0, 0.0, 0.1], device=env.device)
                        cam.set_world_poses_from_view(
                            (aim + torch.tensor([-0.62, -0.62, 0.48], device=env.device) * dist_).unsqueeze(0), aim.unsqueeze(0))
                    tip, _ = env._tip()
                    finger = env.home[0] + env.u[0] * (env_cfg.rest + env._pull()[0])
                    line = torch.stack([tip[0] + (finger - tip[0]) * k / 13 for k in range(14)])
                    if args_cli.grid:  # every arm's finger and scale
                        fingers = env.home + env.u * (env_cfg.rest + env._pull().unsqueeze(-1))
                        pts = torch.cat([fingers] + [tip + (fingers - tip) * k / 5 for k in range(1, 5)])
                        gidx = torch.cat([torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
                                          torch.ones(4 * env.num_envs, dtype=torch.long, device=env.device)])
                        marks.visualize(translations=pts, marker_indices=gidx)
                    else:
                        marks.visualize(translations=torch.cat([finger.unsqueeze(0), line, env.home[0:1]]), marker_indices=idx)
                    im = Image.fromarray(cam.data.output["rgb"][0, ..., :3].cpu().numpy())
                    canvas = Image.new("RGB", (im.width, im.height + 100), (14, 20, 31))
                    canvas.paste(im, (0, 100))
                    dr = ImageDraw.Draw(canvas)
                    title = (args_cli.title or "Trained policy: gives way above the limit") if mode == "policy" else "No policy: the arm holds still"
                    dr.text((24, 12), title, font=font, fill=(244, 247, 251))
                    fv = float(f.max()) if args_cli.grid else float(f[0])
                    what = f"{env.num_envs} arms, highest force {fv:4.2f} N" if args_cli.grid else f"force {fv:4.2f} N ({fv / 9.80665 * 1000:3.0f} gf)"
                    dr.text((24, 58), f"{what}    limit {env_cfg.cap:.2f} N (200 gf)    simulated, t = {i * env.step_dt:4.1f} s",
                            font=small, fill=(255, 110, 100) if fv > 1.05 * env_cfg.cap else (150, 230, 190))
                    os.makedirs(f"{OUT}/frames", exist_ok=True)
                    canvas.save(f"{OUT}/frames/frame_{frame:04d}.png")
                    frame += 1
            held = env.k * env.amp
            mean = lambda v: round(float(v.mean()), 4)
            summary[mode] = {
                "arms": env.num_envs,
                "limit_N": env_cfg.cap,
                "peak_force_N_mean": mean(peak),
                "peak_force_N_worst": round(float(peak.max()), 4),
                "share_of_pulls_over_110pct_limit": mean((peak > 1.1 * env_cfg.cap).float()),
                "seconds_over_110pct_limit_mean": mean(over),
                "give_m_mean": mean(give),
                "drift_when_not_pulled_m_mean": mean(drift_idle / idle_n.clamp(min=1)),
                "peak_force_if_rigid_N_mean": mean(held),
                "share_of_pulls_that_would_exceed_limit_if_rigid": mean((held > env_cfg.cap).float()),
            }
            traces[mode] = rows
            print(f"[EVAL] {mode}:", json.dumps(summary[mode]), flush=True)
        with open(f"{OUT}/summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        with open(f"{OUT}/trace.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run", "t_s", "force_N", "finger_pull_m", "gripper_along_m", "gripper_side_m", "gripper_up_m"])
            for mode, rows in traces.items():
                w.writerows([(mode, *r) for r in rows])
        print("[EVAL] wrote", OUT, "frames:", frame, flush=True)
        env.close()


if __name__ == "__main__":
    main()
