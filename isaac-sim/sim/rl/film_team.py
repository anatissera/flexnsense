"""Film the two trained policies working together on the wooden table, in the layout of the earlier simulation video:
bench | overhead camera (reads the scale) | camera on the move arm, with the phase and the force written on top.

    source ~/isaac-env.sh && cd ~/IsaacLab
    PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/film_team.py \\
        --checkpoint ~/IsaacLab/logs/skrl/flex_team/<run>/checkpoints/best_agent.pt --episodes 3
Frames: <out>/frames/ (10 pictures a second of simulated time). Also prints what happened in each filmed episode.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--untrained", action="store_true", help="Do not load a checkpoint: what the arms do before any learning.")
parser.add_argument("--stage", default="", help="A line shown at the top right, e.g. 'after 20 minutes of learning'.")
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--seed", type=int, default=21, help="which tables and pulls are drawn")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--out", default="~/flexnsense-sim/runs/team/film")
parser.add_argument("--repo", default="~/flexnsense-sim/stage/repo",
                    help="a copy of the team repository's gauge_reader/ and lib/: its sticker reader is run on the overhead camera's pictures")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

import os

import torch
import yaml
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import CameraCfg

import flex_tasks
from flex_tasks.team_env_cfg import TeamEnvCfg

OUT = os.path.expanduser(args_cli.out)


def cam(path):
    return CameraCfg(prim_path=path, update_period=0, height=540, width=640, data_types=["rgb"],
                     renderer_cfg=IsaacRtxRendererCfg(enable_scene_partitioning=False),
                     spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                                                      clipping_range=(0.01, 100.0)))


def main():
    env_cfg = TeamEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.place_within_start = env_cfg.place_within  # the real 8 mm tolerance
    env_cfg.scene.light.spawn = sim_utils.DomeLightCfg(intensity=900.0, color=(0.9, 0.93, 1.0))  # as in the earlier video
    env_cfg.scene.cam_bench, env_cfg.scene.cam_top, env_cfg.scene.cam_wrist = cam("/World/CamBench"), cam("/World/CamTop"), cam("/World/CamWrist")
    with launch_simulation(env_cfg, args_cli):
        from PIL import Image, ImageDraw, ImageFont
        from skrl.utils.runner.torch import Runner

        from isaaclab.utils.math import matrix_from_quat
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
        runner = Runner(wrapped, agent_cfg)
        if not args_cli.untrained:
            runner.agent.load(os.path.expanduser(args_cli.checkpoint))
        runner.agent.enable_training_mode(False, apply_to_models=True)
        marks = make_markers()
        # the team's own scale reader, on the simulated overhead camera
        import sys

        import numpy as np

        sys.path.insert(0, os.path.expanduser(args_cli.repo))
        try:
            from arm1.gauge_reader.stickers import StickerConfig, StickerTracker

            from flex_tasks.bench_draw import U0, U1

            tracker = StickerTracker(StickerConfig(u_range=(U0 - 0.05, U1 + 0.05)))
        except Exception as err:  # the film still works without it
            print("[FILM] gauge reader not available:", err, flush=True)
            tracker = None
        read = {"n": 0, "found": 0, "err": [], "why": {}}
        o = env.scene.env_origins
        i0 = 0
        e0 = o[i0]
        T = lambda v: torch.tensor(v, device=dev)
        env.scene["cam_bench"].set_world_poses_from_view((e0 + T([-0.50, 0.05, 0.44])).unsqueeze(0), (e0 + T([0.08, 0.36, 0.04])).unsqueeze(0))
        env.scene["cam_top"].set_world_poses_from_view((e0 + T([0.0, 0.415, 0.50])).unsqueeze(0), (e0 + T([0.0, 0.43, 0.0])).unsqueeze(0))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
            small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except OSError:
            font = small = ImageFont.load_default()
        os.makedirs(f"{OUT}/frames", exist_ok=True)
        obs, _ = wrapped.reset()
        states = wrapped.state()
        frame, steps = 0, int(env.max_episode_length) - 1
        for ep in range(args_cli.episodes):
            start = "the loop lies on the table" if bool(env.free[i0]) else "the loop is on a finger"
            seen = {"placed": False, "failed": "", "peak": 0.0}
            for i in range(steps):
                with torch.inference_mode():
                    out = runner.agent.act(obs, states, timestep=0, timesteps=0)
                    actions = {a: out[-1][a].get("mean_actions", out[0][a]) for a in env.cfg.possible_agents}
                    if args_cli.untrained:  # before learning the arms act at random
                        actions = {a: out[0][a] for a in env.cfg.possible_agents}
                    was = (bool(env.dropped[i0]), bool(env.snagged[i0]), bool(env.late[i0]))
                    obs, _, _, _, _ = wrapped.step(actions)
                    states = wrapped.state()
                draw(env, marks)
                # the camera on the move arm rides with its gripper and looks at what the jaws reach for
                pb = env.B.data.body_pose_w.torch[i0, env.ee]
                pointing = matrix_from_quat(pb[3:7].unsqueeze(0))[0] @ env.point
                eye = pb[0:3] - pointing * 0.07 + T([0.0, 0.0, 0.09])  # above and behind the jaws, looking past them
                env.scene["cam_wrist"].set_world_poses_from_view(eye.unsqueeze(0), (pb[0:3] + pointing * 0.16).unsqueeze(0))
                ph, held, placed = int(env._phase()[i0]), bool(env.held[i0]), bool(env.placed[i0])
                force = float(env.force[i0])
                seen["peak"] = max(seen["peak"], force)
                seen["placed"] |= placed
                if ph == 1 and not placed:
                    title = "Move arm: carrying the loop to the target finger" if held else "Move arm: awake, going to the scale"
                elif ph == 1:
                    title = "Move arm: loop placed, going back to rest"
                else:
                    title = "Pull arm: the finger pulls, the arm gives way above the limit"
                ims = [Image.fromarray(env.scene[c].data.output["rgb"][0, ..., :3].cpu().numpy()) for c in ("cam_bench", "cam_top", "cam_wrist")]
                cam_text = ""
                if tracker is not None:  # what the real reader makes of the overhead picture
                    det = tracker.detect(np.ascontiguousarray(np.asarray(ims[1])[:, :, ::-1]))
                    read["n"] += 1
                    if det.found:
                        # The reader counts from whichever sticker it calls number 1. On the real bench the calibration
                        # clicks fix that; here sticker 1 is the one nearer the hand, which is higher in this picture.
                        u = det.u if det.s1[1] < det.s2[1] else 1.0 - det.u
                        grams = (u - U0) / (U1 - U0) * 600.0
                        read["found"] += 1
                        read["err"].append(grams - force / 9.80665 * 1000)
                        cam_text = f"     gauge reader on the overhead picture: {max(0.0, grams):3.0f} gf"
                    else:
                        read["why"][det.reason] = read["why"].get(det.reason, 0) + 1
                        cam_text = f"     gauge reader: no reading ({det.reason})"
                canvas = Image.new("RGB", (640 * 3, 540 + 118), (14, 20, 31))
                for k, im in enumerate(ims):
                    canvas.paste(im, (640 * k, 118))
                dr = ImageDraw.Draw(canvas)
                dr.text((28, 14), title, font=font, fill=(244, 247, 251))
                dr.text((28, 66), f"force {force:4.2f} N  ({force / 9.80665 * 1000:3.0f} gf)     limit {env_cfg.cap:.2f} N (200 gf)     "
                        f"two trained policies, simulated     t = {i * env.step_dt:4.1f} s{cam_text}", font=small,
                        fill=(255, 110, 100) if force > env_cfg.cap * 1.05 else (150, 230, 190))
                if args_cli.stage:
                    w = dr.textlength(args_cli.stage, font=font)
                    dr.text((640 * 3 - 28 - w, 14), args_cli.stage, font=font, fill=(255, 214, 102))
                for k, name in enumerate(("bench", "overhead camera (reads the scale)", "camera on the move arm")):
                    dr.text((640 * k + 24, 132), name, font=small, fill=(244, 247, 251))
                canvas.save(f"{OUT}/frames/frame_{frame:04d}.png")
                frame += 1
                now = (bool(env.dropped[i0]), bool(env.snagged[i0]), bool(env.late[i0]))
                if i < 3:
                    continue
                if env.episode_length_buf[i0] <= 1:  # this table's episode ended and a new one began
                    seen["failed"] = seen["failed"] or ("ended early" if not seen["placed"] else "")
                    break
            print(f"[FILM] episode {ep + 1}: start: {start} | loop placed: {seen['placed']} | peak force {seen['peak']:.2f} N "
                  f"{'| ' + seen['failed'] if seen['failed'] else ''}", flush=True)
        if tracker is not None and read["n"]:
            e = np.array(read["err"]) if read["err"] else np.zeros(1)
            print(f"[READER] the team's gauge reader on {read['n']} overhead pictures: a reading in {100 * read['found'] / read['n']:.0f} % | "
                  f"error against the true force: mean {e.mean():+.1f} gf, spread {e.std():.1f} gf, worst {np.abs(e).max():.1f} gf | "
                  f"no reading because: {read['why']}", flush=True)
        print("[FILM] wrote", frame, "frames to", OUT, flush=True)
        env.close()


if __name__ == "__main__":
    main()
