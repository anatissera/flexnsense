"""Film and record the physical-contact task with the bench's two cameras.

    source ~/isaac-env.sh && cd ~/IsaacLab
    PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/film_phys.py \\
        --checkpoint <model.pt> --episodes 4 --out ~/flexnsense-sim/runs/phys/film
    ... --record 40      # also save 40 episodes (joints + the two camera pictures of every frame) for the dataset

The two cameras are the ones on the real bench: the overhead camera that looks down on the scale (no arm in its way),
and the camera on the move arm. The film shows a third, outside view of the bench beside them, for the viewer only.
Films table 0. Writes <out>/frames/frame_*.png, and with --record <out>/episodes/ep_*.npz (jpegs inside).
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=21)
parser.add_argument("--record", type=int, default=0, help="episodes to save with pictures (they replace the film)")
parser.add_argument("--untrained", action="store_true")
parser.add_argument("--stage", default="", help="text for the top right corner, e.g. 'after 20 min of training'")
parser.add_argument("--out", default="~/flexnsense-sim/runs/phys/film")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

import io
import os

import numpy as np
import torch
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import CameraCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from flex_tasks.agents import PhysPPORunnerCfg
from flex_tasks.phys_env_cfg import PhysEnvCfg

OUT = os.path.expanduser(args_cli.out)
STAGE_TEXT = ["Move arm: taking the scale from the table to the INDEX finger",
              "Pull arm: pulling until the scale shows 200 g, then back to zero",
              "Move arm: moving the loop from the index to the MIDDLE finger",
              "Pull arm: pulling until the scale shows 200 g, then back to zero"]


def cam(path):
    return CameraCfg(prim_path=path, update_period=0, height=540, width=640, data_types=["rgb"],
                     renderer_cfg=IsaacRtxRendererCfg(enable_scene_partitioning=False),
                     spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                                                      clipping_range=(0.01, 100.0)))


def main():
    env_cfg = PhysEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.start_on_index_share = 0.0  # every filmed episode is the whole sequence, from the table
    env_cfg.scene.light.spawn = sim_utils.DomeLightCfg(intensity=900.0, color=(0.9, 0.93, 1.0))
    env_cfg.scene.cam_bench, env_cfg.scene.cam_top, env_cfg.scene.cam_wrist = cam("/World/CamBench"), cam("/World/CamTop"), cam("/World/CamWrist")
    agent_cfg = PhysPPORunnerCfg()
    with launch_simulation(env_cfg, args_cli):
        from PIL import Image, ImageDraw, ImageFont

        from isaaclab.utils import to_dict
        from isaaclab.utils.math import matrix_from_quat

        from flex_tasks.phys_env import PhysEnv

        env = PhysEnv(env_cfg)
        dev = env.device
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
        if not args_cli.untrained:
            runner.load(os.path.expanduser(args_cli.checkpoint))
        policy = runner.get_inference_policy(device=dev)
        i0, e0 = 0, env.scene.env_origins[0]
        T = lambda v: torch.tensor(v, device=dev)
        env.scene["cam_bench"].set_world_poses_from_view((e0 + T([-0.50, 0.05, 0.44])).unsqueeze(0), (e0 + T([0.08, 0.36, 0.04])).unsqueeze(0))
        env.scene["cam_top"].set_world_poses_from_view((e0 + T([0.0, 0.415, 0.50])).unsqueeze(0), (e0 + T([0.0, 0.43, 0.0])).unsqueeze(0))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
            small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except OSError:
            font = small = ImageFont.load_default()
        record = args_cli.record > 0
        os.makedirs(f"{OUT}/{'episodes' if record else 'frames'}", exist_ok=True)
        jaw = int(env.jaw_ids[1])
        arm = env.arm_ids
        lim = env.limits
        lo = torch.cat([lim[arm, 0], lim[jaw : jaw + 1, 0]]).cpu().numpy()
        hi = torch.cat([lim[arm, 1], lim[jaw : jaw + 1, 1]]).cpu().numpy()
        a_arm, a_jaw = env.a_arm, int(env.A.find_joints("rev_motor_08_2")[0][0])
        la = env.A.data.joint_pos_limits.torch[0]
        lo_a = torch.cat([la[a_arm, 0], la[a_jaw : a_jaw + 1, 0]]).cpu().numpy()
        hi_a = torch.cat([la[a_arm, 1], la[a_jaw : a_jaw + 1, 1]]).cpu().numpy()
        frame, steps, kept, tried = 0, int(env.max_episode_length) - 2, 0, 0
        todo = args_cli.record if record else args_cli.episodes
        jpeg = lambda im: (lambda b: (im.save(b, format="JPEG", quality=88), b.getvalue())[1])(io.BytesIO())
        with torch.inference_mode():
            env.reset()
            obs = wrapped.get_observations()
        while (kept if record else tried) < todo and tried < todo * 4 + 4:
            tried += 1
            rec = {k: [] for k in ("pos_move", "goal_move", "pos_pull", "goal_pull", "force", "stage", "top", "wrist")}
            seen = {"far": 0, "peak": 0.0, "done": False}
            for i in range(steps):
                with torch.inference_mode():
                    act = policy(obs)
                    if args_cli.untrained:
                        act = torch.randn_like(act) * 0.5
                    obs, _, _, _ = wrapped.step(act)
                pb = env.B.data.body_pose_w.torch[i0, env.ee]
                pointing = matrix_from_quat(pb[3:7].unsqueeze(0))[0] @ env.point
                # on a short mast above the gripper, looking past the jaws at what they reach for; the mast leans away
                # from the arm's own body so that the forearm never fills the picture
                flat = pointing.clone()
                flat[2] = 0.0
                flat = flat / flat.norm().clamp(min=1e-6)
                eye = pb[0:3] - flat * 0.05 + T([0.0, 0.0, 0.16])
                env.scene["cam_wrist"].set_world_poses_from_view(eye.unsqueeze(0), (pb[0:3] + pointing * 0.14).unsqueeze(0))
                st, done = int(env.stage[i0]), bool(env.finished[i0])
                force = float(env._where()[5][i0])
                seen["far"], seen["done"] = max(seen["far"], st), seen["done"] or done
                if st in (1, 3):
                    seen["peak"] = max(seen["peak"], force)
                ims = {c: Image.fromarray(env.scene[f"cam_{c}"].data.output["rgb"][0, ..., :3].cpu().numpy()) for c in ("bench", "top", "wrist")}
                if record:
                    qb, qa = env.B.data.joint_pos.torch[i0], env.A.data.joint_pos.torch[i0]
                    pk = lambda q, t, ar, jw: (torch.cat([q[ar], q[jw : jw + 1]]).cpu().numpy(), torch.cat([t[ar], t[jw : jw + 1]]).cpu().numpy())
                    pm, gm = pk(qb, env.qB_tgt[i0], arm, jaw)
                    pp, gp = pk(qa, env.qA_tgt[i0], a_arm, a_jaw)
                    for k, v in (("pos_move", pm), ("goal_move", gm), ("pos_pull", pp), ("goal_pull", gp), ("force", force), ("stage", st)):
                        rec[k].append(v)
                    rec["top"].append(jpeg(ims["top"]))
                    rec["wrist"].append(jpeg(ims["wrist"]))
                else:
                    title = "All done: both fingers treated, move arm back at rest" if done else STAGE_TEXT[st]
                    canvas = Image.new("RGB", (640 * 3, 540 + 118), (14, 20, 31))
                    for k, c in enumerate(("bench", "top", "wrist")):
                        canvas.paste(ims[c], (640 * k, 118))
                    dr = ImageDraw.Draw(canvas)
                    dr.text((28, 14), title, font=font, fill=(244, 247, 251))
                    dr.text((28, 66), f"scale reads {force:4.2f} N  ({force / 9.80665 * 1000:3.0f} g)     the move arm's policy is trained; the grip and the "
                            f"loop are held by contact only     t = {i * env.step_dt:4.1f} s", font=small, fill=(150, 230, 190))
                    if args_cli.stage:
                        w = dr.textlength(args_cli.stage, font=font)
                        dr.text((640 * 3 - 28 - w, 14), args_cli.stage, font=font, fill=(255, 214, 102))
                    for k, name in enumerate(("outside view", "overhead camera", "camera on the move arm")):
                        dr.text((640 * k + 24, 132), name, font=small, fill=(244, 247, 251))
                    canvas.save(f"{OUT}/frames/frame_{frame:04d}.png")
                    frame += 1
                if i > 3 and env.episode_length_buf[i0] <= 1:  # this table's episode ended early and a new one began
                    break
            ok = seen["done"]
            print(f"[FILM] episode {tried}: reached stage {seen['far']} of 3 | whole sequence done: {ok} | highest reading in a pull {seen['peak']:.2f} N", flush=True)
            if record and ok:
                np.savez_compressed(
                    f"{OUT}/episodes/ep_{kept:04d}.npz", lo_move=lo, hi_move=hi, lo_pull=lo_a, hi_pull=hi_a, dt=env.step_dt,
                    **{k: np.stack(rec[k]) for k in ("pos_move", "goal_move", "pos_pull", "goal_pull")},
                    force=np.array(rec["force"], np.float32), stage=np.array(rec["stage"], np.int8),
                    top=np.array(rec["top"], dtype=object), wrist=np.array(rec["wrist"], dtype=object))
                kept += 1
            if not ok or i < steps - 1:
                pass
            with torch.inference_mode():  # start every filmed episode from a fresh reset of all tables
                env.reset()
                obs = wrapped.get_observations()
        print(f"[FILM] wrote {frame} frames; episodes tried {tried}, saved {kept} -> {OUT}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
