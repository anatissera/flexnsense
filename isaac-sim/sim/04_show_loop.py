"""Phase 2, made visible: the same force loop as 03_yield_loop.py, drawn and recorded.

Runs ON the workstation:

    source ~/isaac-env.sh && cd ~/IsaacLab
    uv run --extra isaacsim python ~/flexnsense_sim/04_show_loop.py --tag yield            # records frames
    uv run --extra isaacsim python ~/flexnsense_sim/04_show_loop.py --no_yield --tag hold
    LIVESTREAM=1 uv run --extra isaacsim python ~/flexnsense_sim/04_show_loop.py --cycles 100000 --no_record --viz kit
                                                                                         # live, for the WebRTC client

What is drawn (visual only, no physics): the fingertip (skin-coloured ball), the tether and scale (grey dots), the
scale's marker (red ball, its place along the line is force / 600 gf) and the limit tick (dark ball at 200 gf).
Each recorded frame gets a caption with the true force. Frames go to ~/flexnsense-sim/runs/show/<tag>/.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Force-limited pull on the ElRobot.")
parser.add_argument("--no_yield", action="store_true", help="Baseline: the arm holds its position.")
parser.add_argument("--cap", type=float, default=1.96, help="Force limit [N] (1.96 N = 200 gf).")
parser.add_argument("--gain", type=float, default=0.08, help="Admittance gain B [m/s per N].")
parser.add_argument("--pull", type=float, default=0.08, help="How far the finger pulls away [m].")
parser.add_argument("--tag", default=None, help="Output folder name under runs/show/ (default: yield or hold).")
parser.add_argument("--cycles", type=int, default=2, help="Pull cycles to run.")
parser.add_argument("--no_record", action="store_true", help="Do not save frames (for the live stream).")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

import csv
import json
import math
import os

import torch
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors.camera.utils import save_images_to_file
from isaaclab.utils import instantiate
from isaaclab.utils.math import subtract_frame_transforms

ROOT = os.path.expanduser("~/flexnsense-sim")
USD = f"{ROOT}/assets/elrobot/usd/elrobot_follower.usd/elrobot_follower/elrobot_follower.usda"
OUT = f"{ROOT}/runs/show/{args_cli.tag or ('hold' if args_cli.no_yield else 'yield')}"

K_SPRING = 5.886 / 0.10  # N/m: 600 gf full scale over an ASSUMED 0.10 m of travel
REST = 0.10  # m: tether + scale length at zero force
SETTLE, PULL_T, HOLD_T, RELAX_T, REST_T = 1.5, 2.5, 2.5, 1.5, 1.5  # seconds
FULL_SCALE = 5.886  # N: the PESOLA reads up to 600 gf

ELROBOT_CFG = ArticulationCfg(
    prim_path="/World/ElRobot",
    spawn=sim_utils.UsdFileCfg(usd_path=USD, fix_root_link=True),  # bolted to the bench
    actuators={
        "servos": ImplicitActuatorCfg(
            joint_names_expr=[".*"], joint_effort_limit=2.94, joint_velocity_limit=4.71, stiffness=40.0, damping=2.0
        )
    },
)


def ease(p: float) -> float:
    p = min(1.0, max(0.0, p))
    return 2 * p * p if p < 0.5 else 1 - (-2 * p + 2) ** 2 / 2


def pull_at(t: float) -> float:
    """Finger travel away from the gripper [m] at time t after settling."""
    cycle = PULL_T + HOLD_T + RELAX_T + REST_T
    t = t % cycle
    if t < PULL_T:
        return args_cli.pull * ease(t / PULL_T)
    if t < PULL_T + HOLD_T:
        return args_cli.pull
    if t < PULL_T + HOLD_T + RELAX_T:
        return args_cli.pull * (1 - ease((t - PULL_T - HOLD_T) / RELAX_T))
    return 0.0


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/defaultGroundPlane", cfg)
        cfg = sim_utils.DomeLightCfg(intensity=900.0, color=(0.9, 0.93, 1.0))
        cfg.func("/World/Light", cfg)
        robot = Articulation(cfg=ELROBOT_CFG)
        sim_utils.create_prim("/World/CamRig", "Xform")
        camera = instantiate(
            CameraCfg(
                prim_path="/World/CamRig/Camera",
                update_period=0,
                height=720,
                width=1280,
                data_types=["rgb"],
                renderer_cfg=IsaacRtxRendererCfg(),
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0)
                ),
            )
        )
        mat = lambda c: sim_utils.PreviewSurfaceCfg(diffuse_color=c)
        rig = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/rig",
                markers={
                    "finger": sim_utils.SphereCfg(radius=0.011, visual_material=mat((0.93, 0.72, 0.60))),
                    "tether": sim_utils.SphereCfg(radius=0.0035, visual_material=mat((0.55, 0.58, 0.62))),
                    "marker": sim_utils.SphereCfg(radius=0.008, visual_material=mat((0.90, 0.12, 0.10))),
                    "limit": sim_utils.SphereCfg(radius=0.005, visual_material=mat((0.05, 0.07, 0.10))),
                },
            )
        )
        N_DOTS = 16
        marker_idx = torch.tensor([0] + [1] * N_DOTS + [2, 3], device=sim.device)
        EYE, LOOK = [[0.22, -0.62, 0.30]], [[0.22, 0.0, 0.13]]
        sim.set_camera_view(EYE[0], LOOK[0])  # the Kit viewport (and the live stream) use the same view
        sim.reset()
        camera.set_world_poses_from_view(
            torch.tensor(EYE, device=sim.device), torch.tensor(LOOK, device=sim.device)
        )
        os.makedirs(OUT, exist_ok=True)
        dev, dt = sim.device, sim.get_physics_dt()

        arm_ids, arm_names = robot.find_joints("rev_motor_0[1-7]", preserve_order=True)
        ee_ids, ee_names = robot.find_bodies("Gripper_Base_v1_1")
        ee_id = ee_ids[0]
        jac_body = ee_id - 1 if robot.is_fixed_base else ee_id
        jac_joints = [j + robot.num_base_dofs for j in arm_ids]
        print("[LOOP] arm joints:", arm_names, "| gripper body:", ee_names, "| fixed base:", robot.is_fixed_base, "| dt:", dt)

        ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
            num_envs=1,
            device=dev,
        )

        # a bent start pose (the pose of phase 1 at t = 0), held until the arm has settled
        limits = robot.data.joint_pos_limits.torch[0]
        mid, half = (limits[:, 0] + limits[:, 1]) / 2, (limits[:, 1] - limits[:, 0]) / 2
        start = (mid + 0.2 * half * torch.sin(torch.arange(robot.num_joints, device=dev) * 0.6)).unsqueeze(0)

        def ee_state():
            pose = robot.data.body_pose_w.torch[:, ee_id]
            root = robot.data.root_pose_w.torch
            pos_b, quat_b = subtract_frame_transforms(root[:, 0:3], root[:, 3:7], pose[:, 0:3], pose[:, 3:7])
            return pose[:, 0:3].clone(), pose[:, 3:7].clone(), pos_b, quat_b, root

        def step_sim():
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)
            camera.update(dt=dt)

        for _ in range(int(SETTLE / dt)):
            robot.set_joint_position_target_index(target=start)
            step_sim()

        p0, _, p0_b, _, root = ee_state()
        # pull direction: horizontal, from the base towards the gripper
        u = p0[0].clone() - root[0, 0:3]
        u[2] = 0.0
        u = u / u.norm()
        # look at the rig from the side, square to the pull direction
        side = torch.stack([-u[1], u[0], torch.zeros((), device=dev)])
        centre = p0[0] + u * 0.02
        eye = centre + side * 0.58 + torch.tensor([0.0, 0.0, 0.12], device=dev)
        camera.set_world_poses_from_view(eye.unsqueeze(0), centre.unsqueeze(0))
        sim.set_camera_view(eye.tolist(), centre.tolist())
        print("[LOOP] pull direction:", [round(float(v), 3) for v in u], "| gripper at:", [round(float(v), 3) for v in p0[0]])
        target_b = p0_b.clone()  # gripper target in the base frame (the root is at the origin, unrotated)
        joint_des = robot.data.joint_pos.torch[:, arm_ids].clone()
        other_ids = [j for j in range(robot.num_joints) if j not in arm_ids]
        other_des = start[:, other_ids].clone()

        from PIL import Image, ImageDraw, ImageFont

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
            small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except OSError:
            font = small = ImageFont.load_default()
        mode = "Arm holds (no yielding)" if args_cli.no_yield else "Arm gives way above the limit"
        rows, frames = [], 0
        n_steps = int(args_cli.cycles * (PULL_T + HOLD_T + RELAX_T + REST_T) / dt)
        shot_every = 2  # 30 pictures per simulated second
        for step in range(n_steps):
            t = step * dt
            p, _, p_b, q_b, _ = ee_state()
            finger = p0[0] + u * (REST + pull_at(t))
            d = finger - p[0]
            dist = d.norm()
            force = K_SPRING * max(0.0, float(dist) - REST)
            dirn = d / dist
            # the spring pulls the gripper towards the finger
            robot.permanent_wrench_composer.set_forces_and_torques_index(
                forces=(dirn * force).reshape(1, 1, 3), body_ids=[ee_id], is_global=True
            )
            # admittance: give way while the force is above the limit
            if not args_cli.no_yield and force > args_cli.cap:
                target_b[0] += dirn * args_cli.gain * (force - args_cli.cap) * dt
            ik.set_command(target_b, ee_quat=q_b)
            jac = robot.data.body_link_jacobian_w.torch[:, jac_body, :, jac_joints]
            joint_des = ik.compute(p_b, q_b, jac, robot.data.joint_pos.torch[:, arm_ids])
            robot.set_joint_position_target_index(target=joint_des, joint_ids=arm_ids)
            robot.set_joint_position_target_index(target=other_des, joint_ids=other_ids)
            step_sim()
            give = float(torch.dot(p[0] - p0[0], u))
            rows.append((round(t, 4), round(force, 4), round(pull_at(t), 4), round(give, 5)))
            # draw the rig: fingertip, tether dots, the scale's marker and the limit tick
            fr = torch.linspace(0.46, 0.94, N_DOTS, device=dev).unsqueeze(1)  # from the gripper tip to the finger
            dots = p[0] + fr * d
            at = lambda f: (p[0] + d * (0.50 + 0.40 * min(1.0, f / FULL_SCALE))).unsqueeze(0)
            rig.visualize(
                translations=torch.cat([finger.unsqueeze(0), dots, at(force), at(args_cli.cap)]), marker_indices=marker_idx
            )
            if step % shot_every == 0 and not args_cli.no_record:
                rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
                im = Image.fromarray(rgb)
                dr = ImageDraw.Draw(im)
                over = force > args_cli.cap * 1.05
                dr.rectangle([0, 0, im.width, 118], fill=(14, 20, 31))
                dr.text((28, 14), mode, font=font, fill=(244, 247, 251))
                dr.text(
                    (28, 66),
                    f"force {force:4.2f} N  ({force / 9.80665 * 1000:3.0f} gf)     limit {args_cli.cap:.2f} N (200 gf)     t = {t:4.1f} s",
                    font=small,
                    fill=(255, 110, 100) if over else (150, 230, 190),
                )
                im.save(f"{OUT}/frame_{frames:04d}.png")
                frames += 1

        with open(f"{OUT}/log.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "force_N", "finger_pull_m", "gripper_give_m"])
            w.writerows(rows)
        forces = [r[1] for r in rows]
        over = [r for r in rows if r[1] > args_cli.cap * 1.1]
        summary = {
            "mode": "hold" if args_cli.no_yield else "yield",
            "cap_N": args_cli.cap,
            "gain_m_per_s_per_N": args_cli.gain,
            "spring_N_per_m": round(K_SPRING, 3),
            "finger_pull_m": args_cli.pull,
            "physics_dt_s": dt,
            "max_force_N": round(max(forces), 3),
            "max_force_gf": round(max(forces) / 9.80665 * 1000, 1),
            "seconds_above_110pct_cap": round(len(over) * dt, 3),
            "max_gripper_give_m": round(max(r[3] for r in rows), 4),
            "frames": frames,
        }
        with open(f"{OUT}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("[LOOP] summary:", json.dumps(summary))


if __name__ == "__main__":
    main()
