"""The force loop arranged like the real bench: the gripper clamps the top end of the scale from the sides, the scale
lies flat just above the table and points at the finger, which sits in the loop at the far end.

Runs ON the workstation:

    source ~/isaac-env.sh && cd ~/IsaacLab
    uv run --extra isaacsim python ~/flexnsense_sim/05_bench_scene.py --tag yield
    uv run --extra isaacsim python ~/flexnsense_sim/05_bench_scene.py --no_yield --tag hold
    LIVESTREAM=1 uv run --extra isaacsim python ~/flexnsense_sim/05_bench_scene.py --cycles 100000 --no_record --viz kit

Compared with 03/04:
  * The arm first moves the gripper to a low, horizontal pose (jaws closing sideways), then keeps that orientation.
  * The spring force acts at the jaw tips (where the scale's ring is clamped), not at the gripper's origin.
  * The scale is drawn like the PESOLA: ring, tube, rod and loop, with the red marker sliding along the tube and a dark
    tick at the limit. Drawn only: the physics is still a virtual spring, F = K * (stretch of the scale).

Assumed numbers, to replace with measurements: K (600 gf over an ASSUMED 0.10 m of marker travel), the scale's length
(0.23 m ring to loop), the jaw-tip offset, the servo stiffness and damping, the admittance gain.

Writes ~/flexnsense-sim/runs/bench/<tag>/ : frame_*.png, log.csv, summary.json.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Force-limited pull on the ElRobot, arranged like the bench.")
parser.add_argument("--no_yield", action="store_true", help="Baseline: the arm holds its position.")
parser.add_argument("--cap", type=float, default=1.96, help="Force limit [N] (1.96 N = 200 gf).")
parser.add_argument("--gain", type=float, default=1.0, help="Admittance gain B [m/s per N].")
parser.add_argument("--pull", type=float, default=0.06, help="How far the finger pulls away [m].")
parser.add_argument("--tag", default=None, help="Output folder name under runs/bench/ (default: yield or hold).")
parser.add_argument("--cycles", type=int, default=2, help="Pull cycles to run.")
parser.add_argument("--no_record", action="store_true", help="Do not save frames (for the live stream).")
parser.add_argument("--reach", type=float, default=0.20, help="Distance of the gripper from the base [m].")
parser.add_argument("--height", type=float, default=0.05, help="Height of the scale above the table [m].")
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
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import instantiate
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix, subtract_frame_transforms

ROOT = os.path.expanduser("~/flexnsense-sim")
USD = f"{ROOT}/assets/elrobot/usd/elrobot_follower.usd/elrobot_follower/elrobot_follower.usda"
OUT = f"{ROOT}/runs/bench/{args_cli.tag or ('hold' if args_cli.no_yield else 'yield')}"

FULL_SCALE = 5.886  # N: the PESOLA reads up to 600 gf
TRAVEL = 0.10  # m: ASSUMED marker travel at full scale
K_SPRING = FULL_SCALE / TRAVEL  # N/m
SCALE_LEN = 0.23  # m: ring (in the jaws) to the finger loop, at zero force
TUBE_START, TUBE_LEN = 0.025, 0.150  # m along the scale: where the tube begins and how long it is
TIP = 0.10  # m: ASSUMED distance from the gripper body's origin to the point between the jaw tips (the jaws'
# centres of mass are 0.062 m out, from the URDF)
SETTLE, REACH_T, PULL_T, HOLD_T, RELAX_T, REST_T = 1.0, 4.0, 2.5, 2.5, 1.5, 1.5  # seconds

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
    """Finger travel away from the gripper [m] at time t after the arm is in place."""
    cycle = PULL_T + HOLD_T + RELAX_T + REST_T
    t = t % cycle
    if t < PULL_T:
        return args_cli.pull * ease(t / PULL_T)
    if t < PULL_T + HOLD_T:
        return args_cli.pull
    if t < PULL_T + HOLD_T + RELAX_T:
        return args_cli.pull * (1 - ease((t - PULL_T - HOLD_T) / RELAX_T))
    return 0.0


def quat_z_to(d: torch.Tensor) -> torch.Tensor:
    """Quaternion (x, y, z, w) that turns the +Z axis onto the unit vector d."""
    z = torch.tensor([0.0, 0.0, 1.0], device=d.device)
    axis = torch.linalg.cross(z, d)
    s = axis.norm()
    if s < 1e-8:
        return torch.tensor([0.0, 0.0, 0.0, 1.0], device=d.device)
    ang = math.atan2(float(s), float(torch.dot(z, d)))
    return torch.cat([axis / s * math.sin(ang / 2), torch.tensor([math.cos(ang / 2)], device=d.device)])


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
        mat = lambda c, m=0.0: sim_utils.PreviewSurfaceCfg(diffuse_color=c, metallic=m)
        rig = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/rig",
                markers={
                    "finger": sim_utils.SphereCfg(radius=0.010, visual_material=mat((0.93, 0.72, 0.60))),
                    "wire": sim_utils.SphereCfg(radius=0.0025, visual_material=mat((0.60, 0.62, 0.66), 0.6)),
                    "tube": sim_utils.CylinderCfg(
                        radius=0.0075, height=TUBE_LEN, visual_material=mat((0.80, 0.82, 0.85), 0.5)
                    ),
                    "marker": sim_utils.SphereCfg(radius=0.0095, visual_material=mat((0.90, 0.12, 0.10))),
                    "limit": sim_utils.CylinderCfg(radius=0.0085, height=0.003, visual_material=mat((0.05, 0.07, 0.10))),
                },
            )
        )
        N_RING, N_ROD = 4, 10
        marker_idx = torch.tensor([0] + [1] * (N_RING + N_ROD) + [2, 3, 4], device=sim.device)
        sim.set_camera_view([0.5, 0.2, 0.4], [0.0, 0.2, 0.05])
        sim.reset()
        os.makedirs(OUT, exist_ok=True)
        dev, dt = sim.device, sim.get_physics_dt()
        ident = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=dev)

        arm_ids, _ = robot.find_joints("rev_motor_0[1-7]", preserve_order=True)
        grip_ids, grip_names = robot.find_joints("rev_motor_08.*")
        ee_id = robot.find_bodies("Gripper_Base_v1_1")[0][0]
        jac_body = ee_id - 1 if robot.is_fixed_base else ee_id
        jac_joints = [j + robot.num_base_dofs for j in arm_ids]
        limits = robot.data.joint_pos_limits.torch[0]
        mid, half = (limits[:, 0] + limits[:, 1]) / 2, (limits[:, 1] - limits[:, 0]) / 2
        start = (mid + 0.2 * half * torch.sin(torch.arange(robot.num_joints, device=dev) * 0.6)).unsqueeze(0)
        # gripper closed on the ring. From the URDF: motor 8 at its upper limit brings the jaws together
        # (jaw 08_1 to its lower limit, jaw 08_2 to its upper limit). 0.9 = closed on a ring, not fully shut.
        closed = {"rev_motor_08": 1.0, "rev_motor_08_1": 0.0, "rev_motor_08_2": 1.0}
        grip_des = torch.stack(
            [limits[j, 0] + (limits[j, 1] - limits[j, 0]) * (0.1 + 0.8 * closed[n]) for j, n in zip(grip_ids, grip_names)]
        ).unsqueeze(0)

        def ee_state():
            pose = robot.data.body_pose_w.torch[:, ee_id]
            root = robot.data.root_pose_w.torch
            pos_b, quat_b = subtract_frame_transforms(root[:, 0:3], root[:, 3:7], pose[:, 0:3], pose[:, 3:7])
            return pose[:, 0:3].clone(), pose[:, 3:7].clone(), pos_b, quat_b

        def step_sim():
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)
            camera.update(dt=dt)

        for _ in range(int(SETTLE / dt)):
            robot.set_joint_position_target_index(target=start)
            step_sim()

        # --- the gripper's own axes, from NormaCore's URDF: it points along local +y (the jaws' centres of mass sit at
        # y = +0.062 m, x = -0.027 and +0.029 m) and the jaws close along local x
        p, q, _, _ = ee_state()
        point_ax = torch.tensor([0.0, 1.0, 0.0], device=dev)
        close_ax = torch.tensor([1.0, 0.0, 0.0], device=dev)

        # --- the pose the bench uses: gripper low, horizontal, pointing away from the base, jaws closing sideways
        u = p[0].clone()
        u[2] = 0.0
        u = u / u.norm()  # horizontal direction from the base to where the gripper already is
        h = torch.linalg.cross(torch.tensor([0.0, 0.0, 1.0], device=dev), u)  # horizontal, square to u
        world = torch.stack([u, h, torch.linalg.cross(u, h)], dim=1)
        local = torch.stack([point_ax, close_ax, torch.linalg.cross(point_ax, close_ax)], dim=1)
        q_goal = quat_from_matrix((world @ local.T).unsqueeze(0))
        p_goal = (u * args_cli.reach + torch.tensor([0.0, 0.0, args_cli.height], device=dev)).unsqueeze(0)
        ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=1,
            device=dev,
        )

        def drive(pos_b, quat_b_goal):
            _, _, cur_p, cur_q = ee_state()
            ik.set_command(torch.cat([pos_b, quat_b_goal], dim=1))
            jac = robot.data.body_link_jacobian_w.torch[:, jac_body, :, jac_joints]
            des = ik.compute(cur_p, cur_q, jac, robot.data.joint_pos.torch[:, arm_ids])
            robot.set_joint_position_target_index(target=des, joint_ids=arm_ids)
            robot.set_joint_position_target_index(target=grip_des, joint_ids=grip_ids)

        p_start = p.clone()
        for i in range(int(REACH_T / dt)):
            a = ease(i * dt / (REACH_T * 0.8))
            drive(p_start + (p_goal - p_start) * a, q_goal)  # root is at the origin, unrotated: base frame = world
            step_sim()
        p0, q0, _, _ = ee_state()
        tip0 = p0[0] + u * TIP
        pos_err = float((p0 - p_goal).norm())
        ang_err = float(torch.rad2deg(torch.acos(torch.clamp(torch.dot(matrix_from_quat(q0)[0] @ point_ax, u), -1, 1))))
        print(f"[BENCH] in place: position error {pos_err*1000:.1f} mm, pointing error {ang_err:.1f} deg, gripper at",
              [round(float(v), 3) for v in p0[0]])

        # camera: above and to the side, like the bench photos
        centre = tip0 + u * 0.04
        eye = centre + h * 0.56 + torch.tensor([0.0, 0.0, 0.36], device=dev)
        camera.set_world_poses_from_view(eye.unsqueeze(0), centre.unsqueeze(0))
        sim.set_camera_view(eye.tolist(), centre.tolist())

        from PIL import Image, ImageDraw, ImageFont

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
            small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except OSError:
            font = small = ImageFont.load_default()
        mode = "Arm holds (no yielding)" if args_cli.no_yield else "Arm gives way above the limit"
        target = p0.clone()
        rows, frames = [], 0
        n_steps = int(args_cli.cycles * (PULL_T + HOLD_T + RELAX_T + REST_T) / dt)
        for step in range(n_steps):
            t = step * dt
            p, q, _, _ = ee_state()
            tip = p[0] + matrix_from_quat(q)[0] @ point_ax * TIP
            finger = tip0 + u * (SCALE_LEN + pull_at(t))
            d = finger - tip
            dist = d.norm()
            dirn = d / dist
            stretch = max(0.0, float(dist) - SCALE_LEN)
            force = K_SPRING * stretch
            # the scale pulls on the jaws, towards the finger
            robot.permanent_wrench_composer.set_forces_and_torques_index(
                forces=(dirn * force).reshape(1, 1, 3), positions=tip.reshape(1, 1, 3), body_ids=[ee_id], is_global=True
            )
            # admittance: give way along the scale while the force is above the limit
            if not args_cli.no_yield and force > args_cli.cap:
                target[0] += dirn * args_cli.gain * (force - args_cli.cap) * dt
            drive(target, q_goal)
            step_sim()

            # --- draw the scale: ring in the jaws, tube, rod to the loop, fingertip, marker, limit tick
            along = lambda s: tip + dirn * s
            ring = torch.stack([along(s) for s in torch.linspace(0.0, TUBE_START, N_RING).tolist()])
            rod = torch.stack([along(s) for s in torch.linspace(TUBE_START + TUBE_LEN, float(dist) - 0.012, N_ROD).tolist()])
            mark_at = lambda f: along(TUBE_START + 0.012 + (TUBE_LEN - 0.024) * min(1.0, f / FULL_SCALE))
            qd = quat_z_to(dirn)
            trans = torch.cat([finger.unsqueeze(0), ring, rod, along(TUBE_START + TUBE_LEN / 2).unsqueeze(0),
                               mark_at(force).unsqueeze(0), mark_at(args_cli.cap).unsqueeze(0)])
            orient = ident.repeat(trans.shape[0], 1)
            orient[-3] = qd
            orient[-1] = qd
            rig.visualize(translations=trans, orientations=orient, marker_indices=marker_idx)

            give = float(torch.dot(p[0] - p0[0], u))
            rows.append((round(t, 4), round(force, 4), round(pull_at(t), 4), round(give, 5)))
            if step % 2 == 0 and not args_cli.no_record:
                rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
                im = Image.fromarray(rgb)
                dr = ImageDraw.Draw(im)
                dr.rectangle([0, 0, im.width, 118], fill=(14, 20, 31))
                dr.text((28, 14), mode, font=font, fill=(244, 247, 251))
                dr.text(
                    (28, 66),
                    f"force {force:4.2f} N  ({force / 9.80665 * 1000:3.0f} gf)     limit {args_cli.cap:.2f} N (200 gf)     t = {t:4.1f} s",
                    font=small,
                    fill=(255, 110, 100) if force > args_cli.cap * 1.05 else (150, 230, 190),
                )
                im.save(f"{OUT}/frame_{frames:04d}.png")
                frames += 1

        with open(f"{OUT}/log.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "force_N", "finger_pull_m", "gripper_give_m"])
            w.writerows(rows)
        forces = [r[1] for r in rows]
        summary = {
            "mode": "hold" if args_cli.no_yield else "yield",
            "cap_N": args_cli.cap,
            "gain_m_per_s_per_N": args_cli.gain,
            "spring_N_per_m": round(K_SPRING, 3),
            "finger_pull_m": args_cli.pull,
            "reach_position_error_mm": round(pos_err * 1000, 1),
            "reach_pointing_error_deg": round(ang_err, 1),
            "max_force_N": round(max(forces), 3),
            "max_force_gf": round(max(forces) / 9.80665 * 1000, 1),
            "seconds_above_110pct_cap": round(sum(1 for v in forces if v > args_cli.cap * 1.1) * dt, 3),
            "max_gripper_give_m": round(max(r[3] for r in rows), 4),
            "frames": frames,
        }
        with open(f"{OUT}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("[BENCH] summary:", json.dumps(summary))


if __name__ == "__main__":
    main()
