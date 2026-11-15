"""The arm takes the scale off one finger and puts it on the next, then the pull is repeated there.

Runs ON the workstation:

    source ~/isaac-env.sh && cd ~/IsaacLab
    uv run --extra isaacsim python ~/flexnsense_sim/06_move_scale.py --tag move
    LIVESTREAM=1 uv run --extra isaacsim python ~/flexnsense_sim/06_move_scale.py --cycles 100000 --no_record --viz kit

One cycle: pull on the index finger (the arm gives way above the limit) -> lift the loop off -> move it over the middle
finger -> lower it on -> pull on the middle finger -> move it back.

This is the task the team's policies were trained for ("move device from right index to right middle finger"), but
here the motion is SCRIPTED with inverse kinematics. No trained policy runs in this script.

The hand is drawn only (four fingers and a palm, no physics); the scale is drawn as in 05_bench_scene.py; the force is
still a virtual spring between the jaw tips and the finger that wears the loop.

Writes ~/flexnsense-sim/runs/move/<tag>/ : frame_*.png, log.csv, summary.json.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Force-limited pull on the ElRobot, arranged like the bench.")
parser.add_argument("--no_yield", action="store_true", help="Baseline: the arm holds its position.")
parser.add_argument("--cap", type=float, default=1.96, help="Force limit [N] (1.96 N = 200 gf).")
parser.add_argument("--gain", type=float, default=1.0, help="Admittance gain B [m/s per N].")
parser.add_argument("--pull", type=float, default=0.06, help="How far the finger pulls away [m].")
parser.add_argument("--tag", default=None, help="Output folder name under runs/bench/ (default: yield or hold).")
parser.add_argument("--cycles", type=int, default=1, help="Cycles to run (index -> middle -> back to index).")
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
OUT = f"{ROOT}/runs/move/{args_cli.tag or ('hold' if args_cli.no_yield else 'move')}"

FULL_SCALE = 5.886  # N: the PESOLA reads up to 600 gf
TRAVEL = 0.10  # m: ASSUMED marker travel at full scale
K_SPRING = FULL_SCALE / TRAVEL  # N/m
SCALE_LEN = 0.23  # m: ring (in the jaws) to the finger loop, at zero force
TUBE_START, TUBE_LEN = 0.025, 0.150  # m along the scale: where the tube begins and how long it is
TIP = 0.10  # m: ASSUMED distance from the gripper body's origin to the point between the jaw tips (the jaws'
# centres of mass are 0.062 m out, from the URDF)
SETTLE, REACH_T, PULL_T, HOLD_T, RELAX_T, REST_T = 1.0, 4.0, 2.0, 1.5, 1.2, 0.8  # seconds
LIFT_T, SHIFT_T, LOWER_T, SETTLE_T = 1.5, 1.5, 1.5, 0.8  # seconds: loop off, over, on, and a pause before pulling
SPACING, LIFT, FINGER_LEN = 0.024, 0.045, 0.075  # m: between fingers, how high the loop is lifted, drawn finger
FINGERS = ["index", "middle", "ring", "little"]

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
        wood = sim_utils.CuboidCfg(size=(1.8, 1.4, 0.004), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.56, 0.36), roughness=0.8))
        wood.func("/World/TableTop", wood, translation=(0.0, 0.35, 0.0015))
        card = sim_utils.CuboidCfg(size=(0.62, 0.52, 0.003), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.66, 0.52, 0.37), roughness=0.95))
        card.func("/World/Cardboard", card, translation=(0.0, 0.42, 0.0045))
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
                    "finger": sim_utils.SphereCfg(radius=0.0095, visual_material=mat((0.93, 0.72, 0.60))),
                    "bone": sim_utils.CylinderCfg(radius=0.0095, height=FINGER_LEN, visual_material=mat((0.93, 0.72, 0.60))),
                    "palm": sim_utils.CuboidCfg(size=(0.10, 0.105, 0.030), visual_material=mat((0.93, 0.72, 0.60))),
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
        nf = len(FINGERS)
        marker_idx = torch.tensor([0] * nf + [1] * nf + [2] + [3] * (N_RING + N_ROD) + [4, 5, 6], device=sim.device)
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
            q_now = robot.data.joint_pos.torch[:, arm_ids]
            robot.set_joint_position_target_index(target=ik.compute(cur_p, cur_q, jac, q_now), joint_ids=arm_ids)
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
        centre = tip0 + u * 0.13 + h * 0.03
        eye = centre - u * 0.10 + h * 0.40 + torch.tensor([0.0, 0.0, 0.50], device=dev)
        camera.set_world_poses_from_view(eye.unsqueeze(0), centre.unsqueeze(0))
        sim.set_camera_view(eye.tolist(), centre.tolist())

        from PIL import Image, ImageDraw, ImageFont

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
            small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except OSError:
            font = small = ImageFont.load_default()
        up = torch.tensor([0.0, 0.0, 1.0], device=dev)
        q_u = quat_z_to(u)
        q_palm = quat_from_matrix(torch.stack([u, h, up], dim=1).unsqueeze(0))[0]
        LEVER = TIP + SCALE_LEN  # gripper origin to the loop
        yaw_of = lambda j: math.atan2(SPACING * j, LEVER)  # turn of the gripper that puts the loop on finger j
        TILT = math.atan2(LIFT, LEVER)  # tilt of the gripper that lifts the loop clear of the fingertip

        def aim(yaw: float, tilt: float):
            """Gripper orientation (and its pointing direction) for a yaw about the vertical and an upward tilt."""
            d = u * math.cos(yaw) * math.cos(tilt) + h * math.sin(yaw) * math.cos(tilt) + up * math.sin(tilt)
            side = torch.linalg.cross(up, d)
            side = side / side.norm()
            world_t = torch.stack([d, side, torch.linalg.cross(d, side)], dim=1)
            return quat_from_matrix((world_t @ local.T).unsqueeze(0)), d

        base_tip = lambda j: p0[0] + aim(yaw_of(j), 0.0)[1] * LEVER  # where finger j's tip rests
        pull_len = PULL_T + HOLD_T + RELAX_T + REST_T
        # one cycle: (kind, seconds, finger the loop is on / going to)
        plan = [("pull", pull_len, 0), ("lift", LIFT_T, 0), ("shift", SHIFT_T, 1), ("lower", LOWER_T, 1), ("settle", SETTLE_T, 1),
                ("pull", pull_len, 1), ("lift", LIFT_T, 1), ("shift", SHIFT_T, 0), ("lower", LOWER_T, 0), ("settle", SETTLE_T, 0)]
        cycle_len = sum(d for _, d, _ in plan)
        label = {"pull": "Pull on the {f} finger", "lift": "Taking the scale off the {f} finger",
                 "shift": "Moving the scale to the {f} finger", "lower": "Putting the scale on the {f} finger",
                 "settle": "Putting the scale on the {f} finger"}
        target = p0.clone()
        rows, frames, last_kind = [], 0, None
        n_steps = int(args_cli.cycles * cycle_len / dt)
        for step in range(n_steps):
            t = step * dt
            tc = t % cycle_len
            for kind, dur, fj in plan:
                if tc < dur:
                    break
                tc -= dur
            a = ease(tc / dur)
            p, q, _, _ = ee_state()
            point_w = matrix_from_quat(q)[0] @ point_ax
            tip = p[0] + point_w * TIP
            pulls = [0.0] * nf
            yaw, tilt = yaw_of(fj), 0.0
            if kind == "pull":
                if last_kind != "pull":
                    target[0] = p0[0]
                pulls[fj] = pull_at(tc)
                loop_end = base_tip(fj) + u * pulls[fj]
                d = loop_end - tip
                dist = d.norm()
                dirn = d / dist
                force = K_SPRING * max(0.0, float(dist) - SCALE_LEN)
                if not args_cli.no_yield and force > args_cli.cap and float((target[0] - p0[0]).norm()) < 0.08:
                    target[0] += dirn * args_cli.gain * (force - args_cli.cap) * dt
            else:  # the loop is off the finger: the scale sticks straight out of the jaws, no force
                if kind == "lift":
                    if last_kind != "lift":
                        lift_from = target[0].clone()
                    target[0] = lift_from + (p0[0] - lift_from) * a  # back to where the pull started
                    tilt = TILT * a
                elif kind == "shift":
                    yaw, tilt = yaw_of(1 - fj) + (yaw_of(fj) - yaw_of(1 - fj)) * a, TILT
                elif kind == "lower":
                    tilt = TILT * (1 - a)
                dirn, dist, force = point_w, SCALE_LEN, 0.0
            if kind != last_kind:
                err = float(torch.rad2deg(torch.acos(torch.clamp(torch.dot(point_w, aim(yaw, tilt)[1]), -1, 1))))
                print(f"[MOVE] t={t:5.2f} -> {kind:6s} finger {fj} | gripper", [round(float(v), 3) for v in p[0]],
                      "| target", [round(float(v), 3) for v in target[0]], f"| pointing error {err:.1f} deg", flush=True)
            last_kind = kind
            robot.permanent_wrench_composer.set_forces_and_torques_index(
                forces=(dirn * force).reshape(1, 1, 3), positions=tip.reshape(1, 1, 3), body_ids=[ee_id], is_global=True
            )
            drive(target, aim(yaw, tilt)[0])
            step_sim()

            # --- draw: the hand (tips, fingers, palm), then the scale (ring, rod, tube, marker, limit tick)
            tips = torch.stack([base_tip(j) + u * pulls[j] for j in range(nf)])
            bones = tips + u * (FINGER_LEN / 2)
            palm = (tip0 + u * (SCALE_LEN + FINGER_LEN + 0.045) + h * SPACING * (nf - 1) / 2 - up * 0.008).unsqueeze(0)
            along = lambda sdist: tip + dirn * sdist
            ring = torch.stack([along(v) for v in torch.linspace(0.0, TUBE_START, N_RING).tolist()])
            rod = torch.stack([along(v) for v in torch.linspace(TUBE_START + TUBE_LEN, float(dist) - 0.004, N_ROD).tolist()])
            mark_at = lambda f: along(TUBE_START + 0.012 + (TUBE_LEN - 0.024) * min(1.0, f / FULL_SCALE))
            qd = quat_z_to(dirn)
            trans = torch.cat([tips, bones, palm, ring, rod, along(TUBE_START + TUBE_LEN / 2).unsqueeze(0),
                               mark_at(force).unsqueeze(0), mark_at(args_cli.cap).unsqueeze(0)])
            orient = ident.repeat(trans.shape[0], 1)
            orient[nf : 2 * nf] = q_u
            orient[2 * nf] = q_palm
            orient[-3] = qd
            orient[-1] = qd
            rig.visualize(translations=trans, orientations=orient, marker_indices=marker_idx)

            rows.append((round(t, 4), round(force, 4), round(pulls[fj] if kind == "pull" else 0.0, 4),
                         round(float(torch.dot(p[0] - p0[0], u)), 5)))
            if step % 2 == 0 and not args_cli.no_record:
                rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
                im = Image.fromarray(rgb)
                dr = ImageDraw.Draw(im)
                dr.rectangle([0, 0, im.width, 118], fill=(14, 20, 31))
                dr.text((28, 14), label[kind].format(f=FINGERS[fj]), font=font, fill=(244, 247, 251))
                dr.text(
                    (28, 66),
                    f"force {force:4.2f} N  ({force / 9.80665 * 1000:3.0f} gf)     limit {args_cli.cap:.2f} N (200 gf)     "
                    f"scripted motion, simulated     t = {t:4.1f} s",
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
            "mode": "move the scale between fingers (scripted)",
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
