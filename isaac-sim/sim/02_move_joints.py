"""Phase 1 check: load the imported ElRobot, move every joint, and save rendered pictures.

Runs ON the workstation, after 01_import_elrobot.sh:

    source ~/isaac-env.sh && cd ~/IsaacLab
    uv run --extra isaacsim python ~/flexnsense_sim/02_move_joints.py

Writes ~/flexnsense-sim/runs/phase1/ : frame_*.png (RTX camera) and report.json (joint names, limits, and how well
each joint followed a sine target). Patterned on Isaac Lab's tutorials 01_assets/run_articulation.py and
04_sensors/run_usd_camera.py at main@727e4e43a.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Move the ElRobot's joints and render it.")
parser.add_argument("--steps", type=int, default=480, help="Physics steps to run.")
parser.add_argument("--amp", type=float, default=0.2, help="Sweep size as a fraction of each joint's half range. "
                    "0.6 drives the arm into the table, which shows up as large tracking errors.")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

import json
import math
import os

import torch
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors.camera.utils import save_images_to_file
from isaaclab.utils import instantiate

ROOT = os.path.expanduser("~/flexnsense-sim")
USD = f"{ROOT}/assets/elrobot/usd/elrobot_follower.usd/elrobot_follower/elrobot_follower.usda"
OUT = f"{ROOT}/runs/phase1"

# The ST3215: 30 kg.cm stall = 2.94 N.m, 4.71 rad/s (both from NormaCore's URDF). Stiffness and damping are a first
# guess for a position servo; they have to be fitted to the real arm later.
ELROBOT_CFG = ArticulationCfg(
    prim_path="/World/ElRobot",
    spawn=sim_utils.UsdFileCfg(usd_path=USD, fix_root_link=True),  # bolted to the bench
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    actuators={
        "servos": ImplicitActuatorCfg(
            joint_names_expr=[".*"], joint_effort_limit=2.94, joint_velocity_limit=4.71, stiffness=20.0, damping=1.0
        )
    },
)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/defaultGroundPlane", cfg)
        cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.93, 1.0))
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
        sim.reset()
        camera.set_world_poses_from_view(
            torch.tensor([[0.45, -0.45, 0.35]], device=sim.device), torch.tensor([[0.0, 0.0, 0.15]], device=sim.device)
        )
        os.makedirs(OUT, exist_ok=True)

        names = list(robot.joint_names)
        limits = robot.data.joint_pos_limits.torch[0].clone()  # (J, 2)
        lo, hi = limits[:, 0], limits[:, 1]
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        print("[ELROBOT] joints:", names)
        print("[ELROBOT] bodies:", list(robot.body_names))

        dt = sim.get_physics_dt()
        err_sum = torch.zeros(len(names), device=sim.device)
        err_max = torch.zeros(len(names), device=sim.device)
        frames = 0
        for step in range(args_cli.steps):
            t = step * dt
            # every joint sweeps a part of its range, each with a different phase
            phase = torch.arange(len(names), device=sim.device) * 0.6
            target = (mid + args_cli.amp * half * torch.sin(2 * math.pi * 0.25 * t + phase)).unsqueeze(0)
            robot.set_joint_position_target_index(target=target)
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)
            camera.update(dt=dt)
            if step > 60:  # after the start transient
                e = (robot.data.joint_pos.torch[0] - target[0]).abs()
                err_sum += e
                err_max = torch.maximum(err_max, e)
            if step % (args_cli.steps // 6) == 0 and step > 0:
                rgb = camera.data.output["rgb"][0:1, ..., :3].float() / 255.0
                save_images_to_file(rgb, f"{OUT}/frame_{frames:02d}.png")
                frames += 1

        n = max(1, args_cli.steps - 61)
        report = {
            "usd": USD,
            "physics_dt": dt,
            "steps": args_cli.steps,
            "amp": args_cli.amp,
            "joints": [
                {
                    "name": nm,
                    "lower": round(float(lo[i]), 4),
                    "upper": round(float(hi[i]), 4),
                    "mean_abs_error": round(float(err_sum[i] / n), 5),
                    "max_abs_error": round(float(err_max[i]), 5),
                }
                for i, nm in enumerate(names)
            ],
            "bodies": list(robot.body_names),
            "frames": frames,
        }
        with open(f"{OUT}/report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("[ELROBOT] wrote", OUT, "frames:", frames)
        for j in report["joints"]:
            print("[ELROBOT]", j)


if __name__ == "__main__":
    main()
