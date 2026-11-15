"""Find out why driving the ElRobot's gripper turns its wrist in the converted model.

Four copies of the arm, side by side, all holding the same arm posture. They differ only in how the gripper is driven:
  A  motor 8 (the gear) and both jaws driven to "closed"
  B  only motor 8 driven, jaws free
  C  only the jaws driven, motor 8 free
  D  motor 8 and jaws driven, but to where they already are (no motion asked)
Prints, for each, how far the wrist roll (motor 7) turned in 4 seconds and where the gripper joints ended up.

    source ~/isaac-env.sh && cd ~/IsaacLab
    uv run --extra isaacsim python ~/flexnsense_sim/dbg_gripper.py
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()

import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg

USD = os.environ.get("ELROBOT_USD") or os.path.expanduser(
    "~/flexnsense-sim/assets/elrobot/usd/elrobot_follower.usd/elrobot_follower/elrobot_follower.usda"
)  # set ELROBOT_USD to test another conversion of the arm
SEED = {"rev_motor_01": 0.0, "rev_motor_02": 0.04, "rev_motor_03": 0.6, "rev_motor_04": 0.072, "rev_motor_05": 0.0,
        "rev_motor_06": -0.6, "rev_motor_07": 0.0, "rev_motor_08": 0.0, "rev_motor_08_1": 0.0, "rev_motor_08_2": 0.0}
CLOSED = {"rev_motor_08": 1.98, "rev_motor_08_1": -0.0229, "rev_motor_08_2": 0.0229}


def cfg(path, x, gear_k, jaw_k):
    act = lambda expr, k: ImplicitActuatorCfg(
        joint_names_expr=expr, joint_effort_limit=2.94, joint_velocity_limit=4.71, stiffness=k, damping=2.0 if k else 0.5
    )
    return ArticulationCfg(
        prim_path=path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD, fix_root_link=True,
            articulation_props=[PhysxArticulationCfg(enabled_self_collisions=False)] if os.environ.get("DBG_NOSELF") else None,
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(x, 0.0, 0.0 if os.environ.get("DBG_TABLE") else 0.3), joint_pos=SEED),
        actuators={"arm": act(["rev_motor_0[1-7]"], 40.0), "gear": act(["rev_motor_08"], gear_k),
                   "jaws": act(["rev_motor_08_[12]"], jaw_k)},
    )


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)
        light = sim_utils.DomeLightCfg(intensity=1000.0)
        light.func("/World/Light", light)
        if os.environ.get("DBG_TABLE"):  # a table top under the arms, its surface at z = 0
            slab = sim_utils.CuboidCfg(size=(4.0, 2.0, 0.03), collision_props=sim_utils.UsdPhysicsCollisionCfg())
            slab.func("/World/Table", slab, translation=(0.75, 0.0, -0.015))
        cases = {"A gear+jaws driven": (40.0, 40.0, True), "B gear only": (40.0, 0.0, True),
                 "C jaws only": (0.0, 40.0, True), "D driven, no motion asked": (40.0, 40.0, False)}
        arms = {}
        for i, (name, (gk, jk, _)) in enumerate(cases.items()):
            arms[name] = Articulation(cfg=cfg(f"/World/Arm{i}", 0.5 * i, gk, jk))
        sim.reset()
        dt = sim.get_physics_dt()
        any_arm = next(iter(arms.values()))
        names = list(any_arm.joint_names)
        print("[GRIP] joints:", names, flush=True)
        targets = {}
        for name, (_, _, move) in cases.items():
            t = arms[name].data.joint_pos.torch.clone()
            if move:
                for k, v in CLOSED.items():
                    t[0, names.index(k)] = v
            targets[name] = t
        for step in range(int(4.0 / dt)):
            for name, arm in arms.items():
                arm.set_joint_position_target_index(target=targets[name])
                arm.write_data_to_sim()
            sim.step()
            for arm in arms.values():
                arm.update(dt)
        for name, arm in arms.items():
            q = arm.data.joint_pos.torch[0]
            g = lambda k: round(float(q[names.index(k)]), 4)
            root_z = float(arm.data.body_pose_w.torch[0, :, 2].max())
            print(f"[GRIP] {name:28s} highest body z = {root_z:6.3f} m | all finite: {bool(torch.isfinite(q).all())}", flush=True)
            print(f"[GRIP] {name:28s} wrist roll (motor 7) = {g('rev_motor_07'):8.3f} rad | wrist pitch (6) = {g('rev_motor_06'):6.3f} | "
                  f"gear = {g('rev_motor_08'):6.3f} (asked {float(targets[name][0, names.index('rev_motor_08')]):.2f}) | "
                  f"jaws = {g('rev_motor_08_1'):7.4f}, {g('rev_motor_08_2'):7.4f}", flush=True)


if __name__ == "__main__":
    main()
