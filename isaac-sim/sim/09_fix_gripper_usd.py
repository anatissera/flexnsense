"""Write a corrected copy of the converted ElRobot model: the jaws back on the gripper.

In the converted model the three gripper joints (rev_motor_08, rev_motor_08_1, rev_motor_08_2) have frames whose two
sides disagree with where the parts are drawn (36 and 87 degrees), so the physics engine puts the jaws about 10 cm off
the gripper. This writes, next to each converted model, `elrobot_follower_fixed.usda`: a thin layer over the original
that only corrects those joint frames. Training environments load the fixed file.

    source ~/isaac-env.sh && cd ~/IsaacLab
    uv run --extra isaacsim python ~/flexnsense_sim/09_fix_gripper_usd.py
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()

import math
import os

import isaaclab.sim as sim_utils

ROOT = os.path.expanduser("~/flexnsense-sim/assets/elrobot")


def fix(folder: str):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    folder, _, name = folder.partition(":")
    name = name or "elrobot_follower"
    src = f"{ROOT}/{folder}/elrobot_follower.usd/{name}/{name}.usda"
    dst = src.replace(f"{name}.usda", f"{name}_fixed.usda")
    if os.path.exists(dst):
        os.remove(dst)
    layer = Sdf.Layer.CreateNew(dst)
    layer.subLayerPaths.append(f"./{name}.usda")
    stage = Usd.Stage.Open(layer)
    src_stage = Usd.Stage.Open(src)
    stage.SetDefaultPrim(stage.GetPrimAtPath(src_stage.GetDefaultPrim().GetPath()))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(src_stage))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(src_stage))
    xf = UsdGeom.XformCache()
    n = 0
    for jp in stage.Traverse():
        if not jp.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(jp)
        t0, t1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if not t0 or not t1:
            continue
        b0, b1 = stage.GetPrimAtPath(t0[0]), stage.GetPrimAtPath(t1[0])

        def side(rot_attr, pos_attr):
            q = rot_attr.Get()
            m = Gf.Matrix4d().SetRotate(Gf.Quatd(q.GetReal(), Gf.Vec3d(*q.GetImaginary())))
            m.SetTranslateOnly(Gf.Vec3d(*pos_attr.Get()))
            return m

        w0 = side(joint.GetLocalRot0Attr(), joint.GetLocalPos0Attr()) * xf.GetLocalToWorldTransform(b0)
        w1 = side(joint.GetLocalRot1Attr(), joint.GetLocalPos1Attr()) * xf.GetLocalToWorldTransform(b1)
        gap = (w0.ExtractTranslation() - w1.ExtractTranslation()).GetLength()
        dq = w0.RemoveScaleShear().ExtractRotationQuat() * w1.RemoveScaleShear().ExtractRotationQuat().GetInverse()
        turn = math.degrees(2 * math.acos(min(1.0, abs(dq.GetReal()))))
        if gap > 5e-4 or turn > 0.5:
            in_b1 = w0 * xf.GetLocalToWorldTransform(b1).GetInverse()
            q1 = in_b1.RemoveScaleShear().ExtractRotationQuat()
            joint.GetLocalPos1Attr().Set(Gf.Vec3f(*in_b1.ExtractTranslation()))
            joint.GetLocalRot1Attr().Set(Gf.Quatf(float(q1.GetReal()), Gf.Vec3f(*q1.GetImaginary())))
            n += 1
            print(f"[FIX] {folder} {jp.GetName()}: sides disagreed by {gap * 1000:.1f} mm, {turn:.1f} deg; corrected", flush=True)
    layer.Save()
    print(f"[FIX] wrote {dst} ({n} joints corrected)", flush=True)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        for folder in os.environ.get("FIX_FOLDERS", "usd,usd_decomposed").split(","):
            fix(folder)


if __name__ == "__main__":
    main()
