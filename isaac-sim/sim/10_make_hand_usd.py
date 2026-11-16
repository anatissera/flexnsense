"""Write the hand used on the training bench: a palm and four fingers, as one body that the task places per arm.

The index finger's upright part is NOT in this file: it is its own body in the scene, because it is the one that
pulls on the scale's loop and moves. Frame of the hand: origin where the index finger stands on the table, +y along
the scale (away from the arm), +z up; the other fingers stand 24 mm apart towards -x.

    source ~/isaac-env.sh && cd ~/IsaacLab
    uv run --extra isaacsim python ~/flexnsense_sim/10_make_hand_usd.py
Writes ~/flexnsense-sim/assets/bench/hand.usda
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()

import os

import isaaclab.sim as sim_utils

GAP, FLAT_LEN, KNUCKLE_Z, UP_LEN, R = 0.024, 0.075, 0.018, 0.062, 0.0095
SKIN = (0.86, 0.60, 0.47)


def main():
    with launch_simulation(sim_utils.SimulationCfg(device=args_cli.device), args_cli):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        out = os.path.expanduser("~/flexnsense-sim/assets/bench/hand.usda")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        stage = Usd.Stage.CreateNew(out)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        hand = UsdGeom.Xform.Define(stage, "/Hand")
        stage.SetDefaultPrim(hand.GetPrim())
        body = UsdPhysics.RigidBodyAPI.Apply(hand.GetPrim())
        body.CreateKinematicEnabledAttr(True)  # placed by the task, not moved by forces

        def part(name, kind, pos, **attrs):
            prim = kind.Define(stage, f"/Hand/{name}")
            for k, v in attrs.items():
                getattr(prim, f"Create{k}Attr")(v)
            prim.AddTranslateOp().Set(Gf.Vec3d(*pos))
            prim.CreateDisplayColorAttr([Gf.Vec3f(*SKIN)])
            return prim

        for j in range(4):
            x = -GAP * j
            part(f"flat_{j}", UsdGeom.Cylinder, (x, FLAT_LEN / 2, KNUCKLE_Z), Radius=R, Height=FLAT_LEN, Axis="Y")
            part(f"knuckle_{j}", UsdGeom.Sphere, (x, 0.0, KNUCKLE_Z), Radius=R)
            if j:  # the index finger's upright part is a separate, moving body
                part(f"up_{j}", UsdGeom.Cylinder, (x, 0.0, KNUCKLE_Z + UP_LEN / 2), Radius=R, Height=UP_LEN, Axis="Z")
                part(f"tip_{j}", UsdGeom.Sphere, (x, 0.0, KNUCKLE_Z + UP_LEN), Radius=R)
        palm = part("palm", UsdGeom.Cube, (-GAP * 1.5, FLAT_LEN + 0.05, 0.015), Size=1.0)
        palm.AddScaleOp().Set(Gf.Vec3f(0.10, 0.105, 0.030))
        stage.GetRootLayer().Save()
        print("[HAND] wrote", out, flush=True)


if __name__ == "__main__":
    main()
