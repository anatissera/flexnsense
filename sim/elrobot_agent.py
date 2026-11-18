"""The ElRobot follower as a ManiSkill agent, from NormaCore's own URDF.

    cd sim && source venv/bin/activate
    python -m sim.elrobot_agent --pose rest          # open the viewer
    python -m sim.elrobot_agent --ticks 1:1976,2:1719,3:2526,4:2755,5:1907,6:1270,7:1282

Runs in its **own virtual environment** (`sim/venv`, made by `sim/setup_env.sh`), not the
repo's `.venv`. ManiSkill pulls in torch, SAPIEN and a Vulkan runtime, several gigabytes of it,
and the repo's test suite has to keep running on a machine with no GPU at all. Same split
`robot_training/` already uses for its LeRobot environment.

**What this is for, and what it is not for.** Use it to see the arm move, to try a pose before
commanding it, and for a closed-loop simulation rig (see isaac-sim/), where contact and
dynamics genuinely matter. Do **not** use it to answer "which motor moves the gripper along the
tether": `scripts/arm/which_motor.py` answers that exactly, from the same URDF, in milliseconds,
with no GPU and no install, because that question is pure geometry.

The joint order ManiSkill uses is the URDF's active-joint order, which for this arm is
rev_motor_01 to rev_motor_08 and then the two gripper jaws, ten in total. `qpos_from_ticks`
converts a Station tick reading into that vector, reusing the tick convention from
`lib.kinematics` rather than restating it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lib.kinematics import CENTRE_TICKS, ticks_to_radians  # noqa: E402

URDF_PATH = REPO_ROOT / "norma-core/hardware/elrobot/simulation/elrobot_follower.urdf"

#: ManiSkill orders qpos by the URDF's active joints. The two prismatic jaws come last.
ARM_MOTORS = (1, 2, 3, 4, 5, 6, 7, 8)
N_JAWS = 2


def qpos_from_ticks(ticks: dict[int, float], jaw_m: float = 0.0) -> np.ndarray:
    """A Station tick reading to ManiSkill's qpos vector, in radians (and metres for the jaws).

    Motors with no reading are centred, matching `lib.kinematics.Chain.link_transform`: a
    partial reading should still show you a pose rather than refusing.
    """
    q = [ticks_to_radians(ticks.get(m, CENTRE_TICKS)) for m in ARM_MOTORS]
    q += [-abs(jaw_m), abs(jaw_m)]      # the jaws travel in opposite senses, per the URDF limits
    return np.array(q, dtype=np.float32)


def parse_ticks(text: str) -> dict[int, float]:
    out = {}
    for part in text.split(","):
        motor, _, value = part.partition(":")
        out[int(motor)] = float(value)
    return out


def build_agent():
    """Define and register the agent. Imported lazily so this module is importable without
    ManiSkill, which is what lets `tests/sim/` check the qpos maths in the normal .venv."""
    import sapien
    from mani_skill.agents.base_agent import BaseAgent, Keyframe
    from mani_skill.agents.registration import register_agent

    @register_agent()
    class ElRobotFollower(BaseAgent):
        uid = "elrobot_follower"
        urdf_path = str(URDF_PATH)
        #: Every joint centred, which is the URDF's zero and the middle of every calibrated range.
        keyframes = dict(
            rest=Keyframe(
                qpos=qpos_from_ticks({}),
                pose=sapien.Pose(),
            ),
        )

    return ElRobotFollower


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticks", help="show this pose, as 1:1976,2:1719,...")
    p.add_argument("--pose", default="rest", help="a named keyframe (default: rest)")
    args = p.parse_args(argv)

    if not URDF_PATH.is_file():
        raise SystemExit(
            f"URDF not found at {URDF_PATH}.\n"
            "It ships with the hardware; see SETUP.md for the norma-core sparse clone."
        )

    try:
        build_agent()
    except ImportError as exc:
        raise SystemExit(
            f"ManiSkill is not importable here ({exc}).\n\n"
            "This module needs its own environment:\n\n"
            "    ./sim/setup_env.sh && source sim/venv/bin/activate\n\n"
            "See sim/README.md. For the geometry question (which motor moves the gripper "
            "along the tether) use scripts/arm/which_motor.py instead: no GPU, no install."
        ) from None

    import tyro
    import mani_skill.examples.demo_robot as demo

    if args.ticks:
        q = qpos_from_ticks(parse_ticks(args.ticks))
        print("qpos (rad):", np.round(q, 4))
        print("Pass it as a keyframe, or set it on the articulation once the viewer is open.")

    demo_args = tyro.cli(demo.Args, args=["-r", "elrobot_follower", "-k", args.pose])
    demo.main(demo_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
