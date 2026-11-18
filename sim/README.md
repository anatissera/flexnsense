# Simulating the ElRobot

NormaCore ships a URDF with the hardware
(`norma-core/hardware/elrobot/simulation/elrobot_follower.urdf`): 19 links, each with mass,
inertia, collision and visual geometry, plus 19 STL meshes. It is a complete, physics-ready
model of the follower arm, and it is the official description of the arm we actually have,
rather than something reconstructed.

Two different things can be done with it, and they need very different amounts of machinery.

## First, check whether you need a simulator at all

**"Which motor moves the gripper along the tether axis?"** is geometry, not physics. It is
answered exactly by the URDF's joint axes plus the arm's current pose, with no simulator, no
GPU and no install:

```bash
.venv/bin/python scripts/arm/which_motor.py             # reads the live pose from Station
.venv/bin/python scripts/arm/which_motor.py --plot      # same, with a 3D drawing
.venv/bin/python scripts/arm/which_motor.py --check     # verify the URDF against this arm
```

That runs in the repo's normal `.venv` in milliseconds. Reach for ManiSkill when the question
involves **contact, dynamics or rendering**: the closed-loop simulation rig in
a closed-loop rig ([isaac-sim/](../isaac-sim/)), trying a trajectory before commanding it, or generating
images.

## What the URDF already told us

Worth recording, because `docs/elrobot.md` lists the id-to-joint mapping as unverified and
warns against assuming it. Comparing the URDF's joint limits against the hand-measured
firmware ranges in `config/calibration/` settles it:

| Motor | Measured range | URDF limit | Difference |
|---|---|---|---|
| 1 | 169.3 deg | 177.7 deg | -8.4 |
| 2 | 184.9 deg | 184.7 deg | +0.2 |
| 3 | 203.4 deg | 201.8 deg | +1.6 |
| 4 | 202.1 deg | 200.9 deg | +1.2 |
| 5 | 335.6 deg | 336.4 deg | -0.9 |
| 6 | 180.0 deg | 180.0 deg | -0.0 |
| 7 | 340.8 deg | 340.1 deg | +0.8 |
| 8 | 141.7 deg | 126.2 deg | +15.5 (the gripper, see below) |

Six of the seven arm joints agree within 2 degrees, so **`rev_motor_0N` is Station's motor N**.
Motor 1 is 8 degrees narrower, consistent with a conservative manual calibration. Motor 8 is
the gripper, whose calibrated range is set by where the jaws close rather than by the
mechanism's travel, so it legitimately differs.

Every calibrated range is also centred on tick 2048, which is what makes "tick 2048 is the
URDF's zero angle" a measured fact rather than a guess. Both conventions live in
`lib/kinematics.py` and are pinned by `tests/lib/test_kinematics.py`.

The geometry was then checked against the arm itself. Predicted motion directions match what
the arm was observed to do when each motor was jogged by hand, including the detail that
**motor 7 does not translate the gripper at all**: its axis passes through the gripper origin,
so it only orients it. That was observed as "it only tilts the tip" before the URDF was read,
and is the strongest single confirmation that this URDF describes our arm.

## Install

Its own environment, separate from the repo's `.venv`, because ManiSkill pulls in torch,
SAPIEN and a Vulkan runtime and the test suite has to keep running on machines with no GPU.
Same split `robot_training/` uses.

```bash
./sim/setup_env.sh
source sim/venv/bin/activate
python -m sim.elrobot_agent
```

`sim/venv/` is git-ignored.

### What it needs

- **Disk**: roughly 6 to 8 GB. `setup_env.sh` refuses under 12 GB free rather than leaving a
  half-written environment that imports but crashes.
- **GPU**: ManiSkill's own support table says Linux with an NVIDIA GPU is the only fully
  supported combination (CPU sim, GPU sim and rendering). On Windows and macOS, GPU simulation
  does not work; under WSL, only CPU simulation does.
- **Vulkan**, which is not a pip package. SAPIEN renders through it:
  ```bash
  sudo apt install libvulkan1 vulkan-tools
  vulkaninfo --summary          # should list your GPU
  ```

### Not yet run here

The install has not been run on this machine, so these steps come from ManiSkill's own
documentation rather than from a working environment. The disk was at 96 percent when this was
written, which is below what `setup_env.sh` will accept. Expect to debug Vulkan at least once.

## Use

```bash
python -m sim.elrobot_agent                 # the rest pose, every joint centred
python -m sim.elrobot_agent --ticks 1:1976,2:1719,3:2526,4:2755,5:1907,6:1270,7:1282
```

`sim/elrobot_agent.py` registers the arm as a ManiSkill agent with uid `elrobot_follower`, and
`qpos_from_ticks()` converts a Station tick reading into ManiSkill's joint vector, reusing the
tick convention from `lib/kinematics.py` rather than restating it. The joint order is the
URDF's active joints: `rev_motor_01` through `rev_motor_08`, then the two gripper jaws.

That conversion is tested in the repo's normal `.venv` (`tests/sim/`), because it is ordinary
arithmetic and should not need a GPU to check.

## The leader arm

Only the follower has a URDF in NormaCore's repo. The leader is a different mechanism (7.4 V,
different gear variants, see `docs/elrobot.md`) and nothing here models it.
