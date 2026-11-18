#!/usr/bin/env bash
# ManiSkill in its own virtual environment, for simulating the ElRobot from NormaCore's URDF.
#
# Separate from the repo's .venv on purpose: ManiSkill pulls in torch, SAPIEN and a Vulkan
# runtime, several gigabytes, and the test suite has to keep running on a machine with no GPU.
# Same split arm2/training/ uses for its LeRobot environment.
#
#   ./sim/setup_env.sh
#   source sim/venv/bin/activate
#   python -m sim.elrobot_agent
#
# Before running: check you have the disk space. torch with CUDA alone is about 3 GB.
set -euo pipefail

cd "$(dirname "$0")"

FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
if [ "${FREE_GB:-0}" -lt 12 ]; then
  echo "Only ${FREE_GB} GB free. ManiSkill plus torch needs roughly 6 to 8 GB, and a tight" >&2
  echo "disk is how a half-written install leaves an environment that imports but crashes." >&2
  echo "Free some space, or set SIM_SETUP_FORCE=1 to go ahead anyway." >&2
  [ "${SIM_SETUP_FORCE:-0}" = "1" ] || exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; the repo's SETUP.md uses it for the main environment too." >&2
  exit 1
fi

[[ -d venv ]] || uv venv --python 3.12 venv
# shellcheck disable=SC1091
source venv/bin/activate

# torch first: ManiSkill does not pin it, and letting pip resolve it afterwards can pull a
# build that does not match the local CUDA.
uv pip install torch
uv pip install --upgrade mani_skill

echo
echo "Installed. Check the pieces that are easy to get wrong:"
python - <<'PY'
import importlib, sys
for mod in ("torch", "sapien", "mani_skill"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:<12} {getattr(m, '__version__', 'ok')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {mod:<12} FAILED: {exc}")
        sys.exit(1)
try:
    import torch
    print(f"  cuda         {'available' if torch.cuda.is_available() else 'NOT available (CPU sim only)'}")
except Exception as exc:  # noqa: BLE001
    print(f"  cuda         could not check: {exc}")
PY

cat <<'EOF'

Vulkan is the other half, and it is not a pip package: SAPIEN renders through it.
If the viewer fails with a Vulkan or "no device" error:

  Ubuntu/Debian:  sudo apt install libvulkan1 vulkan-tools
  then:           vulkaninfo --summary          # should list your GPU

Next:
  source sim/venv/bin/activate
  python -m sim.elrobot_agent

For "which motor moves the gripper along the tether", do NOT use this: run
scripts/arm/which_motor.py in the normal .venv. Same URDF, exact answer, no GPU.
EOF
