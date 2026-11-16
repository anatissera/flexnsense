#!/usr/bin/env bash
# Phase 1, runs ON the workstation: turn the ElRobot URDF into a USD file that Isaac Sim / Isaac Lab can load.
#   cloud/nebius/sync.sh up                          (on the Mac: copies this script and the model)
#   cloud/nebius/ssh.sh 'bash ~/flexnsense_sim/01_import_elrobot.sh 2>&1 | tee ~/import_elrobot.log'
# Run on 2026-11-15 on Isaac Lab main@727e4e43a (flags there: --fix_base, no --headless). The converter's
# options differ between Isaac Lab versions, so the script prints its --help first; if the conversion fails, read that
# help in ~/import_elrobot.log and adjust the FLAGS line below.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
[ -f ~/isaac-env.sh ] || { echo "no ~/isaac-env.sh: Isaac Sim (Kit) is not enabled on this VM. Run setup_vm.sh with ACCEPT_OMNIVERSE_EULA=yes after accepting the NVIDIA Omniverse licence."; exit 1; }
source ~/isaac-env.sh
IN="$HOME/flexnsense-sim/assets/elrobot/elrobot_follower.urdf"
OUT="$HOME/flexnsense-sim/assets/elrobot/usd/elrobot_follower.usd"
[ -f "$IN" ] || { echo "missing $IN: run cloud/nebius/sync.sh up on the Mac"; exit 1; }
cd ~/IsaacLab
CONV=$(find scripts source -name 'convert_urdf.py' 2>/dev/null | head -1)
[ -n "$CONV" ] || { echo "no convert_urdf.py in this Isaac Lab checkout ($(git log -1 --format=%h)); look for the URDF importer in its docs"; exit 1; }
echo "===== converter: $CONV"
uv run --extra isaacsim python "$CONV" --help 2>&1 | tail -40
mkdir -p "$(dirname "$OUT")"
FLAGS="--fix_base --joint_target_type position"      # the base is bolted to the bench; servos are position-controlled
echo "===== converting with: $FLAGS"
uv run --extra isaacsim python "$CONV" "$IN" "$OUT" $FLAGS
RC=$?
echo "===== exit $RC"; ls -la "$(dirname "$OUT")" 2>/dev/null
exit $RC
