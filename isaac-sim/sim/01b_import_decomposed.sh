#!/usr/bin/env bash
# Runs ON the workstation: convert the ElRobot URDF again, this time with collision shapes that follow the parts
# closely ("Convex Decomposition", made from the visual meshes) and not one coarse hull per part. With one hull per
# part the two jaws are solid wedges and cannot close on a thin object: it gets wedged, not pinched.
#   cloud/nebius/ssh.sh 'bash ~/flexnsense_sim/01b_import_decomposed.sh 2>&1 | tee ~/import_decomposed.log'
# Isaac Lab's converter tool has no flag for the collision type, so this runs a copy of the tool with two settings added.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/isaac-env.sh
IN="$HOME/flexnsense-sim/assets/elrobot/elrobot_follower.urdf"
OUT="$HOME/flexnsense-sim/assets/elrobot/usd_decomposed/elrobot_follower.usd"
cd ~/IsaacLab
TOOL=/tmp/convert_urdf_decomposed.py
sed 's|        usd_dir=dest_path,|        usd_dir=dest_path,\n        collision_from_visuals=True,\n        collision_type="Convex Decomposition",|' scripts/tools/convert_urdf.py > "$TOOL"
grep -c 'Convex Decomposition' "$TOOL" | sed 's/^/settings added: /'
mkdir -p "$(dirname "$OUT")"
uv run --extra isaacsim python "$TOOL" "$IN" "$OUT" --fix_base --joint_target_type position
RC=$?
echo "===== exit $RC"; find "$(dirname "$OUT")" -name '*.usd*' | head -12; du -sh "$(dirname "$OUT")"
exit $RC
