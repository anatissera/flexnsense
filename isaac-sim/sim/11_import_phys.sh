#!/usr/bin/env bash
# Runs ON the workstation: convert the two arm models of the physical-contact task (tools/make_phys_urdfs.py wrote them).
#   usd_pull_scale/   the pull arm with the scale as real bodies (collision shapes as written in the URDF)
#   usd_move_pads/    the move arm with jaw pads (collision shapes that follow the parts), gripper joints corrected
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/isaac-env.sh
A="$HOME/flexnsense-sim/assets/elrobot"
cd ~/IsaacLab
TOOL=/tmp/convert_urdf_decomposed.py
sed 's|        usd_dir=dest_path,|        usd_dir=dest_path,\n        collision_from_visuals=True,\n        collision_type="Convex Decomposition",|' scripts/tools/convert_urdf.py > "$TOOL"
rm -rf "$A/usd_pull_scale" "$A/usd_move_pads"
uv run --extra isaacsim python scripts/tools/convert_urdf.py "$A/elrobot_pull_scale.urdf" "$A/usd_pull_scale/elrobot_follower.usd" --fix_base --joint_target_type position 2>&1 | tail -3
uv run --extra isaacsim python "$TOOL" "$A/elrobot_move_pads.urdf" "$A/usd_move_pads/elrobot_follower.usd" --fix_base --joint_target_type position 2>&1 | tail -3
FIX_FOLDERS=usd_pull_scale,usd_move_pads uv run --extra isaacsim python ~/flexnsense_sim/09_fix_gripper_usd.py 2>&1 | grep "FIX\|Error\|Traceback" -A3
find "$A/usd_pull_scale" "$A/usd_move_pads" -name '*.usda' | head
