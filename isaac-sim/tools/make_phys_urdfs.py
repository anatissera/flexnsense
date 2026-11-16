"""Write the two arm models of the physical-contact task from NormaCore's URDF (which is left unmodified).

  elrobot_pull_scale.urdf   the pull arm with the PESOLA as real bodies on its gripper: a swivel (two free hinges), the
                            tube, and the sliding rod with the wire loop on a spring (58.86 N/m, the datasheet's 600 g
                            over 100 mm). The loop is a ring of eight small bars that a finger can pass through.
  elrobot_move_pads.urdf    the move arm with a flat rubber pad on each jaw (where NormaCore's mesh has its inner face).

    python3 tools/make_phys_urdfs.py
"""
import math
from pathlib import Path

HERE = Path(__file__).parent.parent / "assets/elrobot"
src = (HERE / "elrobot_follower.urdf").read_text()
assert src.rstrip().endswith("</robot>")
body = src.rstrip()[: -len("</robot>")]

TIP = 0.10            # m along the gripper's +y: where the jaws hold the scale's swivel
TUBE_FROM, TUBE_TO = 0.021, 0.186   # m from the swivel: the tube
TUBE_R = 0.006        # 12 mm across (datasheet)
LOOP_AT = 0.225       # m from the swivel: centre of the wire loop without load (datasheet: 225 mm)
RING_R, WIRE = 0.017, 0.004   # m: ring centre-line radius and bar thickness (inner radius 15 mm)
box = lambda name, size, xyz, rpy="0 0 0", rgba="0.75 0.77 0.8 1": f"""
    <visual><origin xyz="{xyz}" rpy="{rpy}"/><geometry><box size="{size}"/></geometry><material name="{name}"><color rgba="{rgba}"/></material></visual>
    <collision><origin xyz="{xyz}" rpy="{rpy}"/><geometry><box size="{size}"/></geometry></collision>"""
inert = lambda m, i, xyz="0 0 0": f"""
    <inertial><origin xyz="{xyz}" rpy="0 0 0"/><mass value="{m}"/><inertia ixx="{i[0]}" iyy="{i[1]}" izz="{i[2]}" ixy="0" ixz="0" iyz="0"/></inertial>"""

L = TUBE_TO - TUBE_FROM
mid = (TUBE_FROM + TUBE_TO) / 2
ring = ""
for k in range(8):
    a = k * math.pi / 4
    ring += box(f"ring{k}", f"{2 * RING_R * math.tan(math.pi / 8) + 0.002:.4f} {WIRE} {WIRE}",
                f"{RING_R * math.sin(a):.5f} {LOOP_AT + RING_R * math.cos(a):.5f} 0", f"0 0 {-a:.5f}")
stem_from, stem_to = TUBE_TO - 0.02, LOOP_AT - RING_R
scale = f"""
  <link name="scale_swivel">{inert(0.003, (2e-6, 2e-6, 2e-6))}
  </link>
  <joint name="scale_yaw" type="revolute">
    <origin xyz="0 {TIP} 0" rpy="0 0 0"/><parent link="Gripper_Base_v1_1"/><child link="scale_swivel"/>
    <axis xyz="0 0 1"/><limit lower="-1.3" upper="1.3" effort="0.0" velocity="20"/>
  </joint>
  <link name="scale_tube">{inert(0.028, (6.4e-5, 1.0e-5, 6.4e-5), f"0 {mid} 0")}
    <visual><origin xyz="0 {mid} 0" rpy="1.5707963 0 0"/><geometry><cylinder radius="{TUBE_R}" length="{L}"/></geometry><material name="tube"><color rgba="0.55 0.36 0.66 1"/></material></visual>
    <collision><origin xyz="0 {mid} 0" rpy="1.5707963 0 0"/><geometry><cylinder radius="{TUBE_R}" length="{L}"/></geometry></collision>
  </link>
  <joint name="scale_pitch" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/><parent link="scale_swivel"/><child link="scale_tube"/>
    <axis xyz="1 0 0"/><limit lower="-1.2" upper="1.2" effort="0.0" velocity="20"/>
  </joint>
  <link name="scale_hook">{inert(0.006, (3e-6, 3e-6, 3e-6), f"0 {LOOP_AT - 0.01} 0")}{box("stem", f"{WIRE} {stem_to - stem_from:.4f} {WIRE}", f"0 {(stem_from + stem_to) / 2:.5f} 0")}{ring}
  </link>
  <joint name="scale_slide" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/><parent link="scale_tube"/><child link="scale_hook"/>
    <axis xyz="0 1 0"/><limit lower="0.0" upper="0.10" effort="10.0" velocity="2.0"/>
  </joint>
"""
(HERE / "elrobot_pull_scale.urdf").write_text(body + scale + "</robot>\n")

# pads: in the gripper's frame the fingers' inner faces are at x = +0.0275 (jaw 01) and -0.0247 (jaw 02), 82 mm along
JOINT_ORIGIN = {"Gripper_Jaw_01_v1_1": (0.308573, -0.243367, -0.049391), "Gripper_Jaw_02_v1_1": (0.311734, -0.199648, 0.032849)}
pads = ""
for jaw, face_x in (("Gripper_Jaw_01_v1_1", 0.0275), ("Gripper_Jaw_02_v1_1", -0.0247)):
    o = JOINT_ORIGIN[jaw]
    xyz = f"{face_x - o[0]:.6f} {0.082 - o[1]:.6f} {0.0 - o[2]:.6f}"
    pads += f"""
  <link name="pad_{jaw[8:14]}">{inert(0.002, (1e-6, 1e-6, 1e-6), xyz)}{box("pad", "0.004 0.040 0.026", xyz, rgba="0.05 0.05 0.06 1")}
  </link>
  <joint name="pad_fix_{jaw[8:14]}" type="fixed"><origin xyz="0 0 0" rpy="0 0 0"/><parent link="{jaw}"/><child link="pad_{jaw[8:14]}"/></joint>
"""
(HERE / "elrobot_move_pads.urdf").write_text(body + pads + "</robot>\n")
print("wrote", HERE / "elrobot_pull_scale.urdf", "and", HERE / "elrobot_move_pads.urdf")
