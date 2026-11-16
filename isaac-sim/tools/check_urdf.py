"""Check the ElRobot URDF without any simulator: joints, limits, masses, and that every mesh file exists.

    python3 tools/check_urdf.py [assets/elrobot/elrobot_follower.urdf]

Standard library only. Exit code 1 if a mesh is missing or a moving link has no mass.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent.parent / "assets/elrobot/elrobot_follower.urdf")
root = ET.parse(path).getroot()
links = {l.get("name"): l for l in root.findall("link")}
joints = root.findall("joint")
problems = []

print(f"robot: {root.get('name')}   links: {len(links)}   joints: {len(joints)}\n")
print(f"{'joint':28} {'type':10} {'parent -> child':44} {'axis':12} {'lower':>7} {'upper':>7} {'effort':>7} {'vel':>6}  dynamics")
for j in joints:
    lim, ax, dyn = j.find("limit"), j.find("axis"), j.find("dynamics")
    g = lambda e, k: "-" if e is None or e.get(k) is None else e.get(k)
    pc = f"{j.find('parent').get('link')} -> {j.find('child').get('link')}"
    print(f"{j.get('name'):28} {j.get('type'):10} {pc:44} {g(ax,'xyz'):12} {g(lim,'lower'):>7} {g(lim,'upper'):>7} "
          f"{g(lim,'effort'):>7} {g(lim,'velocity'):>6}  {'' if dyn is None else dict(dyn.attrib)}")
    if j.get("type") in ("revolute", "prismatic") and lim is None:
        problems.append(f"joint {j.get('name')}: no <limit>")
    if j.find("mimic") is not None:
        print(f"{'':28} mimic of {j.find('mimic').get('joint')} x {j.find('mimic').get('multiplier', '1')}")

total = 0.0
print(f"\n{'link':28} {'mass kg':>8}  inertia  visual/collision meshes")
for name, l in links.items():
    inertial = l.find("inertial")
    mass = None if inertial is None or inertial.find("mass") is None else float(inertial.find("mass").get("value"))
    total += mass or 0.0
    meshes = []
    for kind in ("visual", "collision"):
        for m in l.findall(f"{kind}/geometry/mesh"):
            f = m.get("filename")
            rel = f.split("://", 1)[-1] if "://" in f else f
            ok = any((path.parent / c).exists() for c in (rel, Path(rel).name, Path("assets") / Path(rel).name))
            meshes.append(f"{kind[0]}:{Path(rel).name}{'' if ok else ' MISSING'}")
            if not ok:
                problems.append(f"link {name}: mesh not found: {f}")
    has_i = inertial is not None and inertial.find("inertia") is not None
    print(f"{name:28} {'-' if mass is None else format(mass, '.4f'):>8}  {'yes' if has_i else 'NO ':7}  {' '.join(meshes)}")
    if mass is None and any(j.find('child').get('link') == name and j.get('type') != 'fixed' for j in joints):
        problems.append(f"link {name}: moves but has no mass")
print(f"\ntotal mass: {total:.3f} kg")
print("\nproblems:" if problems else "\nno problems found")
for p in problems:
    print("  -", p)
sys.exit(1 if problems else 0)
