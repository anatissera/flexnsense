"""Find joint angles that put the ElRobot's gripper at a wanted place and direction, from the URDF alone.

    python3 tools/seed_pose.py X Y Z  PX PY PZ  CX CY CZ
        X Y Z      gripper origin in the arm's base frame [m] (the arm faces +y)
        PX PY PZ   direction the gripper should point
        CX CY CZ   direction the jaws should close along

Forward kinematics by hand (numpy only), then a damped least-squares search from several starts. Prints the best
angles for rev_motor_01..07 and the remaining error. Used to seed the simulator's inverse kinematics.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

root = ET.parse(Path(__file__).parent.parent / "assets/elrobot/elrobot_follower.urdf").getroot()
joints = {j.find("child").get("link"): j for j in root.findall("joint")}
chain, link = [], "Gripper_Base_v1_1"
while link in joints:
    chain.append(joints[link])
    link = joints[link].find("parent").get("link")
chain.reverse()
moving = [j for j in chain if j.get("type") == "revolute"]
lim = np.array([[float(j.find("limit").get("lower")), float(j.find("limit").get("upper"))] for j in moving])


def rot(axis, a):
    x, y, z = axis / np.linalg.norm(axis)
    c, s, C = np.cos(a), np.sin(a), 1 - np.cos(a)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def rpy(r, p, y):
    return rot(np.array([0, 0, 1.0]), y) @ rot(np.array([0, 1.0, 0]), p) @ rot(np.array([1.0, 0, 0]), r)


def fk(q):
    R, t, k = np.eye(3), np.zeros(3), 0
    for j in chain:
        o = j.find("origin")
        xyz = np.array([float(v) for v in o.get("xyz", "0 0 0").split()])
        R0 = rpy(*[float(v) for v in o.get("rpy", "0 0 0").split()])
        t, R = t + R @ xyz, R @ R0
        if j.get("type") == "revolute":
            R = R @ rot(np.array([float(v) for v in j.find("axis").get("xyz").split()]), q[k])
            k += 1
    return t, R


def err(q, pos, point, close):
    t, R = fk(q)
    return np.concatenate([t - pos, 0.12 * (R[:, 1] - point), 0.06 * (R[:, 0] - close)])  # local +y points, local x closes


def solve(pos, point, close, tries=60, seed=0):
    rng, best = np.random.default_rng(seed), (1e9, None)
    for _ in range(tries):
        q = rng.uniform(lim[:, 0] * 0.8, lim[:, 1] * 0.8)
        for _ in range(250):
            e = err(q, pos, point, close)
            J = np.stack([(err(q + np.eye(len(q))[i] * 1e-5, pos, point, close) - e) / 1e-5 for i in range(len(q))], axis=1)
            q = np.clip(q - J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(e)), e), lim[:, 0], lim[:, 1])
        e = err(q, pos, point, close)
        # prefer solutions away from the joint limits
        cost = np.linalg.norm(e) + 0.002 * np.sum((np.abs(q - lim.mean(1)) / (lim[:, 1] - lim[:, 0])) ** 2)
        if cost < best[0]:
            best = (cost, q.copy())
    return best[1]


if __name__ == "__main__":
    v = [float(a) for a in sys.argv[1:10]]
    pos, point, close = np.array(v[0:3]), np.array(v[3:6]), np.array(v[6:9])
    point, close = point / np.linalg.norm(point), close / np.linalg.norm(close)
    q = solve(pos, point, close)
    t, R = fk(q)
    print("joints rev_motor_01..07:", [round(float(a), 3) for a in q])
    print("gripper at", [round(float(a), 3) for a in t], "| position error mm", round(float(np.linalg.norm(t - pos)) * 1000, 1),
          "| pointing error deg", round(float(np.degrees(np.arccos(np.clip(R[:, 1] @ point, -1, 1)))), 1),
          "| closing error deg", round(float(np.degrees(np.arccos(np.clip(abs(R[:, 0] @ close), -1, 1)))), 1))
    print("zero pose: gripper at", [round(float(a), 3) for a in fk(np.zeros(len(q)))[0]], "pointing", [round(float(a), 2) for a in fk(np.zeros(len(q)))[1][:, 1]])
