"""Draw the bench of the two-arm tasks the way sim/08_physical_scale.py does: the hand with its fingers standing,
and the scale (rod, tube, purple band, white cap, C-shaped wire loop round the finger, three red points).

The scale is the PESOLA Medio-Line 40600 of the team's datasheet (docs/pesola-medio-line-40600-datasheet.pdf):
anodised aluminium tube 12 mm across, purple markings, 600 g over a 100 mm scale. The red marks are the ones the
team's gauge reader looks for (gauge_reader/stickers.py): a round red sticker just before the 0 g line, another just
past the 600 g line, and between them the marker, a red band across the tube that slides with the force.
Everything here is drawing; positions come from the task (flex_tasks/duo_env.py, team_env.py).
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_from_angle_axis

FLAT_LEN, KNUCKLE_Z, UP_LEN, FINGER_R = 0.075, 0.018, 0.062, 0.0095
TUBE_LEN, CAP_LEN, BAIL_LEG, BAIL_R = 0.165, 0.012, 0.032, 0.013
N_ROD, N_LEG, N_ARC = 6, 6, 12
MARKER, LIMIT = 9, 10  # places of "marker" and "limit" in the list of drawn parts below (they follow "dot", which is 8)
TUBE_R = 0.006  # the datasheet's 12 mm diameter
RED = (0.90, 0.05, 0.04)
# Along the tube, measured back from the loop: the marker sits at ZERO_AT with no force and SCALE_TRAVEL further at
# 600 g (the datasheet's 100 mm scale); the two reference stickers sit just outside that stretch.
ZERO_AT, SCALE_TRAVEL, STICKER_1, STICKER_2 = 0.080, 0.100, 0.070, 0.190
U0 = (ZERO_AT - STICKER_1) / (STICKER_2 - STICKER_1)  # where the 0 g line is between the stickers, as the reader counts it
U1 = (ZERO_AT + SCALE_TRAVEL - STICKER_1) / (STICKER_2 - STICKER_1)  # and the 600 g line


def make_markers(prim_path: str = "/Visuals/bench") -> VisualizationMarkers:
    mat = lambda c, m=0.0, e=0.0: sim_utils.PreviewSurfaceCfg(diffuse_color=c, metallic=m, emissive_color=tuple(e * v for v in c))
    skin = (0.86, 0.62, 0.50)
    return VisualizationMarkers(VisualizationMarkersCfg(prim_path=prim_path, markers={
        "flat": sim_utils.CylinderCfg(radius=FINGER_R, height=FLAT_LEN, visual_material=mat(skin)),
        "up": sim_utils.CylinderCfg(radius=FINGER_R, height=UP_LEN, visual_material=mat(skin)),
        "tip": sim_utils.SphereCfg(radius=FINGER_R, visual_material=mat(skin)),
        "palm": sim_utils.CuboidCfg(size=(0.105, 0.10, 0.030), visual_material=mat(skin)),
        "tube": sim_utils.CylinderCfg(radius=TUBE_R, height=TUBE_LEN, visual_material=mat((0.62, 0.65, 0.70), 0.75)),
        "band": sim_utils.CylinderCfg(radius=TUBE_R + 0.0004, height=0.022, visual_material=mat((0.42, 0.20, 0.52))),
        "cap": sim_utils.CylinderCfg(radius=0.0048, height=CAP_LEN, visual_material=mat((0.95, 0.95, 0.95))),
        "wire": sim_utils.SphereCfg(radius=0.0019, visual_material=mat((0.55, 0.57, 0.62), 0.7)),
        "dot": sim_utils.SphereCfg(radius=0.0045, visual_material=mat(RED, 0.0, 0.5)),  # a round reference sticker
        "marker": sim_utils.CylinderCfg(radius=TUBE_R + 0.0012, height=0.0045, visual_material=mat(RED, 0.0, 0.5)),  # the sliding band
        "limit": sim_utils.CylinderCfg(radius=0.0081, height=0.002, visual_material=mat((0.05, 0.07, 0.10))),  # the force limit on the tube
    }))


def _z_to(d: torch.Tensor) -> torch.Tensor:
    """Quaternions (x, y, z, w) turning +Z onto each unit vector d (N, 3)."""
    z = torch.tensor([0.0, 0.0, 1.0], device=d.device).expand_as(d)
    axis = torch.linalg.cross(z, d)
    norm = axis.norm(dim=-1, keepdim=True)
    axis = torch.where(norm > 1e-6, axis / norm.clamp(min=1e-6), torch.tensor([1.0, 0.0, 0.0], device=d.device).expand_as(d))
    return quat_from_angle_axis(torch.acos(d[:, 2].clamp(-1.0, 1.0)), axis)


def draw(env, marks: VisualizationMarkers, grow: float = 1.0) -> None:
    """Place every drawn part for every table. `grow` enlarges the red points for views from far away."""
    dev, n, o = env.device, env.num_envs, env.scene.env_origins
    tip_a, _, _ = env._tips()
    up = env.up.expand(n, 3)
    home, u0, hd = env.home, env.u0, env.hdir
    ident = torch.zeros(n, 4, device=dev)
    ident[:, 3] = 1.0
    pos, quat, idx, scl = [], [], [], []

    def add(kind: int, p: torch.Tensor, q: torch.Tensor | None = None, s: float = 1.0):
        pos.append(p + o)
        quat.append(ident if q is None else q)
        idx.append(torch.full((n,), kind, dtype=torch.long, device=dev))
        scl.append(torch.full((n, 3), s, device=dev))

    # --- the hand: four fingers in a row, each lying flat and then standing up; the one in the loop moves when it pulls
    on = env._on_finger()
    _, u_on = env._finger(on)
    pull = env._pull_now().unsqueeze(-1) * u_on
    q_flat = _z_to(u0)
    table = torch.zeros(n, 3, device=dev)
    for j in range(4):
        stand = home + u0 * env.cfg.rest + hd * env.cfg.finger_gap * j
        stand = stand + torch.where(((on == j) & ~env.held).unsqueeze(-1), pull, torch.zeros_like(pull))
        stand = torch.cat([stand[:, 0:2], table[:, 2:3]], dim=-1)
        add(0, stand + u0 * (FLAT_LEN / 2) + up * KNUCKLE_Z, q_flat)
        add(1, stand + up * (KNUCKLE_Z + UP_LEN / 2))
        add(2, stand + up * (KNUCKLE_Z + UP_LEN))
        add(2, stand + up * KNUCKLE_Z)
    palm = home + u0 * (env.cfg.rest + FLAT_LEN + 0.05) + hd * env.cfg.finger_gap * 1.5
    palm = torch.cat([palm[:, 0:2], table[:, 2:3] + 0.015], dim=-1)
    add(3, palm, quat_from_angle_axis(torch.atan2(u0[:, 1], u0[:, 0]) - math.pi / 2, up))
    # --- the scale, from the pull arm's jaws to the loop
    d = env.loop - tip_a
    d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    q_d = _z_to(d)
    tube_end = env.loop - d * (BAIL_LEG + CAP_LEN)
    tube_start = tube_end - d * TUBE_LEN
    add(4, tube_end - d * (TUBE_LEN / 2), q_d)
    add(5, tube_end - d * 0.014, q_d)
    add(6, env.loop - d * (BAIL_LEG + CAP_LEN / 2), q_d)
    for k in range(N_ROD):  # the swivel and rod between the jaws and the tube
        add(7, tip_a + (tube_start - tip_a) * (k / (N_ROD - 1)))
    lat = torch.linalg.cross(up, d)
    lat = lat / lat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    for sgn in (-1.0, 1.0):  # the two legs of the wire loop
        for k in range(N_LEG):
            add(7, env.loop - d * (BAIL_LEG * (1 - k / (N_LEG - 1))) + lat * (sgn * BAIL_R))
    for k in range(N_ARC):  # and its arc, round the finger
        a = math.pi * k / (N_ARC - 1)
        add(7, env.loop + d * (BAIL_R * math.sin(a)) + lat * (BAIL_R * math.cos(a)))
    top = up * (TUBE_R + 0.0005)
    add(8, env.loop - d * STICKER_1 + top, None, grow)   # reference sticker just before the 0 g line
    add(8, env.loop - d * STICKER_2 + top, None, grow)   # reference sticker just past the 600 g line
    indicator = env._indicator(tip_a) if hasattr(env, "_indicator") else env.loop - d * ZERO_AT
    add(MARKER, indicator, q_d, 1.0)                      # the marker: a red band across the tube, at the force
    at_limit = SCALE_TRAVEL * env.cfg.cap / 5.886
    add(LIMIT, env.loop - d * (ZERO_AT + at_limit), q_d)  # a dark ring where the marker is at the limit
    marks.visualize(translations=torch.cat(pos), orientations=torch.cat(quat), scales=torch.cat(scl), marker_indices=torch.cat(idx))
