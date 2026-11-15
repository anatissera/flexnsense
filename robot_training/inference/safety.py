"""Conversión acción -> ticks y límites de seguridad del goal (solo numpy, sin Station).

Separado de station_io.py para poder probarlo sin norma-core ni robot.
"""

from __future__ import annotations

import numpy as np

def goal_norm_to_ticks(goal_norm: np.ndarray, ranges: list[tuple[int, int]]) -> np.ndarray:
    """goal_norm [0,1] -> ticks con (range_min, range_max) del frame (igual que run_policy.py), recortado al rango."""
    out = []
    for g, (rmin, rmax) in zip(np.clip(goal_norm, 0.0, 1.0), ranges):
        out.append(max(rmin, min(rmax, int(round(rmin + float(g) * (rmax - rmin))))))
    return np.asarray(out, dtype=np.int64)


class GoalSafety:
    """Protección en dos niveles, calibrada con el dataset (ticks, a 10 Hz):

    1) Límite de velocidad: el goal enviado no cambia más de `max_step` ticks por tick
       respecto al goal anterior (el primero parte de la posición actual).
       Dataset, |Δgoal| p99 por tick: brazo 36-104, gripper 265.
    2) Abort: si en algún joint del BRAZO |goal - posición| > `max_delta`, el tick no se envía.
       El gripper se excluye: al apretar el objeto el goal va muy por delante
       (|goal-pos| p99 = 712, máx 1030); en el brazo el máx del dataset es ~490.
    """

    def __init__(self, n_joints: int = 8, gripper_index: int = 7, max_step: int = 150,
                 gripper_max_step: int = 400, max_delta: int = 600):
        self.step = np.full(n_joints, max_step, dtype=np.int64)
        self.step[gripper_index] = gripper_max_step
        self.arm = np.ones(n_joints, dtype=bool)
        self.arm[gripper_index] = False
        self.max_delta = max_delta
        self.prev: np.ndarray | None = None

    def apply(self, target: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray | None, str]:
        base = positions if self.prev is None else self.prev
        goal = base + np.clip(target - base, -self.step, self.step)
        delta = np.abs(goal - positions)
        if self.max_delta > 0 and (delta[self.arm] > self.max_delta).any():
            j = int(np.argmax(np.where(self.arm, delta, -1)))
            return None, f"ABORT tick: motor {j + 1} |goal-pos|={int(delta[j])} > {self.max_delta}"
        self.prev = goal
        clipped = int((goal != target).sum())
        return goal, (f"{clipped} joint(s) limitados por velocidad" if clipped else "")
