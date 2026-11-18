"""Conversión acción -> ticks y límites de seguridad del goal (solo numpy, sin Station).

Separado de station_io.py para poder probarlo sin norma-core ni robot.
"""

from __future__ import annotations

from collections import deque

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
    3) Motor bloqueado: los servos del ElRobot entran en protección por sobrecarga (status
       0x20) si la carga pasa del 25 % durante 2 s (EEPROM: overload_torque 25, protection_time
       200), y con un servo en ese estado Station deja de publicar frames: el script se para.
       Medido en el brazo: carga ~ 4.3 x |goal - posición| (en reposo, solo por la gravedad,
       los motores 3 y 4 ya van a ~41 ticks y 18 %), así que el 25 % son unos 58 ticks.
       Le pasa a cualquier motor que la policy manda contra algo (la mano, la mesa, la anilla).
       Para evitarlo, el goal de un motor se retiene en `posición + hold_lead` ticks (hacia
       donde empujaba) cuando
         a) lleva `stall_ticks` ticks seguidos sin moverse (< 15 ticks) con el goal enviado a
            más de `lead` ticks de su posición, o
         b) solo el gripper: su corriente llega a `gripper_current` (la regla del teleop con el
            que se grabaron las demos, st3215-remote-teleop-py/mirror.py, umbral 100).
       Se libera cuando la policy pide ir hacia el otro lado o el motor se vuelve a mover
       (el obstáculo ya no está).
       Dataset, gripper bloqueado con el goal por delante: adelanto mediana 60, corriente
       mediana 25 y máx 94, nunca más de 3.5 s seguidos.
    """

    def __init__(self, n_joints: int = 8, gripper_index: int = 7, max_step: int = 150,
                 gripper_max_step: int = 400, max_delta: int = 600, gripper_current: int = 100,
                 lead: int = 55, stall_ticks: int = 8, hold_lead: int = 45):
        self.step = np.full(n_joints, max_step, dtype=np.int64)
        self.step[gripper_index] = gripper_max_step
        self.arm = np.ones(n_joints, dtype=bool)
        self.arm[gripper_index] = False
        self.max_delta = max_delta
        self.prev: np.ndarray | None = None
        self.gripper = gripper_index
        self.gripper_current = gripper_current
        self.lead, self.stall_ticks, self.hold_lead = lead, stall_ticks, hold_lead
        self.blocked = np.zeros(n_joints, dtype=np.int64)   # dirección retenida por motor: -1, 0 o +1
        self.hold = np.zeros(n_joints, dtype=np.int64)      # goal retenido
        self.hold_pos = np.zeros(n_joints, dtype=np.int64)  # posición al retener
        self.stalled = np.zeros(n_joints, dtype=np.int64)   # ticks seguidos bloqueado
        self.stall_pos = np.zeros(n_joints, dtype=np.int64) # posición al empezar a contar
        self.why = [""] * n_joints
        self.history: deque = deque(maxlen=40)              # (posición, goal enviado) de los últimos ticks

    def describe_history(self) -> str:
        """goal - posición de cada motor en los últimos ticks, para ver qué pasó antes de un fallo."""
        lines = [f"  motor {j + 1}: pos {int(self.history[-1][0][j])} | goal - pos: "
                 + " ".join(str(int(g[j] - p[j])) for p, g in self.history)
                 for j in range(len(self.blocked))] if self.history else []
        return "\n".join(lines)

    def _hold_stalled(self, target: np.ndarray, positions: np.ndarray, currents) -> tuple[np.ndarray, str]:
        if self.lead <= 0 and self.gripper_current <= 0:
            return target, ""
        target = np.asarray(target).copy()
        sent = target if self.prev is None else self.prev
        notes = []
        for j in range(len(target)):
            pushing = int(sent[j] - positions[j])
            # Una vez contando, basta con seguir por encima de hold_lead: si el adelanto oscila
            # alrededor de `lead` (visto en el gripper al abrir contra la anilla: 75, 52, 70...)
            # la cuenta no se reinicia, porque el servo sigue acumulando sobrecarga.
            limit = min(self.lead, self.hold_lead) if self.stalled[j] else self.lead
            # "Parado" = menos de 15 ticks desde que empezó la cuenta: un motor que cede despacio
            # contra algo elástico (la anilla) también acumula sobrecarga.
            if self.lead > 0 and abs(pushing) > limit and abs(int(positions[j]) - self.stall_pos[j]) < 15:
                self.stalled[j] += 1
            else:
                self.stalled[j], self.stall_pos[j] = 0, positions[j]
            why = ""
            if j == self.gripper and currents is not None and 0 < self.gripper_current <= currents[j]:
                why = f"corriente {int(currents[j])}"
            elif self.stall_ticks and self.stalled[j] >= self.stall_ticks:
                why = f"parado {self.stalled[j]} ticks con el goal a {abs(pushing)}"
            if why and not self.blocked[j] and pushing:
                self.blocked[j] = int(np.sign(pushing))
                self.hold_pos[j], self.why[j] = positions[j], why
            if self.blocked[j]:
                d = int(self.blocked[j])
                advanced = (int(positions[j]) - self.hold_pos[j]) * d > 30      # el obstáculo ya no está
                # El goal retenido sigue a la posición: si la anilla empuja el motor hacia atrás,
                # un goal fijo quedaría cada vez más lejos (visto: retenido a 45, muerto a 80).
                hold = int(positions[j]) + d * self.hold_lead
                if not advanced and (target[j] - hold) * d > 0:
                    target[j] = self.hold[j] = hold
                    self.stalled[j], self.stall_pos[j] = 0, positions[j]
                    notes.append(f"motor {j + 1} retenido en {hold} ({self.why[j]})")
                else:
                    self.blocked[j] = 0
        return target, ", ".join(notes)

    def apply(self, target: np.ndarray, positions: np.ndarray, currents=None) -> tuple[np.ndarray | None, str]:
        target, held = self._hold_stalled(np.asarray(target), positions, currents)
        base = positions if self.prev is None else self.prev
        goal = base + np.clip(target - base, -self.step, self.step)
        delta = np.abs(goal - positions)
        if self.max_delta > 0 and (delta[self.arm] > self.max_delta).any():
            j = int(np.argmax(np.where(self.arm, delta, -1)))
            return None, f"ABORT tick: motor {j + 1} |goal-pos|={int(delta[j])} > {self.max_delta}"
        self.prev = goal
        self.history.append((np.asarray(positions).copy(), goal.copy()))
        clipped = int((goal != target).sum())
        notes = [f"{clipped} joint(s) limitados por velocidad"] if clipped else []
        return goal, ", ".join(notes + ([held] if held else []))
