#!/usr/bin/env python
"""Lleva el follower a su pose de home (home_pose.json) y suelta el torque al llegar.

    python -s go_home.py --check      # dónde está y a cuánto de home; no mueve nada
    python -s go_home.py              # MUEVE el brazo, sin preguntar (lo lanza un botón)

Tres fases, para no arrastrar el dedo si el gripper tiene la anilla cogida:
  1. abrir: solo el gripper, hasta `gripper_open`; el resto quieto.
  2. brazo: motores 1-7 a home con el gripper abierto.
  3. cerrar: el gripper a su valor de home.

Los goals pasan por GoalSafety (safety.py): avanzan como mucho --max-step-ticks por tick y se
retienen si un motor se queda bloqueado. Si alguna fase no llega a tiempo el script sale con 1
y el brazo se queda CON torque, sujetando donde esté: el torque solo se suelta en home, que es
una pose de reposo.

Código de salida: 0 = en home (y sin torque, salvo --keep-torque); 1 = no llegó.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path

import numpy as np

from safety import GoalSafety

HOME_POSE = Path(__file__).with_name("home_pose.json")
PHASES = ("abrir", "brazo", "cerrar")
PERIOD_S = 0.1      # 10 Hz, el ritmo del bucle de control de la policy


def pidfile_path(bus_serial: str) -> Path:
    """Donde run_local.py / robot_client.py dejan su PID mientras controlan ese bus."""
    return Path(tempfile.gettempdir()) / f"run_local-{bus_serial}.pid"


def load_home(path: str | Path = HOME_POSE) -> tuple[str, dict[int, int], int | None]:
    """(bus_serial, {motor: tick de home}, tick de gripper abierto o None)."""
    raw = json.loads(Path(path).read_text())
    home = {int(k): int(v) for k, v in raw["ticks"].items()}
    is_open = raw.get("gripper_open")
    return raw["bus_serial"], home, None if is_open is None else int(is_open)


def phase_target(phase: str, positions: dict[int, int], home: dict[int, int],
                 gripper_open: int | None) -> tuple[dict[int, int], list[int]]:
    """Objetivo de una fase y los motores que tienen que llegar para darla por hecha.

    `positions` es la posición al EMPEZAR la fase: lo que no se mueve en ella se queda ahí.
    El gripper es el motor de ID más alto. Sin `gripper_open` no se toca hasta la fase "cerrar".
    """
    motors = sorted(home)
    gripper, arm = motors[-1], motors[:-1]
    held_gripper = positions[gripper] if gripper_open is None else gripper_open
    if phase == "abrir":
        return {**{m: int(positions[m]) for m in arm}, gripper: int(held_gripper)}, [gripper]
    if phase == "brazo":
        return {**{m: home[m] for m in arm}, gripper: int(held_gripper)}, arm
    if phase == "cerrar":
        return dict(home), motors
    raise ValueError(f"fase desconocida: {phase}")


def reached(positions: dict[int, int], target: dict[int, int], tolerance: int, joints: list[int]) -> bool:
    return all(abs(int(positions[m]) - target[m]) <= tolerance for m in joints)


def worst(positions: dict[int, int], target: dict[int, int], joints: list[int]) -> tuple[int, int]:
    """(motor, ticks) del motor más lejos de su objetivo."""
    m = max(joints, key=lambda j: abs(int(positions[j]) - target[j]))
    return m, abs(int(positions[m]) - target[m])


def phase_timeout(positions: dict[int, int], target: dict[int, int], max_step: int) -> float:
    """El doble de lo que tarda el goal en recorrer la distancia, más margen para asentarse."""
    far = max(abs(int(positions[m]) - target[m]) for m in target)
    return 2.0 * far / max_step * PERIOD_S + 4.0


def print_table(positions: dict[int, int], home: dict[int, int]) -> None:
    print(f"{'motor':<7}{'ahora':>8}{'home':>8}{'delta':>9}")
    for m in sorted(home):
        print(f"{m:<7}{int(positions[m]):>8}{home[m]:>8}{home[m] - int(positions[m]):>+9}")


async def go_home(args) -> int:
    import station_io as io     # aquí y no arriba: necesita norma-core, y los tests no lo tienen

    bus_file, home, gripper_open = load_home(args.home_pose)
    bus = args.bus_serial or bus_file
    motor_ids = sorted(home)
    client = await io.new_station_client(args.station, logging.getLogger("go_home"))
    positions = await io.read_bus_positions(client, bus, motor_ids)
    print_table(positions, home)
    if args.check:
        m, d = worst(positions, home, motor_ids)
        print(f"el motor {m} es el que más lejos está de home: {d} ticks")
        return 0

    if gripper_open is None:
        print("AVISO: home_pose.json no tiene gripper_open; el gripper NO se abre antes de mover el brazo.")
    await io.safe_enable_torque(client, bus, motor_ids)      # goal = posición actual, luego torque
    safety = GoalSafety(n_joints=len(motor_ids), gripper_index=len(motor_ids) - 1,
                        max_step=args.max_step_ticks, gripper_max_step=args.gripper_max_step_ticks)
    for phase in PHASES:
        positions = await io.read_bus_positions(client, bus, motor_ids)
        target, joints = phase_target(phase, positions, home, gripper_open)
        if reached(positions, target, args.tolerance, joints):
            continue
        deadline = time.monotonic() + phase_timeout(positions, target, args.max_step_ticks)
        want = np.array([target[m] for m in motor_ids], dtype=np.int64)
        print(f"fase '{phase}' ...")
        while not reached(positions, target, args.tolerance, joints):
            if time.monotonic() > deadline:
                m, d = worst(positions, target, joints)
                print(f"NO llegó: fase '{phase}', el motor {m} se queda a {d} ticks. "
                      "El brazo sigue CON torque, sujetando donde está.")
                return 1
            goal, msg = safety.apply(want, np.array([positions[m] for m in motor_ids], dtype=np.int64))
            if goal is not None:
                await io.send_commands(client, [io.goal_command(bus, motor_ids, goal.tolist())])
            if msg and "retenido" in msg:
                print(f"  {msg}")
            await asyncio.sleep(PERIOD_S)
            positions = await io.read_bus_positions(client, bus, motor_ids)

    m, d = worst(positions, home, motor_ids)
    print(f"en home (el motor {m} es el más lejano, a {d} ticks).")
    if args.keep_torque:
        print("el follower mantiene el torque (--keep-torque).")
        return 0
    await asyncio.sleep(0.5)
    await io.set_torque(client, bus, motor_ids, False)
    print("torque del follower DESACTIVADO.")
    return 0


def add_home_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--station", default="localhost", help="host[:puerto] del Station")
    p.add_argument("--bus-serial", default=None, help="bus del follower (por defecto, el de home_pose.json)")
    p.add_argument("--home-pose", default=str(HOME_POSE))
    p.add_argument("--check", action="store_true", help="solo informa; no mueve nada")
    p.add_argument("--keep-torque", action="store_true", help="no soltar el torque al llegar a home")
    p.add_argument("--max-step-ticks", type=int, default=40,
                   help="máx. avance del goal por tick de 0.1 s en el brazo (40 son unos 35 grados/s)")
    p.add_argument("--gripper-max-step-ticks", type=int, default=100)
    p.add_argument("--tolerance", type=int, default=60,
                   help="ticks a los que un motor cuenta como llegado (los motores 3 y 4 ya cuelgan ~41 por gravedad)")


def run_go_home(args) -> int:
    try:
        return asyncio.run(go_home(args))
    except KeyboardInterrupt:
        print("\ninterrumpido: el brazo se queda con el torque que tuviera.")
    except Exception as e:      # Station caído, bus desconectado...: no se toca el torque
        print(f"ERROR: {e}")
    return 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_home_args(p)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return run_go_home(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
