"""Bucle de control común a run_local.py y robot_client.py.

Por cada frame nuevo de inference/normvla (~10 Hz, el mismo ritmo del dataset):
  1. obs = state (position_norm 8) + 3 JPEG; las cámaras se reordenan a (side, gripper, table).
  2. Si la cola de acciones está vacía -> pedir chunk (local o remoto) y guardar n_action_steps.
  3. Sacar una acción (goal_norm 8) -> ticks con range_min/range_max -> GoalSafety -> sync write.

`get_chunk(state, jpeg_by_view, task)` es una corrutina que devuelve (chunk (T,8), n_action_steps, info).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from station_io import (CameraMapper, FrameShapeError, GoalSafety, VIEWS, fetch_fresh_obs, goal_command, goal_norm_to_ticks,
                        new_station_client, safe_enable_torque, send_commands, set_torque)

TASK = ""   # ACT ignora la frase; las policies con lenguaje (SmolVLA) la necesitan: --task
FOLLOWER_SERIAL = "5B3D050113"


def add_robot_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("robot / Station")
    g.add_argument("--station", default="localhost", help="host[:puerto] del Station (TCP 8888 por defecto)")
    g.add_argument("--bus-serial", default=FOLLOWER_SERIAL, help="serial del bus ST3215 del FOLLOWER")
    g.add_argument("--motor-ids", default="1,2,3,4,5,6,7,8")
    g.add_argument("--task", default=TASK, help="instrucción de lenguaje (la misma del dataset)")
    g.add_argument("--camera-refs", default=None, help="banco de cámaras (por defecto inference/camera_refs.npz)")
    g.add_argument("--camera-order", default=None,
                   help="forzar el orden de images[] de Station, p.ej. 'table,side,gripper' (por defecto: autodetección)")
    g.add_argument("--enable-torque", action=argparse.BooleanOptionalAction, default=True,
                   help="activar torque del follower al arrancar (goal=posición actual antes). normvla no emite sin torque")
    g.add_argument("--torque-off-on-exit", action="store_true",
                   help="desactivar torque al salir (¡el brazo caerá por gravedad!). Por defecto se queda sujetando")
    g.add_argument("--dry-run", action="store_true",
                   help="predice e imprime pero NO envía goals (el brazo se queda quieto sujetando)")
    g.add_argument("--yes", action="store_true", help="no pedir confirmación antes de mover")
    g.add_argument("--max-ticks", type=int, default=0, help="parar tras N ticks (0 = sin límite)")
    g.add_argument("--timeout", type=float, default=5.0)
    s = p.add_argument_group("seguridad (ticks)")
    s.add_argument("--max-step-ticks", type=int, default=150, help="máx. cambio del goal por tick (brazo)")
    s.add_argument("--gripper-max-step-ticks", type=int, default=400, help="máx. cambio del goal por tick (gripper)")
    s.add_argument("--max-delta-ticks", type=int, default=600,
                   help="abortar tick si |goal-pos| de un joint del brazo supera esto (0 = desactivar)")
    s.add_argument("--max-aborts", type=int, default=5, help="parar tras N aborts consecutivos")


def _report_cameras(e: FrameShapeError, args) -> None:
    """Dice qué cámaras llegan y cuál falta, y guarda lo recibido para verlo."""
    from PIL import Image
    import io
    print(f"Station está publicando {len(e.jpegs)} cámara(s) y la policy necesita {e.exp_images} {VIEWS}.")
    print("Station pone una imagen por cámara activa, así que hay una cámara que Station no tiene abierta.")
    if e.jpegs:
        try:
            got = CameraMapper(fixed_order=None, **({"refs_path": args.camera_refs} if args.camera_refs else {})).identify(e.jpegs)
            for i, (v, sc) in enumerate(got):
                print(f"  images[{i}] se parece a: {v}  (puntuación {sc:.2f})")
            missing = [v for v in VIEWS if v not in {g[0] for g in got}]
            if len(missing) == e.exp_images - len(e.jpegs):
                print(f"  => falta la cámara: {', '.join(missing)}")
        except SystemExit:
            pass   # sin camera_refs.npz no podemos identificarlas; aun así guardamos la lámina
        sheet = Image.new("RGB", (224 * len(e.jpegs), 224))
        for i, j in enumerate(e.jpegs):
            sheet.paste(Image.open(io.BytesIO(j)).convert("RGB").resize((224, 224)), (224 * i, 0))
        out = Path("camera_received.jpg").resolve()
        sheet.save(out)
        print(f"  imágenes recibidas guardadas en {out}")
    print("Qué mirar: la web de Station (:8889) debe mostrar las 3 cámaras; cable/puerto USB de la que falta;\n"
          "  en Linux, la regla udev de las cámaras UVC (grupo plugdev); y reinicia Station tras reconectar.")


async def _confirm(prompt: str) -> bool:
    print(prompt, end="", flush=True)
    line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
    return line.strip().lower() in ("s", "si", "sí", "y", "yes")


async def run_control(args, get_chunk, log: logging.Logger) -> None:
    motor_ids = [int(x) for x in args.motor_ids.split(",")]
    n = len(motor_ids)
    client = await new_station_client(args.station, log)
    print(f"conectado a Station {args.station} | bus {args.bus_serial} | motores {motor_ids}")

    torque_enabled_here = False
    if args.enable_torque:
        print("activando torque del follower (goal = posición actual) ...")
        await safe_enable_torque(client, args.bus_serial, motor_ids)
        torque_enabled_here = True

    try:
        # ── 1. cámaras ──
        cams = CameraMapper(fixed_order=args.camera_order.split(",") if args.camera_order else None,
                            **({"refs_path": args.camera_refs} if args.camera_refs else {}))
        obs = await fetch_fresh_obs(client, b"", args.timeout, n_joints=n)
        while not cams.observe(obs.jpegs):
            obs = await fetch_fresh_obs(client, obs.frame_id, args.timeout, n_joints=n)
        preview = Path("camera_preview.jpg").resolve()
        cams.save_preview(obs.jpegs, preview)
        how = "forzado por --camera-order" if args.camera_order else f"autodetectado, margen {cams.margin:.2f}"
        print(f"cámaras: {cams.describe()}  ({how})\n  comprueba la lámina side|gripper|table: {preview}")

        if not args.dry_run and not args.yes:
            if not await _confirm("¿Empezar a mover el brazo? [s/N] "):
                print("cancelado.")
                return

        # ── 2. bucle ──
        safety = GoalSafety(n_joints=n, gripper_index=n - 1, max_step=args.max_step_ticks,
                            gripper_max_step=args.gripper_max_step_ticks, max_delta=args.max_delta_ticks)
        queue: deque = deque()
        tick, aborts, last_id = 0, 0, obs.frame_id
        mode = "DRY-RUN (no se envía nada)" if args.dry_run else "ENVIANDO goals"
        print(f"\n{mode}. Ctrl-C para parar.\n")
        while not args.max_ticks or tick < args.max_ticks:
            obs = await fetch_fresh_obs(client, last_id, args.timeout, n_joints=n)
            last_id = obs.frame_id
            tick += 1
            t0 = time.perf_counter()
            info = ""
            if not queue:
                chunk, n_steps, info = await get_chunk(obs.state, cams.by_view(obs.jpegs), args.task)
                queue.extend(np.asarray(chunk)[:n_steps])
            target = goal_norm_to_ticks(queue.popleft(), obs.ranges)
            goal, msg = safety.apply(target, obs.positions)
            ms = (time.perf_counter() - t0) * 1000
            if goal is None:
                aborts += 1
                print(f"[{tick:5d}] {msg}  ({aborts}/{args.max_aborts})")
                if aborts >= args.max_aborts:
                    print("demasiados aborts seguidos: paro.")
                    return
                continue
            aborts = 0
            if not args.dry_run:
                await send_commands(client, [goal_command(args.bus_serial, motor_ids, goal.tolist())])
            if tick % 10 == 1 or msg:
                d = goal - obs.positions
                print(f"[{tick:5d}] pos {obs.positions.tolist()}  goal Δ {d.tolist()}  "
                      f"cola {len(queue)}  {ms:.0f} ms {info} {msg}")
    except KeyboardInterrupt:
        print("\ninterrumpido.")
    except FrameShapeError as e:
        print(f"\nERROR: {e}")
        if len(e.jpegs) != e.exp_images:
            _report_cameras(e, args)
        raise SystemExit(1)
    finally:
        if torque_enabled_here and args.torque_off_on_exit:
            try:
                await set_torque(client, args.bus_serial, motor_ids, False)
                print("torque del follower DESACTIVADO.")
            except Exception:
                log.exception("no pude desactivar el torque")
        else:
            print("el follower mantiene el torque (sujeta la última posición). "
                  "Desactívalo desde la web de Station (:8889) o con --torque-off-on-exit.")
