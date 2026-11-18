"""Bucle de control común a run_local.py y robot_client.py.

Por cada frame nuevo de inference/normvla (~10 Hz, el mismo ritmo del dataset):
  1. obs = state (position_norm 8) + 3 JPEG; las cámaras se reordenan a (side, gripper, table).
  2. Si la cola de acciones está vacía -> pedir chunk (local o remoto) y guardar n_action_steps.
  3. Sacar una acción (goal_norm 8) -> ticks con range_min/range_max -> GoalSafety -> sync write.

`get_chunk(state, jpeg_by_view, task, prev)` es una corrutina que devuelve (chunk (T,8), n_action_steps, info);
`prev` = (state, jpeg_by_view) del frame anterior (None en el primero), para policies con n_obs_steps > 1.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from go_home import pidfile_path
from station_io import (describe_bus, CameraMapper, FrameShapeError, GoalSafety, VIEWS, fetch_fresh_obs, goal_command, goal_norm_to_ticks,
                        decode_rgb, new_station_client, safe_enable_torque, send_commands, set_torque)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))      # arm2/training/ -> hand_features
from hand_features import FINGERS, HAND_DIM, HAND_VIEW, HandFill, mark_jpeg, pick_tips, task_fingers  # noqa: E402

HAND_SOURCE_OVERRIDE = None   # selftest.py pone aquí una fuente que devuelve las columnas grabadas

TASK = ""   # SmolVLA la necesita; con yemas de los dedos decide dedo origen y destino (vacía: índice -> corazón)
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
    h = p.add_argument_group("yemas de los dedos (solo policies entrenadas con --hand)")
    h.add_argument("--hand-source", choices=["wilor", "mediapipe"], default="wilor",
                   help="la misma fuente que en el dataset: wilor (--hand dataset) o mediapipe (--hand mediapipe)")
    h.add_argument("--wilor-root", default=None, help="checkout de WiLoR (por defecto $WILOR_ROOT)")
    e = p.add_argument_group("temporal ensembling (solo inferencia, no cambia el entrenamiento)")
    e.add_argument("--replan-every", type=int, default=0,
                   help="pedir un chunk nuevo cada N ticks y promediar los chunks que se solapan (ACT temporal "
                        "ensembling). 0 = desactivado: se ejecutan n_action_steps y se vuelve a pedir")
    e.add_argument("--ensemble-coeff", type=float, default=0.01,
                   help="peso exp(-coeff * i) del chunk i (0 = el más antiguo); 0.01 es el valor del ACT original")
    s = p.add_argument_group("seguridad (ticks)")
    s.add_argument("--max-step-ticks", type=int, default=150, help="máx. cambio del goal por tick (brazo)")
    s.add_argument("--gripper-max-step-ticks", type=int, default=400, help="máx. cambio del goal por tick (gripper)")
    s.add_argument("--max-delta-ticks", type=int, default=600,
                   help="abortar tick si |goal-pos| de un joint del brazo supera esto (0 = desactivar)")
    s.add_argument("--max-aborts", type=int, default=5, help="parar tras N aborts consecutivos")
    s.add_argument("--gripper-current-limit", type=int, default=100,
                   help="si la corriente del gripper llega a esto, su goal se fija en la posición actual "
                        "(como el teleop; evita la protección por sobrecarga del servo). 0 = desactivar")
    s.add_argument("--stall-lead-limit", type=int, default=55,
                   help="si un motor lleva 0.8 s parado con el goal a más de esto (ticks) de su posición, el goal "
                        "se retiene: el servo salta por sobrecarga al 25 %% de carga durante 2 s (~58 ticks). "
                        "0 = desactivar")


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


def _hand_adder(args):
    """Devuelve f(state8, jpeg_by_view) -> state13, o None si la policy no usa las yemas de los dedos.

    Solo se llama al pedir un chunk (no en cada tick): WiLoR tarda ~60 ms por imagen.
    Bloquea (carga y ejecuta WiLoR en la GPU): llamarla siempre con asyncio.to_thread, nunca
    directamente en el bucle de asyncio. Si el bucle se queda bloqueado unos segundos (cargar
    WiLoR, o su primera llamada, que compila), el cliente de Station deja de atender la conexión
    y ya no recibe frames ni confirma comandos, aunque Station siga publicando.
    """
    hand_dim = getattr(args, "hand_dim", 0)
    if not hand_dim:
        return None
    if hand_dim != HAND_DIM:
        raise SystemExit(f"la policy espera un estado de {8 + hand_dim} valores; este código sabe dar 8 o {8 + HAND_DIM}")
    source = HAND_SOURCE_OVERRIDE
    if source is None:
        from hand_source import make_hand_source
        source = make_hand_source(args.hand_source, args.wilor_root)
        print(f"yemas de los dedos: {args.hand_source} sobre la cámara '{HAND_VIEW}'")
    fill, last = HandFill(), [None]
    name = {v: k for k, v in reversed(FINGERS.items())}
    src, dst = (name[i] for i in task_fingers(args.task))
    marks = getattr(args, "hand_marks", False)
    print(f"dedo origen: {src}, dedo destino: {dst} (de --task {args.task!r})"
          f"{'; yemas dibujadas en la imagen ' + HAND_VIEW if marks else ''}")

    def add(state, jpeg_by_view):
        t0 = time.perf_counter()
        tips = pick_tips(source(decode_rgb(jpeg_by_view[HAND_VIEW])), args.task)
        last[0] = (tips, (time.perf_counter() - t0) * 1000)
        vec = fill.step(tips)
        if marks and fill.seen:       # igual que convert_normacore.py --hand-marks
            jpeg_by_view[HAND_VIEW] = mark_jpeg(jpeg_by_view[HAND_VIEW], vec)
        return np.concatenate([np.asarray(state, np.float32), vec])

    def info():
        tips, ms = last[0]
        where = "SIN MANO" if tips is None else f"{src} ({tips[0]:.2f}, {tips[1]:.2f}) {dst} ({tips[2]:.2f}, {tips[3]:.2f})"
        return f" [mano {ms:.0f} ms: {where}]"

    add.info = info
    return add


async def run_control(args, get_chunk, log: logging.Logger) -> None:
    motor_ids = [int(x) for x in args.motor_ids.split(",")]
    n = len(motor_ids)
    client = await new_station_client(args.station, log)
    print(f"conectado a Station {args.station} | bus {args.bus_serial} | motores {motor_ids}")

    # SIGTERM = "para y SUJETA": otro proceso (stop_and_home.py) se queda con el brazo. No se
    # suelta el torque ni con --torque-off-on-exit. Ctrl-C sigue haciendo lo de siempre.
    handover = False
    main_task = asyncio.current_task()

    def _on_sigterm() -> None:
        nonlocal handover
        handover = True
        main_task.cancel()

    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, _on_sigterm)
    pidfile = pidfile_path(args.bus_serial)
    pidfile.write_text(str(os.getpid()))

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
                            gripper_max_step=args.gripper_max_step_ticks, max_delta=args.max_delta_ticks,
                            gripper_current=args.gripper_current_limit, lead=args.stall_lead_limit)
        queue: deque = deque()
        tick, aborts, last_id = 0, 0, obs.frame_id
        prev = None      # (state, jpeg_by_view) del tick anterior
        plans: list = []  # temporal ensembling: (tick en que se pidió, chunk entero)
        add_hand = await asyncio.to_thread(_hand_adder, args)
        if add_hand:        # la primera llamada a WiLoR compila (~5 s): fuera del bucle de control
            await asyncio.to_thread(add_hand, obs.state, cams.by_view(obs.jpegs))
            obs = await fetch_fresh_obs(client, last_id, args.timeout, n_joints=n)
            last_id = obs.frame_id
        mode = "DRY-RUN (no se envía nada)" if args.dry_run else "ENVIANDO goals"
        print(f"\n{mode}. Ctrl-C para parar.\n")
        while not args.max_ticks or tick < args.max_ticks:
            obs = await fetch_fresh_obs(client, last_id, args.timeout, n_joints=n)
            last_id = obs.frame_id
            tick += 1
            t0 = time.perf_counter()
            info = ""
            if (tick - 1) % args.replan_every == 0 if args.replan_every else not queue:
                jpegs = cams.by_view(obs.jpegs)
                if add_hand:        # 8 joints -> 13: las yemas del frame anterior primero, por el relleno causal
                    prev = (await asyncio.to_thread(add_hand, *prev), prev[1]) if prev is not None else None
                    state = await asyncio.to_thread(add_hand, obs.state, jpegs)
                    hand_info = add_hand.info()
                else:
                    state, hand_info = obs.state, ""
                chunk, n_steps, info = await get_chunk(state, jpegs, args.task, prev)
                info += hand_info
                if args.replan_every:
                    plans.append((tick, np.asarray(chunk)))
                else:
                    queue.extend(np.asarray(chunk)[:n_steps])
            prev = (obs.state, cams.by_view(obs.jpegs))
            if args.replan_every:      # media ponderada de lo que cada chunk vivo predijo para este tick
                plans = [(born, c) for born, c in plans if tick - born < len(c)]
                acts = np.stack([c[tick - born] for born, c in plans])
                w = np.exp(-args.ensemble_coeff * np.arange(len(acts)))
                action = (acts * w[:, None]).sum(0) / w.sum()
            else:
                action = queue.popleft()
            target = goal_norm_to_ticks(action, obs.ranges)
            goal, msg = safety.apply(target, obs.positions, obs.currents)
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
    except asyncio.CancelledError:
        if not handover:
            raise
        print("\nSIGTERM: paro y dejo el brazo sujetando para quien lo recoja.")
    except TimeoutError as e:        # Station ha dejado de publicar: se mira por qué ANTES de quitar el torque
        print(f"\nERROR: {e}")
        try:
            print(f"estado de los servos del bus {args.bus_serial} (un estado distinto de 0x00 para los frames):")
            print(await describe_bus(client, args.bus_serial))
            print("últimos ticks enviados (goal - posición, el más reciente al final):")
            print(safety.describe_history())
        except Exception as e2:
            print(f"  no se pudo leer el estado de los servos: {e2!r}")
        raise SystemExit(1)
    except FrameShapeError as e:
        print(f"\nERROR: {e}")
        if len(e.jpegs) != e.exp_images:
            _report_cameras(e, args)
        raise SystemExit(1)
    finally:
        pidfile.unlink(missing_ok=True)
        if torque_enabled_here and args.torque_off_on_exit and not handover:
            try:
                await set_torque(client, args.bus_serial, motor_ids, False)
                print("torque del follower DESACTIVADO.")
            except Exception:
                log.exception("no pude desactivar el torque")
        else:
            print("el follower mantiene el torque (sujeta la última posición). "
                  "Desactívalo desde la web de Station (:8889) o con --torque-off-on-exit.")
