#!/usr/bin/env python
"""Un solo proceso para la demo: lanza la policy, espera la tecla `d` (done) y vuelve a home.

    TASK="move device from right index to right middle finger"
    python -s run_skill.py --task "$TASK"

  1. Lanza run_local.py en segundo plano (con --yes y SIN --torque-off-on-exit: pase lo que
     pase, la policy sale sujetando el brazo).
  2. Espera a que pulses `d`: "ya ha terminado bien" o "nos rendimos", da igual.
  3. Para la policy (SIGTERM, y SIGKILL si en 3 s no ha salido) y ejecuta go_home.py: abrir
     gripper, brazo a home, cerrar gripper, soltar torque.

Teclas: `d` = parar y volver a home. Ctrl-C = parar y quedarse sujetando donde esté, sin ir a
home (para cuando el camino a home no está libre).

Si la policy se muere sola (Station sin frames, falta una cámara...) también se va a home, pero
el código de salida es 1.

Sin terminal (lanzado por otro programa): vale una línea `d` por stdin, o un SIGTERM a este
proceso.

Código de salida: 0 = policy parada con `d` y brazo en home sin torque; 1 = la policy falló o
no se llegó a home (el brazo sigue con torque); 130 = Ctrl-C, el brazo sigue con torque.

Las opciones que este script no conoce se pasan tal cual a run_local.py (p. ej. --device cpu).
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import select
import signal
import subprocess
import sys
from pathlib import Path

from go_home import add_home_args, load_home, run_go_home
from stop_and_home import stop

HERE = Path(__file__).resolve().parent
DONE_KEY = "d"


def default_action_steps(task: str) -> int:
    """Los valores que mejor han ido en el brazo (tabla "Modelo elegido" del README)."""
    return 9 if "ring to right pinky" in task.lower() else 15


def policy_command(args, extra: list[str], bus: str) -> list[str]:
    n = args.n_action_steps or default_action_steps(args.task)
    return [sys.executable, "-s", str(HERE / "run_local.py"), "--checkpoint", os.path.expanduser(args.checkpoint),
            "--task", args.task, "--bus-serial", bus, "--station", args.station,
            "--n-action-steps", str(n), "--yes", *extra]


def wait_for_done(proc, stdin, is_tty: bool, stopped: list) -> str:
    """Bloquea hasta que pase algo. Devuelve "done" (tecla d o SIGTERM) o "died" (la policy salió sola)."""
    watch_stdin = True
    while True:
        if stopped:
            return "done"
        if proc.poll() is not None:
            return "died"
        ready = select.select([stdin], [], [], 0.1)[0] if watch_stdin else []
        if not ready:
            if not watch_stdin:
                select.select([], [], [], 0.1)
            continue
        text = os.read(stdin.fileno(), 64).decode(errors="ignore")
        if not text and not is_tty:
            watch_stdin = False       # stdin cerrado: solo queda esperar a la policy o a un SIGTERM
        elif DONE_KEY in text.lower():
            return "done"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="la frase de la tarea (tabla del README)")
    p.add_argument("--checkpoint", default="~/ckpt/multi_marks")
    p.add_argument("--n-action-steps", type=int, default=None,
                   help="por defecto, el de la tabla del README para esa tarea (15, o 9 de anular a meñique)")
    add_home_args(p)
    args, extra = p.parse_known_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if not os.environ.get("NORMA_CORE") and importlib.util.find_spec("station_py") is None:
        raise SystemExit("falta norma-core en el entorno:  export NORMA_CORE=/ruta/a/norma-core  (y WILOR_ROOT)")
    bus = args.bus_serial or load_home(args.home_pose)[0]
    args.check = False

    stopped: list = []
    signal.signal(signal.SIGTERM, lambda *_: stopped.append(1))
    cmd = policy_command(args, extra, bus)
    print(f"lanzando la policy: {' '.join(cmd)}\n"
          f">>> pulsa '{DONE_KEY}' para parar y volver a home | Ctrl-C para parar y quedarse sujetando <<<\n", flush=True)
    # Sesión propia: el Ctrl-C del terminal llega solo a este proceso, que decide qué hacer con la policy.
    proc = subprocess.Popen(cmd, cwd=HERE, stdin=subprocess.DEVNULL, start_new_session=True)

    is_tty = sys.stdin.isatty()
    saved = None
    if is_tty:
        import termios
        import tty
        saved = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())     # teclas sueltas, sin Enter; Ctrl-C sigue siendo SIGINT
    try:
        why = wait_for_done(proc, sys.stdin, is_tty, stopped)
    except KeyboardInterrupt:
        print("\nCtrl-C: paro la policy; el brazo se queda CON torque donde está (no voy a home).")
        stop([proc.pid])
        proc.wait()
        return 130
    finally:
        if saved is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, saved)

    if why == "died":
        print(f"\nla policy ha terminado SOLA (código {proc.returncode}): mira su salida. Vuelvo a home.")
    else:
        print(f"\n'{DONE_KEY}': paro la policy y vuelvo a home.")
        stop([proc.pid])
        if proc.poll() is None and not _reaped(proc):
            print("ERROR: la policy sigue viva; no muevo el brazo con dos procesos mandando goals.")
            return 1
    home = run_go_home(args)
    return 1 if why == "died" else home


def _reaped(proc) -> bool:
    try:
        proc.wait(timeout=2)
        return True
    except subprocess.TimeoutExpired:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
