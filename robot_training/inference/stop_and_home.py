#!/usr/bin/env python
"""Para la policy que controla el follower SIN que el brazo caiga y lo lleva a home.

    python -s stop_and_home.py --bus-serial 5B3D050113

Es lo que ejecuta el botón del orquestador cuando el usuario ve que la tarea ha terminado, o
que es mejor rendirse. Hace dos cosas, en este orden:

  1. Para run_local.py con SIGTERM, que le hace salir SUJETANDO el brazo aunque se lanzara con
     --torque-off-on-exit (control.py). Si en 3 s no ha salido, SIGKILL: tampoco suelta el
     torque, el servo se queda en su último goal.
  2. go_home.py: abrir gripper, brazo a home, cerrar gripper, soltar torque.

Si no hay ninguna policy corriendo va directo al paso 2.

Código de salida: 0 = en home y sin torque; 1 = no llegó, el brazo sigue sujetando.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from pathlib import Path

from go_home import add_home_args, load_home, pidfile_path, run_go_home

SCRIPTS = ("run_local.py", "robot_client.py")     # los dos que usan el bucle de control.py


def _cmdline(pid: int) -> list[str]:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")
    except OSError:
        return []


def _is_policy(pid: int) -> bool:
    return any(Path(a).name in SCRIPTS for a in _cmdline(pid))


def _alive(pid: int) -> bool:
    """Un zombi (terminado, sin recoger por quien lo lanzó) ya no controla nada: cuenta como muerto."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[-1].split()[0] != "Z"


def find_policy_pids(bus_serial: str) -> list[int]:
    """PID de la policy de ese bus: el del pidfile si sigue siendo ella; si no, se busca en /proc."""
    try:
        pid = int(pidfile_path(bus_serial).read_text())
        if _alive(pid) and _is_policy(pid):
            return [pid]
    except (OSError, ValueError):
        pass
    me = os.getpid()
    return [int(d.name) for d in Path("/proc").iterdir()
            if d.name.isdigit() and int(d.name) != me and _is_policy(int(d.name)) and _alive(int(d.name))]


def stop(pids: list[int], wait_s: float = 3.0) -> None:
    for sig, name in ((signal.SIGTERM, "SIGTERM"), (signal.SIGKILL, "SIGKILL")):
        pids = [p for p in pids if _alive(p)]
        if not pids:
            return
        for p in pids:
            print(f"{name} a la policy (pid {p})")
            try:
                os.kill(p, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and any(_alive(p) for p in pids):
            time.sleep(0.05)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_home_args(p)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    bus = args.bus_serial or load_home(args.home_pose)[0]
    pids = find_policy_pids(bus)
    if not pids:
        print("no hay ninguna policy corriendo.")
    elif args.check:
        print(f"policy corriendo (pid {pids}); con --check no se para.")
    else:
        stop(pids)
        if any(_alive(x) for x in pids):
            print("ERROR: la policy sigue viva; no muevo el brazo con dos procesos mandando goals.")
            return 1
    return run_go_home(args)


if __name__ == "__main__":
    raise SystemExit(main())
