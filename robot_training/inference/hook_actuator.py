#!/usr/bin/env python
"""El orquestador local de este ordenador: el extremo "actuador" de la sesión en dos máquinas.

    python -s hook_actuator.py serve                 # en ESTE ordenador (el del brazo 2)
    python -s hook_actuator.py request middle        # hace de cerebro, para probar sin el otro ordenador

El otro ordenador (el cerebro, docs/two-machine-session.md) manda por WebSocket un
`hook_request` con un dedo. Este proceso lo convierte en la tarea de la policy, lanza
run_skill.py y contesta con un `hook_report` SOLO cuando todo ha terminado:

    hook_request "middle"  ->  run_skill.py --task "move device from right index to right middle finger"
    alguien mira el brazo y pulsa, EN ESTE TECLADO:
        d = la anilla está en el dedo  ->  para la policy, brazo a home, torque fuera
                                       ->  hook_report "confirmed"
        f = no ha salido, nos rendimos ->  para la policy, brazo a home, torque fuera
                                       ->  hook_report "failed"

Un "confirmed" significa por tanto: la anilla está en el dedo pedido (lo ha visto una persona),
la policy está parada, y el brazo está en home sin torque. Cualquier otra cosa es "failed" o
silencio, nunca "confirmed" (regla 1 del diseño).

El protocolo es el de session/wire.py, sin tocar: este fichero solo sustituye la función
`on_link` de session/link.py, que es el sitio que ese módulo deja para una policy.

El mensaje lleva solo el dedo de DESTINO. El de origen se deduce: la policy sabe mover la
anilla al dedo de al lado (índice a corazón, corazón a anular, anular a meñique). Para un dedo
sin policy (el índice, donde la anilla se pone a mano al empezar) no se mueve nada: se pide a
la persona que la coloque y confirme con las mismas teclas.

Parar: si llega un `abort`, o si el cerebro deja de renovar el lease (0.5 s sin heartbeat, p. ej.
se cae la red), la policy se para y el brazo se queda CON torque donde esté. No va a home solo
ni contesta nada: el cerebro ya da el movimiento por fallido. Después, `python -s go_home.py`.
"""

import argparse
import asyncio
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))      # raíz del repo: session/

from session.arm2 import HookResult  # noqa: E402
from session.remote_arm2 import ActuatorSide, RemoteHookMover  # noqa: E402

from run_skill import DONE_KEY, EXIT_GAVE_UP, FAIL_KEY  # noqa: E402

#: Dedo de destino (nombres de session/fingers.py) -> frase de la tarea con la que se entrenó.
TASKS = {
    "middle": "move device from right index to right middle finger",
    "ring": "move device from right middle to right ring finger",
    "little": "move device from right ring to right pinky finger",
}


class LocalClockLink:
    """Pone la hora de ESTE ordenador en cada lease que llega.

    `issued_at` viene en el reloj monótono del cerebro y `ActuatorSide` lo compara con el de
    aquí. Son dos máquinas: los dos relojes no tienen nada que ver (cada uno cuenta desde su
    arranque), así que sin esto un lease estaría siempre caducado, o no caducaría nunca. Lo que
    el diseño pide (session/wire.py, `Lease`) es "han pasado más de `period_s` desde la última
    renovación que he oído", y eso es exactamente lo que queda al sellar la llegada con el reloj
    local. El mensaje en el cable no cambia.
    """

    def __init__(self, link, clock):
        self.link, self.clock = link, clock

    def send(self, message: dict) -> None:
        self.link.send(message)

    def drain(self) -> list:
        return [{**m, "lease": {**m["lease"], "issued_at": self.clock()}}
                if isinstance(m, dict) and isinstance(m.get("lease"), dict) else m
                for m in self.link.drain()]


class SkillJob:
    """run_skill.py como proceso hijo: la policy, la parada y la vuelta a home."""

    def __init__(self, finger: str, extra: list[str]):
        self.proc = subprocess.Popen(
            [sys.executable, "-s", str(HERE / "run_skill.py"), "--task", TASKS[finger], *extra],
            cwd=HERE, stdin=subprocess.PIPE, text=True)

    def key(self, key: str) -> None:
        try:
            self.proc.stdin.write(key + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass        # ya ha terminado: poll() lo dirá

    def poll(self) -> int | None:
        return self.proc.poll()

    def stop(self) -> None:
        """Como Ctrl-C en run_skill.py: para la policy y el brazo se queda sujetando."""
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class ManualJob:
    """Un dedo sin policy: una persona coloca la anilla y lo confirma. No se mueve ningún brazo."""

    def __init__(self, finger: str, extra: list[str] | None = None):
        self.code: int | None = None

    def key(self, key: str) -> None:
        self.code = 0 if key == DONE_KEY else EXIT_GAVE_UP

    def poll(self) -> int | None:
        return self.code

    def stop(self) -> None:
        pass


def start_job(finger: str, extra: list[str]):
    return SkillJob(finger, extra) if finger in TASKS else ManualJob(finger)


def outcome(finger: str, code: int, manual: bool) -> tuple[HookResult, str]:
    """El código de salida del trabajo como resultado para el cerebro. Solo el 0 confirma."""
    if code == 0:
        return HookResult.CONFIRMED, ("placed by hand and confirmed by the operator" if manual else
                                      "confirmed by the operator; policy stopped, arm 2 at home with torque off")
    if code == EXIT_GAVE_UP:
        return HookResult.FAILED, ("the operator could not place the device" if manual else
                                   "the operator gave up on the move; policy stopped, arm 2 at home with torque off")
    return HookResult.FAILED, (f"run_skill.py exited with code {code}: the policy died or arm 2 did not reach "
                               "home; arm 2 may still be holding torque away from home")


class HookActuator:
    """Lo que decide este extremo, un paso cada vez. Sin sockets ni reloj propios: se prueba en memoria."""

    def __init__(self, side: ActuatorSide, start=start_job, extra: list[str] | None = None, log=print):
        self.side, self.start, self.extra, self.log = side, start, extra or [], log
        self.job = None
        self.finger: str | None = None

    @property
    def busy(self) -> bool:
        return self.job is not None

    def step(self, now: float, keys: str = "") -> None:
        decision = self.side.step(now)
        if self.job is not None and not decision.may_move:
            # abort, o el lease ha caducado: parar. No hay nada que contestar, el cerebro ya lo da por fallido.
            self.log(f"!! PARADA: {decision.reason}. La policy se para y el brazo se queda con torque donde "
                     "esté. Cuando el camino esté libre:  python -s go_home.py")
            self.job.stop()
            self.job = None
            return
        if self.job is not None and decision.move_to != self.finger:
            self.log(f"!! petición nueva ({decision.move_to}) a mitad de mover a {self.finger}: paro y la doy por fallida.")
            self.job.stop()
            self.job = None
            self.side.report(HookResult.FAILED, f"a new request arrived while moving to {self.finger}; "
                                                "the move was stopped and arm 2 is holding torque where it was")
            return
        if self.job is None and decision.may_move:
            self.finger = decision.move_to
            manual = self.finger not in TASKS
            self.log(f"\n>> hook_request: anilla al dedo '{self.finger}'. "
                     + (f"NO hay policy para ese dedo: colócala a mano.  {DONE_KEY} = colocada | {FAIL_KEY} = no se puede"
                        if manual else
                        f"Lanzo la policy.  {DONE_KEY} = hecho, a home | {FAIL_KEY} = fallo, a home"))
            self.job = self.start(self.finger, self.extra)
        if self.job is None:
            return
        for key in keys.lower():
            if key in (DONE_KEY, FAIL_KEY):
                self.job.key(key)
                break
        code = self.job.poll()
        if code is not None:
            result, detail = outcome(self.finger, code, isinstance(self.job, ManualJob))
            self.job = None
            self.side.report(result, detail)
            self.log(f"<< hook_report: {result.value} ({self.finger}). {detail}")


# ─────────────────────────────── teclado y servidor ───────────────────────────────

def read_keys(flush: bool = False) -> str:
    """Las teclas pulsadas desde la última llamada, sin bloquear. Vacío si stdin no es un terminal."""
    if not sys.stdin.isatty():
        return ""
    if flush:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)     # una tecla de antes no confirma un movimiento nuevo
        return ""
    if select.select([sys.stdin], [], [], 0)[0]:
        return os.read(sys.stdin.fileno(), 64).decode(errors="ignore")
    return ""


def serve(args, extra: list[str]) -> int:
    import uvicorn

    from session.link import PATH, app_for

    newest = [0]      # una conexión nueva jubila el bucle de la anterior

    async def on_link(link) -> None:
        newest[0] += 1
        me = newest[0]
        loop = asyncio.get_running_loop()
        actuator = HookActuator(ActuatorSide(LocalClockLink(link, loop.time)), extra=extra)
        print("cerebro conectado.")
        try:
            while me == newest[0] or actuator.busy:
                was_busy = actuator.busy
                actuator.step(loop.time(), read_keys() if was_busy else "")
                if actuator.busy and not was_busy:
                    read_keys(flush=True)
                await asyncio.sleep(0.05)
        finally:
            if actuator.busy:
                actuator.job.stop()

    print(f"actuador escuchando en ws://{args.host}:{args.port}{PATH}  (salud: http://{args.host}:{args.port}/health)")
    print(f"dedos con policy: {', '.join(TASKS)}; el resto se colocan a mano. Ctrl-C para salir.")
    saved = None
    if sys.stdin.isatty():
        import termios
        import tty
        saved = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    try:
        uvicorn.run(app_for(on_link), host=args.host, port=args.port, log_level="warning")
    finally:
        if saved is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, saved)
    return 0


async def request(args) -> int:
    """Hace de cerebro para UN movimiento, con el mismo RemoteHookMover que usará el de verdad."""
    from session.link import connect_to_actuator

    link, task = await connect_to_actuator(args.url)
    mover = RemoteHookMover(link, move_timeout_s=args.timeout)
    mover.request(args.finger)
    print(f"hook_request '{args.finger}' enviado a {args.url}; esperando el hook_report ...")
    try:
        while True:
            got = mover.poll(time.monotonic())
            if got is not None:
                print(f"hook_report: {got.result.value} ({got.finger}). {got.detail}")
                return 0 if got.confirmed else 1
            if task.done():
                print(f"la conexión se ha cerrado sin respuesta: {task.exception()!r}")
                return 1
            await asyncio.sleep(0.1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        mover.abort()
        await asyncio.sleep(0.3)      # que el abort salga antes de cerrar
        print("\nabort enviado.")
        return 130
    finally:
        task.cancel()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="este ordenador: recibe hook_request y ejecuta run_skill.py. "
                                     "Las opciones que no conoce se pasan a run_skill.py (p. ej. --dry-run)")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8770)
    r = sub.add_parser("request", help="hace de cerebro: pide un dedo y espera el resultado")
    r.add_argument("finger", help="index, middle, ring, little (o pinky)")
    r.add_argument("--url", default="ws://localhost:8770/arm2")
    r.add_argument("--timeout", type=float, default=300.0, help="segundos sin respuesta antes de darlo por fallido")
    args, extra = p.parse_known_args(argv)
    if args.cmd == "serve":
        return serve(args, extra)
    if extra:
        p.error(f"opciones desconocidas: {extra}")
    try:
        return asyncio.run(request(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
