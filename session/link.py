"""The actual transport under `session.remote_arm2`: a WebSocket, and the queues either side.

Deliberately thin, and deliberately separate from the protocol. Everything that decides
anything lives in `session/remote_arm2.py` and `session/wire.py`, where it is tested against a
list in memory with a fake clock. What is here is plumbing: move dicts between a socket and a
queue, and do it without ever blocking the loop that is also driving an arm.

Keeping the split means a bug in this file cannot be mistaken for a bug in the lease, and the
interesting failures (a report that never comes, a lease that expired mid-move) stay testable
without starting a server.

Same WebSocket pattern as `gauge_reader/server.py` and `hand_tracking/server.py`, including
the absence of `from __future__ import annotations`: fastapi is imported inside a function, so
a stringified `ws: WebSocket` could not be resolved and every handshake would 403.

    # on arm 2's machine (the actuator)
    python -m session.link serve --host 0.0.0.0 --port 8770

    # on arm 1's machine (the brain), from code
    link = await connect_to_actuator("ws://justo.local:8770/arm2")

Who runs which is not arbitrary and is not a preference: docs/two-machine-session.md.
"""
import argparse
import asyncio
import contextlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8770
#: The one route. Named for what is on the far end of it rather than for what it carries, so
#: a second route later (the snapshot stream) does not have to fight this one for a good name.
PATH = "/arm2"


class QueueLink:
    """A `session.remote_arm2.Link` over two asyncio queues.

    `send` and `drain` are **not** coroutines, and that is the point: the code calling them is
    a pure state machine stepped once per cycle by a loop that is also talking to Station and a
    camera, and it must never be able to await. Messages queue; a separate task does the I/O.

    An unbounded outbox would be wrong here: if the far end stops reading, the queue grows
    until the process dies, which turns a network problem into a crash on the machine holding
    an arm. `maxsize` makes it drop instead, and dropping is safe by design, because the only
    message whose loss matters (a hook report) is one the brain times out on anyway, and a
    timeout resolves to FAILED rather than to a guess.
    """

    def __init__(self, maxsize: int = 256):
        self._outbox: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._inbox: list = []
        #: Messages dropped because the outbox was full. A nonzero value here means the link is
        #: not keeping up and belongs in the session log.
        self.dropped = 0

    # ---------------------------------------------------------------- the Link protocol
    def send(self, message: dict) -> None:
        try:
            self._outbox.put_nowait(message)
        except asyncio.QueueFull:
            self.dropped += 1

    def drain(self) -> list:
        got, self._inbox = self._inbox, []
        return got

    # ---------------------------------------------------------------- the I/O side
    def deliver(self, message: dict) -> None:
        """Called by the reader task when something arrives off the socket."""
        self._inbox.append(message)

    async def next_outgoing(self) -> dict:
        return await self._outbox.get()


async def pump(ws, link: QueueLink) -> None:
    """Run both directions of one connection until it closes.

    The two directions are separate tasks because they block on different things, and one
    waiting must not stop the other: a brain with nothing to send still has to hear a report,
    and an actuator mid-move still has to hear an abort.
    """
    async def outgoing() -> None:
        while True:
            await ws.send_json(await link.next_outgoing())

    async def incoming() -> None:
        while True:
            link.deliver(await ws.receive_json())

    tasks = [asyncio.create_task(outgoing()), asyncio.create_task(incoming())]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            # Surfaces a protocol error rather than letting the connection look like a clean
            # close, which would read as "the far end went away" and hide a real bug.
            task.result()
    finally:
        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


# ------------------------------------------------------------------------- the actuator server
def app_for(on_link):
    """FastAPI app serving the actuator end. `on_link(link)` runs the arm for one connection.

    One connection at a time is not a limitation to fix later, it is rule 4 of
    docs/two-machine-session.md: exactly one orchestrator exists. Two brains connected to one
    arm is two sessions disagreeing about who has custody, so the second is refused.
    """
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse

    app = FastAPI()
    state = {"busy": False}

    @app.get("/health")
    def health() -> JSONResponse:
        """CORS header for the same reason the other two servers carry one: in development the
        dashboard is served from a different port."""
        from session.wire import SCHEMA_VERSION

        return JSONResponse(
            {"ok": True, "source": "arm2", "schema": SCHEMA_VERSION, "busy": state["busy"]},
            headers={"Access-Control-Allow-Origin": "*"},
        )

    @app.websocket(PATH)
    async def arm2(ws: WebSocket) -> None:
        if state["busy"]:
            # Refused rather than queued: a brain that waits in line would believe it has an
            # actuator while another session is driving it.
            await ws.close(code=1013, reason="another session already holds this arm")
            return
        await ws.accept()
        state["busy"] = True
        link = QueueLink()
        try:
            await asyncio.gather(pump(ws, link), on_link(link))
        except WebSocketDisconnect:
            pass
        finally:
            state["busy"] = False

    return app


async def connect_to_actuator(url: str, link: QueueLink | None = None):
    """The brain's side: open the socket and return `(link, task)`.

    The caller keeps `task` alive for the session and cancels it at the end. The link works
    before the socket is even up, because `send` only queues, which is what lets the
    orchestrator be stepped from the first cycle without a connection check in its path.
    """
    import websockets

    link = QueueLink() if link is None else link
    ws = await websockets.connect(url)

    async def run() -> None:
        import json

        async def outgoing() -> None:
            while True:
                await ws.send(json.dumps(await link.next_outgoing()))

        async def incoming() -> None:
            async for raw in ws:
                link.deliver(json.loads(raw))

        tasks = [asyncio.create_task(outgoing()), asyncio.create_task(incoming())]
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done:
                task.result()
        finally:
            for task in tasks:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            await ws.close()

    return link, asyncio.create_task(run())


async def _ask_yes_no(prompt: str) -> bool:
    """Blocking input() off the event loop, so the server keeps answering heartbeats and
    health checks while a person decides."""
    answer = await asyncio.get_running_loop().run_in_executor(None, input, prompt)
    return answer.strip().lower() in ("y", "yes", "s", "si", "sí")


def _policy_on_link(checkpoint: str, task: str, bus_serial: str, inference_dir: str,
                    n_action_steps: int | None):
    """`on_link`, backed by the real `robot_training/inference` policy and `stop_and_home.py`.

    Not `PolicyHookMover`: the policy does not decide the outcome, a person watching the arm
    does, by answering a prompt here. `session/arm2.py`'s rule still holds - success is
    reported, never assumed - this only replaces who confirms it manually (`ManualHookMover`'s
    `confirm()` on this machine) with what the operator here actually has: the real policy
    process and `stop_and_home.py`'s real exit code, not an invented one.

    Untested against real hardware: wired and readable, not run on the arm.
    """
    import subprocess

    from session.arm2 import HookResult
    from session.remote_arm2 import ActuatorSide

    async def on_link(link: QueueLink) -> None:
        actuator = ActuatorSide(link)
        loop = asyncio.get_running_loop()
        last = None
        while True:
            decision = actuator.step(loop.time())
            if decision.halt:
                print(f"!! halt: {decision.reason}")
            if decision.move_to != last:
                last = decision.move_to
                if decision.move_to is None:
                    continue
                finger = decision.move_to
                print(f"\n>> launching the policy toward the {finger} finger "
                      f"(task={task!r}) ...")
                policy = subprocess.Popen(
                    ["python", "-s", "run_local.py", "--checkpoint", checkpoint,
                     "--task", task, "--bus-serial", bus_serial, "--yes",
                     "--torque-off-on-exit",
                     *(["--n-action-steps", str(n_action_steps)] if n_action_steps else [])],
                    cwd=inference_dir,
                )
                print("   Watch the arm. The policy does not know when it is done.")
                done = await _ask_yes_no(
                    f"   Hooked on the {finger} finger? Press Enter once it is, or type n to "
                    "give up [Y/n]: "
                )
                print("   stopping the policy and homing ...")
                stop = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["python", "-s", "stop_and_home.py", "--bus-serial", bus_serial],
                        cwd=inference_dir,
                    ).returncode,
                )
                policy.wait()
                if stop != 0:
                    # stop_and_home.py's own contract: nonzero means it did NOT reach home and
                    # the arm is still holding. Reported as failed whatever the person said,
                    # because a hook on the right finger with the arm then stuck mid-air is
                    # not a state the orchestrator should be told is fine.
                    actuator.report(HookResult.FAILED,
                                    "stop_and_home.py exited nonzero: did not reach home, the "
                                    "arm is still holding")
                elif done:
                    actuator.report(HookResult.CONFIRMED, "confirmed by the operator at arm 2")
                else:
                    actuator.report(HookResult.FAILED, "operator reported the hook did not seat")
            await asyncio.sleep(0.05)

    return on_link


async def _manual_on_link(link: QueueLink) -> None:
    """Ask a person to move the hook by hand, then report their confirmation over the wire."""
    from session.arm2 import HookResult
    from session.remote_arm2 import ActuatorSide

    actuator = ActuatorSide(link)
    loop = asyncio.get_running_loop()
    last = None
    while True:
        decision = actuator.step(loop.time())
        if decision.halt:
            print(f"!! halt: {decision.reason}")
        if decision.move_to != last:
            last = decision.move_to
            if decision.move_to:
                finger = decision.move_to
                print(f"\n>> move the hook to the {finger} finger by hand.")
                confirmed = await _ask_yes_no(
                    f"   Is the hook actually on the {finger} finger? Confirm only after "
                    "checking [Y/n]: "
                )
                if confirmed:
                    actuator.report(HookResult.CONFIRMED,
                                    "confirmed by the operator at arm 2")
                else:
                    actuator.report(HookResult.FAILED,
                                    "operator reported the hook did not seat")
        await asyncio.sleep(0.05)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="run the actuator end on arm 2's machine")
    serve.add_argument("--host", default=DEFAULT_HOST)
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--checkpoint", default=None,
                       help="run the chosen policy toward the requested finger instead of "
                            "waiting on a person to move the hook by hand. Still needs a "
                            "person to say when it is done (robot_training/README.md: the "
                            "policy does not know when the task ends); this only replaces who "
                            "moves the hook, not who confirms the outcome")
    serve.add_argument("--task", default="", help="with --checkpoint: the policy's --task")
    serve.add_argument("--bus-serial", default=None, help="with --checkpoint: the follower bus")
    serve.add_argument("--inference-dir", default="robot_training/inference",
                       help="with --checkpoint: cwd for run_local.py and stop_and_home.py, "
                            "which use flat imports and need it")
    serve.add_argument("--n-action-steps", type=int, default=None)
    args = p.parse_args(argv)

    import uvicorn

    if args.checkpoint:
        if not args.bus_serial:
            raise SystemExit("--checkpoint needs --bus-serial: it is passed straight to "
                             "run_local.py and stop_and_home.py")
        on_link = _policy_on_link(args.checkpoint, args.task, args.bus_serial,
                                  args.inference_dir, args.n_action_steps)
    else:
        on_link = _manual_on_link

    print(f"actuator listening on {args.host}:{args.port}{PATH}")
    print("This machine holds no session state and decides nothing: it executes and reports.")
    if args.checkpoint:
        print(f"--checkpoint set: a hook request launches the policy toward that finger. A "
              "person still confirms the outcome (see robot_training/README.md).")
    uvicorn.run(app_for(on_link), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
