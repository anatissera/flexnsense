"""Serve `SessionSnapshot` to the dashboard, so it renders the arm instead of a timeline.

    .venv/bin/python -m session.snapshot_server          # on the machine running the session

The dashboard currently runs its own clock: `GuideHand.tsx` counts repetitions off
`performance.now()` against hardcoded constants, so what the patient hears and what the arm is
doing are two independent stories that happen to start together. This is the route that
replaces the first with the second.

## One writer, many readers

The opposite of `session/link.py`, deliberately. That one refuses a second connection, because
two brains driving one arm is two sessions disagreeing about custody. This one accepts any
number, because a snapshot is a **view**: a clinician's screen, a patient's screen and a log
window can all watch the same session without any of them being able to change it.

Nothing a reader sends is acted on. The socket is one-way by construction, not by convention:
the handler never reads from it. A dashboard that wants to *stop* a session does not do it
here (see `docs/two-machine-session.md`); it is a view, and a view that could also command
would be a second place where "who may move the arm" is decided.

## Null is not zero

`SessionSnapshot.to_dict` sends `forceN: null` when the camera could not see the marker, and
this route never fills that in. Holding the last reading forward would make a dashboard show a
plausible force for a session that has gone blind, which is the one thing the safety layer
exists to refuse. The dashboard's own type says the same thing (`sessionSnapshot.ts`).

## The publisher

`SnapshotHub.publish(snapshot)` is called by whatever is running the session, once per control
cycle. It never blocks and never fails: a slow or dead reader is dropped, because a view
falling behind must not be able to stall the loop that is moving a finger. That is the same
rule `session.link.QueueLink` follows and for the same reason.

Note there is deliberately no `from __future__ import annotations` here, the same as in
`gauge_reader/server.py` and `hand_tracking/server.py`: fastapi is imported inside a function,
so a stringified `ws: WebSocket` could not be resolved and every handshake would be rejected.
"""
import asyncio
import contextlib
import json
from typing import Any

from session.snapshot import SCHEMA_VERSION, SessionSnapshot

__all__ = ["SnapshotHub", "app_for", "PATH", "DEFAULT_PORT"]

PATH = "/session"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8771

#: Frames a reader may fall behind by before it is dropped. Small on purpose: a snapshot is
#: only useful while it is current, so a reader that cannot keep up is better off reconnecting
#: and getting the latest than being handed a backlog of stale ones.
READER_BACKLOG = 8


class SnapshotHub:
    """Fan-out of the latest snapshot to every connected reader.

    Pure asyncio, no FastAPI, so the publishing side is testable without a socket, and so that
    whatever runs the session can hold one of these without importing a web framework.
    """

    def __init__(self, backlog: int = READER_BACKLOG):
        self.backlog = backlog
        self._readers: set[asyncio.Queue] = set()
        #: The last snapshot published, sent to a reader the moment it connects so a screen
        #: opened mid-session is not blank until the next control cycle.
        self.latest: dict[str, Any] | None = None
        #: Frames dropped because a reader was too far behind. For the session log.
        self.dropped = 0

    def publish(self, snapshot: SessionSnapshot | dict[str, Any]) -> None:
        """Make `snapshot` the current one. Never blocks, never raises.

        Called from the control loop, so it must do neither: a view falling behind cannot be
        allowed to stall the loop that is moving a finger.
        """
        payload = snapshot.to_dict() if isinstance(snapshot, SessionSnapshot) else dict(snapshot)
        self.latest = payload
        for queue in list(self._readers):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                # Drop rather than wait. The reader will get the next one, which is more
                # useful than the one it missed.
                self.dropped += 1

    @contextlib.contextmanager
    def reader(self):
        """A queue fed every published snapshot, removed again on the way out."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.backlog)
        if self.latest is not None:
            queue.put_nowait(self.latest)
        self._readers.add(queue)
        try:
            yield queue
        finally:
            self._readers.discard(queue)

    @property
    def readers(self) -> int:
        return len(self._readers)


def app_for(hub: SnapshotHub):
    """FastAPI app serving `hub` to any number of readers."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.get("/health")
    def health() -> JSONResponse:
        """CORS header for the same reason the other servers carry one: in development the
        dashboard is served from a different port."""
        return JSONResponse(
            {"ok": True, "source": "session", "schema": SCHEMA_VERSION,
             "readers": hub.readers, "live": hub.latest is not None},
            headers={"Access-Control-Allow-Origin": "*"},
        )

    @app.websocket(PATH)
    async def session(ws: WebSocket) -> None:
        await ws.accept()
        with hub.reader() as queue:
            try:
                while True:
                    await ws.send_text(json.dumps(await queue.get()))
            except WebSocketDisconnect:
                pass

    return app


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--host", default=DEFAULT_HOST,
                   help="127.0.0.1 by default: a snapshot stream is for screens on this "
                        "machine or on a network someone deliberately opened it to")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--demo", action="store_true",
                   help="publish a simulated session so the route can be checked with no arm "
                        "and no camera. Every frame it sends is the real SessionSnapshot "
                        "shape, produced by the real RepMachine")
    args = p.parse_args(argv)

    import uvicorn

    hub = SnapshotHub()
    app = app_for(hub)

    if args.demo:
        @app.on_event("startup")
        async def _demo() -> None:
            asyncio.create_task(_run_demo(hub))

    print(f"session snapshots on ws://{args.host}:{args.port}{PATH}")
    if args.demo:
        print("--demo: a SIMULATED session. No arm is attached and no force is measured.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


async def _run_demo(hub: SnapshotHub) -> None:
    """Drive the real `RepMachine` against a simulated patient and publish every step.

    Exists so the route and the dashboard's subscription can be developed and demonstrated
    without the rig. It is the real rep machine and the real snapshot shape, so what a
    dashboard renders here is what it will render from a session; only the force and the
    travel are invented, and `snapshot.py` has no way to mark that, so the caller must.
    """
    import time

    from lib.units import grams_force_to_newtons
    from session.protocol import Block
    from session.reps import RepInput, RepLimits, RepMachine

    block = Block.from_grams("index", "active", reps=3, cap_gf=150, hold_s=2.0)
    machine = RepMachine(block, RepLimits())
    force = grams_force_to_newtons(20)
    travel = 0.0
    while True:
        now = time.monotonic()
        step = machine.step(RepInput(force_n=force, reading_age_s=0.01, travel_m=travel,
                                     now=now, confident=True))
        hub.publish(SessionSnapshot(
            block=block, rep_index=step.rep_index, rep_total=block.reps,
            phase=step.phase, block_index=0, block_total=1,
            force_n=force, travel_m=travel,
            stopped_reason=step.note if step.finished else None,
        ))
        # A simulated rig and patient: the arm takes up slack while ARMED, the patient flexes
        # to the cap, holds, and relaxes while the arm follows them back.
        phase = step.phase.value
        if phase == "armed":
            force += grams_force_to_newtons(4)        # the arm taking the slack out
        elif phase == "cueing":
            force += grams_force_to_newtons(3)        # the patient starting to pull
        elif phase in ("ramp", "work"):
            # Stops at the cap rather than pushing through it, which is what a patient on the
            # real rig does once the arm starts giving way: the force plateaus and the travel
            # stops advancing, which is how `_work` knows the stroke is over.
            if force < block.cap_n:
                force = min(block.cap_n, force + grams_force_to_newtons(6))
                travel -= 0.0012
        elif phase == "reset":
            force = max(grams_force_to_newtons(20), force - grams_force_to_newtons(8))
            travel = min(0.0, travel + 0.0015)
        if step.finished:
            machine = RepMachine(block, RepLimits())
            force, travel = grams_force_to_newtons(20), 0.0
        await asyncio.sleep(1 / 20)


if __name__ == "__main__":
    raise SystemExit(main())
