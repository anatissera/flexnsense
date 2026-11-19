"""The snapshot stream: fan-out to views that can read and cannot command.

Two groups carry the weight. "a reader cannot stall the session" is the safety one: this is
called from the control loop, so a dashboard that falls behind must never be able to hold up
the loop that is moving a finger. "null stays null" is the clinical one: a stale force shown
as a live one is what the whole safety layer exists to refuse.
"""
import asyncio
import json

import pytest

from arm1.lib.units import grams_force_to_newtons
from arm1.session.modes import Phase
from arm1.session.protocol import Block
from arm1.session.snapshot import SCHEMA_VERSION, SessionSnapshot
from arm1.session.snapshot_server import PATH, SnapshotHub, app_for


def block() -> Block:
    return Block.from_grams("index", "active", reps=3, cap_gf=150, hold_s=2.0)


def snap(force_gf: float | None = 120.0, phase: Phase = Phase.WORK,
         travel_m: float = -0.004, stopped: str | None = None) -> SessionSnapshot:
    return SessionSnapshot(
        block=block(), rep_index=1, rep_total=3, phase=phase, block_index=0, block_total=1,
        force_n=None if force_gf is None else grams_force_to_newtons(force_gf),
        travel_m=travel_m, stopped_reason=stopped,
    )


# ------------------------------------------------------------------ the hub, without a socket
def test_a_reader_gets_what_is_published():
    async def run():
        hub = SnapshotHub()
        with hub.reader() as queue:
            hub.publish(snap())
            got = await asyncio.wait_for(queue.get(), 1.0)
        assert got["phase"] == "work"
        assert got["forceGf"] == pytest.approx(120, abs=0.5)

    asyncio.run(run())


def test_a_reader_joining_mid_session_gets_the_latest_immediately():
    """A screen opened halfway through a session must not be blank until the next control
    cycle, which at a slow phase could be a visible gap."""
    async def run():
        hub = SnapshotHub()
        hub.publish(snap(phase=Phase.HOLD))
        with hub.reader() as queue:
            got = await asyncio.wait_for(queue.get(), 1.0)
        assert got["phase"] == "hold"

    asyncio.run(run())


def test_the_first_reader_of_a_session_that_has_not_started_waits():
    """Nothing published yet means nothing to show, not an invented frame."""
    async def run():
        hub = SnapshotHub()
        with hub.reader() as queue:
            assert queue.empty()

    asyncio.run(run())


def test_every_reader_gets_every_frame():
    """A clinician's screen, a patient's screen and a log window watch the same session."""
    async def run():
        hub = SnapshotHub()
        with hub.reader() as a, hub.reader() as b, hub.reader() as c:
            hub.publish(snap())
            for queue in (a, b, c):
                assert (await asyncio.wait_for(queue.get(), 1.0))["phase"] == "work"

    asyncio.run(run())


def test_a_reader_is_forgotten_when_it_leaves():
    async def run():
        hub = SnapshotHub()
        with hub.reader():
            assert hub.readers == 1
        assert hub.readers == 0

    asyncio.run(run())


# --------------------------------------------- a reader must not be able to stall the session
def test_publish_is_not_a_coroutine():
    """It is called from the loop that is commanding an arm. If it could await, a slow reader
    would become a pause in the control cycle."""
    assert not asyncio.iscoroutinefunction(SnapshotHub.publish)


def test_a_reader_that_stops_reading_is_dropped_rather_than_waited_for():
    async def run():
        hub = SnapshotHub(backlog=3)
        with hub.reader() as queue:
            for _ in range(20):
                hub.publish(snap())
            assert queue.qsize() == 3
            assert hub.dropped == 17

    asyncio.run(run())


def test_dropping_one_reader_does_not_stop_the_others():
    """The failure this prevents: one wedged browser tab freezing every other screen."""
    async def run():
        hub = SnapshotHub(backlog=2)
        with hub.reader() as wedged, hub.reader() as healthy:
            for _ in range(5):
                hub.publish(snap())
                # A healthy reader keeps up.
                healthy.get_nowait()
            assert wedged.full()
            hub.publish(snap(phase=Phase.HOLD))
            assert (await asyncio.wait_for(healthy.get(), 1.0))["phase"] == "hold"

    asyncio.run(run())


def test_publishing_with_nobody_watching_is_fine():
    """A session runs whether or not anyone has the page open."""
    hub = SnapshotHub()
    hub.publish(snap())
    assert hub.latest is not None


# ------------------------------------------------------------------------ null stays null
def test_a_missing_force_reading_is_sent_as_null_not_zero():
    """Zero means "no tension measured". Null means "we cannot see the force". Collapsing one
    into the other is the mistake the gate and the deadman both exist to prevent."""
    hub = SnapshotHub()
    hub.publish(snap(force_gf=None))
    assert hub.latest["forceN"] is None
    assert hub.latest["forceGf"] is None


def test_a_blind_frame_does_not_hold_the_last_reading_forward():
    """The hub republishes what it is given. A dashboard showing a plausible force for a
    session that has gone blind is exactly what must not happen."""
    hub = SnapshotHub()
    hub.publish(snap(force_gf=120.0))
    hub.publish(snap(force_gf=None))
    assert hub.latest["forceN"] is None


# ------------------------------------------------------------------------- over a real socket
def test_the_stream_delivers_a_snapshot_over_a_websocket():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    hub = SnapshotHub()
    hub.publish(snap())
    with TestClient(app_for(hub)).websocket_connect(PATH) as ws:
        got = json.loads(ws.receive_text())
    assert got["schema"] == SCHEMA_VERSION
    assert got["block"]["finger"] == "index"
    assert got["instruction"]["text"]


def test_health_reports_the_schema_and_whether_a_session_is_live():
    """So a dashboard can tell "no session running" from "version mismatch" before it renders
    anything, rather than from a frame that may never arrive."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    hub = SnapshotHub()
    client = TestClient(app_for(hub))
    assert client.get("/health").json() == {
        "ok": True, "source": "session", "schema": SCHEMA_VERSION, "readers": 0, "live": False,
    }
    hub.publish(snap())
    assert client.get("/health").json()["live"] is True


def test_many_screens_may_watch_at_once():
    """The opposite of session/link.py, which refuses a second connection because two brains
    driving one arm is two sessions disagreeing about custody. A view changes nothing, so any
    number of them is safe."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    hub = SnapshotHub()
    hub.publish(snap())
    client = TestClient(app_for(hub))
    with client.websocket_connect(PATH) as a, client.websocket_connect(PATH) as b:
        assert json.loads(a.receive_text())["phase"] == "work"
        assert json.loads(b.receive_text())["phase"] == "work"


def test_the_route_never_reads_from_the_socket():
    """One-way by construction, not by convention: a view that could also command would be a
    second place where "who may move the arm" is decided."""
    import inspect

    from arm1.session import snapshot_server

    source = inspect.getsource(snapshot_server.app_for)
    for forbidden in ("receive_text", "receive_json", "receive_bytes", "iter_text"):
        assert forbidden not in source, f"the snapshot route must never {forbidden}"


def test_the_module_has_no_future_annotations_import():
    """fastapi is imported inside a function here, so a stringified `ws: WebSocket` cannot be
    resolved and every handshake is rejected. gauge_reader/server.py documents the same trap,
    and this file hit it on the first run."""
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "arm1" / "session" / "snapshot_server.py"
    # The import itself, at the start of a line, rather than the docstring that explains why
    # it is absent.
    lines = source.read_text().splitlines()
    assert not [ln for ln in lines if ln.startswith("from __future__ import annotations")]
