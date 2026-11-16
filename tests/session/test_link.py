"""The transport: queueing, dropping, and the two directions not blocking each other.

Thin on purpose, because the transport is thin on purpose. Everything that decides anything is
tested in `test_remote_arm2.py` and `test_wire.py` against a list in memory. What is left here
is the handful of properties that only exist once a real queue and a real socket are involved,
and the one that matters is that `send` can never block the loop that is driving an arm.
"""
from __future__ import annotations

import asyncio

import pytest

from session.link import PATH, QueueLink, app_for


def test_send_is_not_a_coroutine():
    """The code calling it is a pure state machine stepped by a loop that is also talking to
    Station and a camera. If `send` could await, a slow link would stall the arm."""
    assert not asyncio.iscoroutinefunction(QueueLink.send)
    assert not asyncio.iscoroutinefunction(QueueLink.drain)


def test_a_message_queued_before_anything_is_connected_is_kept():
    """`send` only queues, which is what lets the orchestrator be stepped from the first cycle
    without a connection check anywhere in its path."""
    link = QueueLink()
    link.send({"kind": "hook_request"})
    assert link._outbox.qsize() == 1


def test_drain_returns_what_arrived_oldest_first_and_clears():
    link = QueueLink()
    link.deliver({"n": 1})
    link.deliver({"n": 2})
    assert link.drain() == [{"n": 1}, {"n": 2}]
    assert link.drain() == []


def test_a_full_outbox_drops_rather_than_growing_without_limit():
    """A far end that stops reading must not turn into a crash on the machine holding an arm.
    Dropping is safe by design: the only message whose loss matters is a hook report, and the
    brain times out on that anyway, which resolves to FAILED rather than to a guess."""
    link = QueueLink(maxsize=3)
    for i in range(10):
        link.send({"n": i})
    assert link._outbox.qsize() == 3
    assert link.dropped == 7


def test_dropping_is_counted_so_it_can_reach_the_session_log():
    link = QueueLink(maxsize=1)
    link.send({"n": 0})
    assert link.dropped == 0
    link.send({"n": 1})
    assert link.dropped == 1


def test_next_outgoing_waits_for_a_message():
    async def run():
        link = QueueLink()
        task = asyncio.create_task(link.next_outgoing())
        await asyncio.sleep(0)
        assert not task.done(), "it should still be waiting"
        link.send({"kind": "abort"})
        assert await asyncio.wait_for(task, timeout=1.0) == {"kind": "abort"}

    asyncio.run(run())


def test_outgoing_order_is_preserved():
    async def run():
        link = QueueLink()
        for i in range(5):
            link.send({"n": i})
        got = [await link.next_outgoing() for _ in range(5)]
        assert [m["n"] for m in got] == [0, 1, 2, 3, 4]

    asyncio.run(run())


# --------------------------------------------------------------------------- the server
def test_health_reports_the_schema_it_speaks():
    """So a version mismatch is visible before a session starts rather than as a refused
    message in the middle of one."""
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from session.wire import SCHEMA_VERSION

    async def on_link(link):
        await asyncio.sleep(3600)

    client = TestClient(app_for(on_link))
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["source"] == "arm2"
    assert body["schema"] == SCHEMA_VERSION


def test_a_second_brain_is_refused_rather_than_queued():
    """Rule 4 of docs/two-machine-session.md: exactly one orchestrator exists. A brain that
    waited in line would believe it has an actuator while another session drives it."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    started = asyncio.Event()

    async def on_link(link):
        started.set()
        await asyncio.sleep(3600)

    client = TestClient(app_for(on_link))
    with client.websocket_connect(PATH):
        with pytest.raises(Exception):
            with client.websocket_connect(PATH):
                pass


def test_the_route_is_named_for_what_is_on_the_far_end():
    assert PATH == "/arm2"


# ------------------------------------------------------- over an actual socket, once
def test_a_hook_move_round_trips_over_a_real_websocket():
    """The one case the fake link cannot cover: that the transport delivers what the protocol
    expects, both ways, through real JSON over a real socket.

    Everything *about* the move (the lease, the timeout, the late report) is tested against the
    fake link, where the clock is an argument. This exists only to catch the plumbing mistakes
    a fake queue cannot have: a message that does not survive serialisation, a direction that
    was never wired, a reader that blocks the writer.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from session.arm2 import HookResult
    from session.remote_arm2 import ActuatorSide
    from session.wire import MessageKind, decode, encode_hook_report

    seen: dict = {}

    async def on_link(link):
        actuator = ActuatorSide(link)
        loop = asyncio.get_running_loop()
        while True:
            decision = actuator.step(loop.time())
            if decision.move_to and "reported" not in seen:
                seen["moved_to"] = decision.move_to
                actuator.report(HookResult.CONFIRMED, "confirmed on the far machine")
                seen["reported"] = True
            await asyncio.sleep(0.01)

    client = TestClient(app_for(on_link))
    with client.websocket_connect(PATH) as ws:
        from session.wire import Lease, encode_hook_request

        ws.send_json(encode_hook_request("middle", Lease(move_id=1, issued_at=0.0,
                                                         period_s=1e6)))
        kind, fields = decode(ws.receive_json())

    assert kind is MessageKind.HOOK_REPORT
    assert fields["result"] is HookResult.CONFIRMED
    assert fields["finger"] == "middle"
    assert fields["move_id"] == 1
    assert seen["moved_to"] == "middle"


def test_the_far_end_refuses_a_request_it_cannot_trust_over_a_real_socket():
    """A wrong-version message must not reach the arm. It is logged and dropped, and crucially
    it does not authorise motion."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from session.remote_arm2 import ActuatorSide

    seen: dict = {"authorised": False}

    async def on_link(link):
        actuator = ActuatorSide(link)
        loop = asyncio.get_running_loop()
        for _ in range(30):
            if actuator.step(loop.time()).may_move:
                seen["authorised"] = True
            seen["rejected"] = list(actuator.rejected)
            await asyncio.sleep(0.01)

    client = TestClient(app_for(on_link))
    with client.websocket_connect(PATH) as ws:
        ws.send_json({"schema": 999, "kind": "hook_request", "finger": "index"})
        import time as _time
        _time.sleep(0.2)

    assert seen["authorised"] is False
    assert seen.get("rejected"), "the refusal should have been logged"
