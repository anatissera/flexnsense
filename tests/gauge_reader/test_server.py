"""The gauge server's routes, against a stub pipeline: no camera, no calibration file, no scale.

Same regression guard as tests/hand_tracking/test_server.py: with `from __future__ import annotations` in
the server module FastAPI would read `ws: WebSocket` as a query parameter and 403 every handshake.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="fastapi is not in requirements.txt")
cv2 = pytest.importorskip("cv2")

from fastapi.testclient import TestClient  # noqa: E402

from gauge_reader.server import app_for, reading_message  # noqa: E402
from lib.units import grams_force_to_newtons  # noqa: E402


class StubResult:
    def __init__(self, grams, found=True, stable=True, reason=""):
        self.grams = grams
        self.newtons = None if grams is None else grams_force_to_newtons(grams)
        self.found = found
        self.stable = stable
        self.reason = reason
        self.detection = SimpleNamespace(confidence=0.9)


class StubPipeline:
    def __init__(self, grams=200.0, found=True, tare_ok=True):
        self.grams, self.found, self.tare_ok = grams, found, tare_ok
        self.seen: list[tuple[int, int]] = []
        self.times: list[float] = []

    def process(self, frame, ts):
        self.seen.append((frame.shape[1], frame.shape[0]))
        self.times.append(ts)
        if not self.found:
            return StubResult(None, found=False, stable=False, reason="no-marker")
        return StubResult(self.grams)

    def tare(self):
        return 0.1 if self.tare_ok else None


def jpeg(width=640, height=480) -> bytes:
    return cv2.imencode(".jpg", np.zeros((height, width, 3), dtype=np.uint8))[1].tobytes()


def client_for(pipe) -> TestClient:
    return TestClient(app_for(lambda: pipe, describe="stub"))


def test_health_allows_the_dashboards_origin():
    r = client_for(StubPipeline()).get("/health")
    assert r.json() == {"ok": True, "source": "gauge", "calibration": "stub"}
    assert r.headers["access-control-allow-origin"] == "*"


def test_a_frame_in_gets_a_reading_in_newtons_back():
    pipe = StubPipeline(grams=200.0)
    with client_for(pipe).websocket_connect("/force") as ws:
        ws.send_bytes(jpeg())
        m = ws.receive_json()
    assert pipe.seen == [(640, 480)], "the server should decode the JPEG it was sent"
    assert m["found"] is True and m["stable"] is True
    assert m["grams"] == 200.0
    assert m["newtons"] == pytest.approx(1.96133, abs=1e-4)
    assert m["reason"] == ""


def test_a_lost_marker_is_null_not_a_stale_number():
    with client_for(StubPipeline(found=False)).websocket_connect("/force") as ws:
        ws.send_bytes(jpeg())
        m = ws.receive_json()
    assert m["found"] is False
    assert m["newtons"] is None and m["grams"] is None
    assert m["reason"] == "no-marker"


def test_the_connection_stays_open_and_time_advances():
    pipe = StubPipeline()
    with client_for(pipe).websocket_connect("/force") as ws:
        for _ in range(3):
            ws.send_bytes(jpeg(320, 240))
            assert ws.receive_json()["found"]
    assert pipe.seen == [(320, 240)] * 3
    assert pipe.times == sorted(pipe.times)


def test_an_undecodable_frame_is_reported_and_the_socket_survives():
    with client_for(StubPipeline()).websocket_connect("/force") as ws:
        ws.send_bytes(b"not a jpeg")
        assert "error" in ws.receive_json()
        ws.send_bytes(jpeg())
        assert ws.receive_json()["found"]


def test_tare_message_reports_success_and_failure():
    with client_for(StubPipeline(tare_ok=True)).websocket_connect("/force") as ws:
        ws.send_text("tare")
        assert ws.receive_json() == {"tare": True, "error": None}
    with client_for(StubPipeline(tare_ok=False)).websocket_connect("/force") as ws:
        ws.send_text("tare")
        m = ws.receive_json()
        assert m["tare"] is False and m["error"]


def test_each_connection_gets_its_own_pipeline():
    made = []
    app = app_for(lambda: made.append(StubPipeline()) or made[-1])
    client = TestClient(app)
    for _ in range(2):
        with client.websocket_connect("/force") as ws:
            ws.send_bytes(jpeg())
            ws.receive_json()
    assert len(made) == 2


def test_reading_message_rounds_timing_and_handles_missing_confidence():
    r = StubResult(100.0)
    r.detection = SimpleNamespace()
    m = reading_message(r, 3.14159)
    assert m["inferenceMs"] == 3.1 and m["confidence"] is None
