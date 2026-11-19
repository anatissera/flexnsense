"""The WiLoR server's routes, against a stub tracker: no GPU, no checkpoint, no camera.

The regression these guard is specific. FastAPI resolves a handler's annotations against its
module globals, so with ``from __future__ import annotations`` in server.py (where fastapi is
imported inside a function, not at module level) it cannot see that ``ws: WebSocket`` is a
WebSocket. It then treats ``ws`` as a required query parameter and rejects every handshake with
a 403, which looks exactly like a networking problem and is not one.
"""
from __future__ import annotations

import numpy as np
import pytest

from arm1.hand_tracking.frame import FLEXION_JOINTS, HandFrame, ManoParams

fastapi = pytest.importorskip("fastapi", reason="fastapi only lives in the WiLoR environment")
pytest.importorskip("cv2")

from fastapi.testclient import TestClient  # noqa: E402

from arm1.hand_tracking.server import app_for  # noqa: E402


class StubTracker:
    """Stands in for WilorTracker: records what it was given, returns one fixed frame."""

    device = "cpu"

    def __init__(self, hands: int = 1) -> None:
        self.hands = hands
        self.seen: list[tuple[int, int]] = []

    def track(self, image, t_ms: float) -> list[HandFrame]:
        self.seen.append((image.shape[1], image.shape[0]))
        points = [(i / 100, i / 200, i / 400) for i in range(21)]
        return [
            HandFrame(
                source="wilor",
                t_ms=t_ms,
                handedness="right",
                landmarks=points,
                world=points,
                angles={name: 0.1 for name in FLEXION_JOINTS},
                tip_wrist_mm=90.0,
                mano=ManoParams(hand_pose=[[0.0, 0.0, 0.0]] * 15, global_orient=[0.0] * 3, betas=[0.0] * 10),
            )
            for _ in range(self.hands)
        ]


def jpeg(width: int = 640, height: int = 480) -> bytes:
    import cv2

    image = np.zeros((height, width, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", image)[1].tobytes()


def test_health_reports_the_device_and_allows_the_dashboards_origin():
    client = TestClient(app_for(StubTracker()))
    response = client.get("/health")
    assert response.json() == {"ok": True, "source": "wilor", "device": "cpu"}
    # The dashboard runs on another port in dev, so this one response needs the header.
    assert response.headers["access-control-allow-origin"] == "*"


def test_a_frame_in_gets_a_handframe_back():
    tracker = StubTracker()
    client = TestClient(app_for(tracker))
    with client.websocket_connect("/track") as ws:
        ws.send_bytes(jpeg())
        message = ws.receive_json()
    assert tracker.seen == [(640, 480)], "the server should decode the JPEG it was sent"
    assert "inferenceMs" in message
    frame = message["frames"][0]
    assert frame["source"] == "wilor"
    assert set(frame["angles"]) == set(FLEXION_JOINTS)
    assert len(frame["mano"]["handPose"]) == 15


def test_every_detected_hand_comes_back():
    client = TestClient(app_for(StubTracker(hands=2)))
    with client.websocket_connect("/track") as ws:
        ws.send_bytes(jpeg())
        assert len(ws.receive_json()["frames"]) == 2


def test_the_connection_stays_open_for_more_frames():
    """One frame in flight at a time means the socket is reused, not reopened per frame."""
    tracker = StubTracker()
    client = TestClient(app_for(tracker))
    with client.websocket_connect("/track") as ws:
        for _ in range(3):
            ws.send_bytes(jpeg(320, 240))
            assert ws.receive_json()["frames"]
    assert tracker.seen == [(320, 240)] * 3


def test_an_undecodable_frame_is_reported_not_crashed_on():
    client = TestClient(app_for(StubTracker()))
    with client.websocket_connect("/track") as ws:
        ws.send_bytes(b"this is not a jpeg")
        assert "error" in ws.receive_json()
        # The socket survives it, so one bad frame does not end the session.
        ws.send_bytes(jpeg())
        assert ws.receive_json()["frames"]
