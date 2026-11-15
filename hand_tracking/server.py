"""Serve WiLoR over a WebSocket, so the dashboard can use it as a HandFrame source.

    python -m hand_tracking.server --wilor-root ../WiLoR

WiLoR cannot run in a browser: it is PyTorch and CUDA behind a 2.5 GB checkpoint. So the browser
keeps the camera (it already has the permission and the video element) and ships one JPEG per
frame here; this process runs the model and ships back HandFrames, the same shape the in-browser
MediaPipe tracker produces. Everything downstream in the dashboard therefore cannot tell which
tracker ran, except by looking for the `mano` block only WiLoR fills.

The protocol is deliberately one frame in flight at a time: the browser sends a frame and waits
for its reply before sending the next. That makes the browser's send rate follow whatever the GPU
can actually do (about 11 FPS on a 4 GB RTX 3050) instead of queueing up frames it cannot keep up
with, and it means no buffering or dropping logic on either side.

Frames are decoded, measured and discarded. Nothing is written to disk, which is the same promise
the in-browser tracker makes.
"""
#
# Note there is deliberately no `from __future__ import annotations` here, unlike the rest of
# hand_tracking. It would turn the annotations below into strings, and FastAPI resolves those
# against the module's globals; since fastapi is imported inside build_app (so that this module
# stays importable without it), `WebSocket` would not be found there. FastAPI would then read
# `ws: WebSocket` as a query parameter and reject every handshake with a 403.
import argparse
import sys
import time

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def build_app(wilor_root: str, *, fast: bool = True, conf: float = 0.3):
    """The FastAPI app, with the model loaded once up front rather than per connection."""
    from hand_tracking.wilor_source import WilorTracker

    print(f"loading WiLoR from {wilor_root} (fast={fast}) ...", flush=True)
    tracker = WilorTracker(wilor_root, fast=fast, conf=conf)
    print("WiLoR ready", flush=True)
    return app_for(tracker)


def app_for(tracker):
    """Wire the routes around anything with ``.track(image, t_ms)`` and ``.device``.

    Split out from :func:`build_app` so the routes can be tested against a stub, with no GPU and
    no checkpoint: see tests/hand_tracking/test_server.py.
    """
    import cv2
    import numpy as np
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.get("/health")
    def health() -> JSONResponse:
        """So the dashboard can say "server not running" rather than just failing to connect.

        The dashboard is served from a different port in dev, so this one response needs the CORS
        header. The WebSocket below does not: browsers do not apply CORS to WebSocket handshakes,
        and Starlette's CORSMiddleware rejects them outright, which is why it is not used here.
        """
        return JSONResponse(
            {"ok": True, "source": "wilor", "device": str(tracker.device)},
            headers={"Access-Control-Allow-Origin": "*"},
        )

    @app.websocket("/track")
    async def track(ws: WebSocket) -> None:
        await ws.accept()
        t0 = time.monotonic()
        try:
            while True:
                data = await ws.receive_bytes()
                image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    await ws.send_json({"error": "could not decode frame"})
                    continue
                started = time.perf_counter()
                frames = tracker.track(image, (time.monotonic() - t0) * 1000.0)
                await ws.send_json({
                    "frames": [f.to_dict() for f in frames],
                    "inferenceMs": round((time.perf_counter() - started) * 1000.0, 1),
                })
        except WebSocketDisconnect:
            pass

    return app


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wilor-root", help="path to a WiLoR checkout (default: WILOR_ROOT from .env, else ./WiLoR)")
    p.add_argument("--host", default=DEFAULT_HOST, help=f"default {DEFAULT_HOST}, i.e. this machine only")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--conf", type=float, default=0.3, help="hand detector confidence threshold")
    p.add_argument("--no-fast", action="store_true", help="skip fp16 and layer dropping")
    args = p.parse_args(argv)

    import uvicorn

    from lib.env import env_get

    root = args.wilor_root or env_get("WILOR_ROOT", default="WiLoR")
    app = build_app(root, fast=not args.no_fast, conf=args.conf)
    print(f"listening on ws://{args.host}:{args.port}/track", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
