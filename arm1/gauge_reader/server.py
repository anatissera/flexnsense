"""Serve the gauge reader over a WebSocket, so the dashboard can show a live force.

    python -m gauge_reader.server                      # uses config/scale_calib.json
    python -m gauge_reader.server --calib other.json --port 8766

The browser owns the camera (so it can pick which one watches the scale), grabs JPEG frames and sends them
here. Each frame goes through the same pipeline as `python -m gauge_reader run`, and one JSON reading comes back:

    {"found": true, "newtons": 1.96, "grams": 200.0, "stable": true, "confidence": 0.9,
     "reason": "", "tMs": 1234.5, "inferenceMs": 3.1}

`newtons` and `grams` are null whenever there is no recent measurement (marker lost, scale out of view). A
consumer must treat null as "no force", never as the last value. Frames are decoded, measured and dropped:
nothing is written to disk. A text message "tare" zeroes the unloaded position for that connection.

Each connection gets its own pipeline, so reconnecting starts with a clean filter.

Note there is deliberately no `from __future__ import annotations` here, for the same reason as in
hand_tracking/server.py: fastapi is imported inside a function, so a stringified `ws: WebSocket` could not be
resolved and every handshake would be rejected with a 403.
"""
import argparse
import sys
import time

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766
DEFAULT_CALIB = "config/scale_calib.json"


def reading_message(result, in_ms: float) -> dict:
    """One pipeline result as the JSON the dashboard reads. Pure, so it is tested without a socket."""
    conf = getattr(result.detection, "confidence", None)
    return {
        "found": bool(result.found),
        "newtons": result.newtons,
        "grams": result.grams,
        "stable": bool(result.stable),
        "confidence": None if conf is None else float(conf),
        "reason": result.reason,
        "inferenceMs": round(in_ms, 1),
    }


def app_for(make_pipeline, *, describe: str = ""):
    """Wire the routes around a zero-argument factory returning a fresh pipeline per connection.

    A pipeline is anything with ``.process(frame_bgr, ts_s)`` returning a result with ``found``, ``newtons``,
    ``grams``, ``stable``, ``reason`` and ``detection.confidence``, and ``.tare()``.
    """
    import cv2
    import numpy as np
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.get("/health")
    def health() -> JSONResponse:
        """CORS header for the same reason as the WiLoR server: the dashboard is on another port in dev."""
        return JSONResponse({"ok": True, "source": "gauge", "calibration": describe},
                            headers={"Access-Control-Allow-Origin": "*"})

    @app.websocket("/force")
    async def force(ws: WebSocket) -> None:
        await ws.accept()
        pipe = make_pipeline()
        t0 = time.monotonic()
        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("text") is not None:
                    if message["text"].strip() == "tare":
                        pos = pipe.tare()
                        await ws.send_json({"tare": pos is not None,
                                            "error": None if pos is not None else "not enough recent detections"})
                    continue
                data = message.get("bytes")
                if data is None:
                    continue
                image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    await ws.send_json({"error": "could not decode frame"})
                    continue
                started = time.perf_counter()
                result = pipe.process(image, time.monotonic() - t0)
                await ws.send_json(reading_message(result, (time.perf_counter() - started) * 1000.0))
        except WebSocketDisconnect:
            pass

    return app


def build_app(calib_path: str = DEFAULT_CALIB):
    from arm1.gauge_reader.cli import _load_calibration, make_pipeline

    calib = _load_calibration(calib_path)
    describe = make_pipeline(calib).describe()
    print(f"calibration {calib_path}: {describe}", flush=True)
    return app_for(lambda: make_pipeline(calib), describe=describe)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib", default=DEFAULT_CALIB, help=f"calibration file (default {DEFAULT_CALIB})")
    p.add_argument("--host", default=DEFAULT_HOST, help=f"default {DEFAULT_HOST}, i.e. this machine only")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = p.parse_args(argv)

    import uvicorn

    app = build_app(args.calib)
    print(f"listening on ws://{args.host}:{args.port}/force", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
