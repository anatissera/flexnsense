#!/usr/bin/env python
"""Servidor de policy (GPU: este PC, un servidor o la nube). No toca el robot.

    python policy_server.py --checkpoint <.../pretrained_model> [--host 127.0.0.1] [--port 9000]

Por defecto escucha SOLO en 127.0.0.1: el portátil llega por un túnel SSH
(ssh -N -L 9000:localhost:9000 usuario@servidor). Si escuchas en 0.0.0.0, usa --token.
Atiende a un cliente a la vez (un brazo); la inferencia va serializada.
"""

import argparse
import asyncio
import hmac
import logging

import numpy as np

from policy_runner import PolicyRunner
from protocol import recv_msg, send_msg
from station_io import VIEWS

log = logging.getLogger("policy_server")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--n-action-steps", type=int, default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--token", default=None, help="secreto compartido; obligatorio si --host no es 127.0.0.1")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.token:
        raise SystemExit("escuchando fuera de localhost sin --token: no. Usa un túnel SSH o pon --token.")

    runner = PolicyRunner(args.checkpoint, device=args.device, n_action_steps=args.n_action_steps)
    # calentamiento (la primera inferencia en CUDA es lenta)
    runner.predict_chunk(np.full(8, 0.5, np.float32), {v: np.zeros((224, 224, 3), np.uint8) for v in VIEWS}, "")
    log.info("policy %s en %s | chunk %d | n_action_steps %d", runner.cfg.type, runner.device,
             runner.chunk_size, runner.n_action_steps)
    busy = asyncio.Lock()

    async def handle(reader, writer):
        peer = writer.get_extra_info("peername")
        if busy.locked():
            await send_msg(writer, {"type": "error", "msg": "servidor ocupado con otro cliente"})
            writer.close()
            return
        async with busy:
            log.info("cliente conectado: %s", peer)
            try:
                await send_msg(writer, {"type": "hello", "chunk_size": runner.chunk_size,
                                        "n_action_steps": runner.n_action_steps, "views": list(VIEWS),
                                        "policy": runner.cfg.type, "device": runner.device,
                                        "auth_required": bool(args.token)})
                if args.token:
                    h, _ = await asyncio.wait_for(recv_msg(reader), 10)
                    if h.get("type") != "auth" or not hmac.compare_digest(str(h.get("token", "")), args.token):
                        await send_msg(writer, {"type": "error", "msg": "token incorrecto"})
                        return
                while True:
                    h, blobs = await recv_msg(reader)
                    if h.get("type") != "obs":
                        await send_msg(writer, {"type": "error", "msg": f"tipo inesperado {h.get('type')}"})
                        continue
                    views = h["views"]
                    if sorted(views) != sorted(VIEWS) or len(blobs) != len(views):
                        await send_msg(writer, {"type": "error", "msg": f"vistas {views} / {len(blobs)} imágenes"})
                        continue
                    state = np.asarray(h["state"], dtype=np.float32)
                    chunk, ms = await asyncio.get_running_loop().run_in_executor(
                        None, runner.predict_from_jpegs, state, dict(zip(views, blobs)), h.get("task", ""))
                    await send_msg(writer, {"type": "chunk", "seq": h.get("seq"),
                                            "actions": np.round(chunk, 6).tolist(),
                                            "n_action_steps": runner.n_action_steps, "infer_ms": round(ms, 1)})
            except (asyncio.IncompleteReadError, ConnectionError) as e:
                log.info("cliente %s desconectado (%s)", peer, e.__class__.__name__)
            except Exception:
                log.exception("error atendiendo a %s", peer)
            finally:
                writer.close()

    async def serve():
        srv = await asyncio.start_server(handle, args.host, args.port)
        log.info("escuchando en %s:%d", args.host, args.port)
        async with srv:
            await srv.serve_forever()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
