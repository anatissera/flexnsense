#!/usr/bin/env python
"""Cliente en el PORTÁTIL (donde están el robot y Station). La policy corre en un servidor remoto.

No necesita torch ni lerobot: solo numpy, pillow y norma-core (station_py + protobufs generados).

    # 1) túnel al servidor (en otra terminal del portátil):
    ssh -N -L 9000:localhost:9000 usuario@servidor
    # 2) cliente:
    python robot_client.py --server localhost:9000 --dry-run
    python robot_client.py --server localhost:9000
"""

import argparse
import asyncio
import logging
import os

import numpy as np

from control import add_robot_args, run_control
from protocol import recv_msg, send_msg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default="localhost:9000", help="host:puerto del policy_server")
    p.add_argument("--token", default=os.environ.get("POLICY_TOKEN"), help="si el servidor usa --token (o env POLICY_TOKEN)")
    p.add_argument("--request-timeout", type=float, default=2.0, help="segundos máx. esperando un chunk")
    add_robot_args(p)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("robot_client")

    async def amain():
        host, port = args.server.rsplit(":", 1)
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, int(port)), 10)
        hello, _ = await recv_msg(reader)
        if hello.get("type") != "hello":
            raise SystemExit(f"el servidor respondió: {hello}")
        print(f"servidor: policy {hello['policy']} en {hello['device']} | chunk {hello['chunk_size']} | "
              f"ejecuta {hello['n_action_steps']} | vistas {hello['views']}")
        if hello.get("auth_required"):
            await send_msg(writer, {"type": "auth", "token": args.token or ""})
        args.hand_dim = int(hello.get("state_dim", 8)) - 8     # 5 si la policy usa las yemas de los dedos
        seq = 0

        async def get_chunk(state, jpeg_by_view, task, prev=None):
            nonlocal seq
            seq += 1
            views = list(hello["views"])
            t0 = asyncio.get_running_loop().time()
            msg = {"type": "obs", "seq": seq, "state": np.asarray(state).tolist(), "views": views, "task": task}
            blobs = [jpeg_by_view[v] for v in views]
            if prev is not None and hello.get("n_obs_steps", 1) > 1:   # la policy pide también el frame anterior
                msg["prev_state"] = np.asarray(prev[0]).tolist()
                blobs += [prev[1][v] for v in views]
            await send_msg(writer, msg, blobs)
            h, _ = await asyncio.wait_for(recv_msg(reader), args.request_timeout)
            if h.get("type") == "error":
                raise SystemExit(f"el servidor rechazó la petición: {h.get('msg')}")
            if h.get("type") != "chunk" or h.get("seq") != seq:
                raise RuntimeError(f"respuesta inesperada del servidor: {h}")
            rtt = (asyncio.get_running_loop().time() - t0) * 1000
            return np.asarray(h["actions"], np.float32), int(h["n_action_steps"]), \
                f"(remoto: inferencia {h['infer_ms']:.0f} ms, ida+vuelta {rtt:.0f} ms)"

        try:
            await run_control(args, get_chunk, log)
        finally:
            writer.close()

    asyncio.run(amain())


if __name__ == "__main__":
    main()
