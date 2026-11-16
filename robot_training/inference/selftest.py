#!/usr/bin/env python
"""Prueba SIN robot: sustituye Station por un episodio grabado (Parquet NormaCore) reproducido a 10 Hz.

Comprueba de punta a punta: autodetección de cámaras, construcción de la observación,
policy (local o remota), denormalización a ticks, GoalSafety y construcción del sync write.
Compara los goals enviados con los goals grabados en el episodio (bucle abierto: el
"robot" no se mueve con nuestras órdenes, así que es solo una comprobación de cordura).

    python selftest.py --parquet <episode_0030.parquet> local  --checkpoint <pretrained_model>
    python selftest.py --parquet <episode_0030.parquet> remote --server localhost:9000
"""

import argparse
import asyncio
import logging
import sys

import numpy as np
import pyarrow.parquet as pq

import control
import station_io
from station_io import FrameShapeError, Obs, goal_norm_to_ticks
from target.gen_python.protobuf.drivers.st3215 import st3215


class FakeStation:
    def __init__(self, parquet: str, drop_camera: int | None = None):
        t = pq.read_table(parquet).to_pydict()
        self.t, self.n, self.i, self.sent, self.drop = t, len(t["task"]), 0, [], drop_camera

    def obs(self, i: int) -> Obs:
        J = self.t["joints"][i]
        return Obs(frame_id=b"%d" % i,
                   state=np.array([j["position_norm"] for j in J], np.float32),
                   positions=np.array([j["position"] for j in J], np.int64),
                   ranges=[(j["range_min"], j["range_max"]) for j in J],
                   jpegs=[im["jpeg"] for im in self.t["images"][i]])

    async def fetch_fresh_obs(self, client, last_id, timeout, n_joints=8, n_images=3):
        await asyncio.sleep(0.1)
        if self.i >= self.n:
            raise KeyboardInterrupt
        o = self.obs(self.i)
        self.i += 1
        if self.drop is not None:      # como Station con una cámara sin abrir
            jpegs = [j for k, j in enumerate(o.jpegs) if k != self.drop]
            raise FrameShapeError(len(o.state), n_joints, jpegs, n_images)
        return o

    async def send_commands(self, client, cmds):
        for c in cmds:
            # decodificamos lo que de verdad iría por la red: DriverCommand.body = st3215.Command
            r = st3215.CommandReader(memoryview(bytes(c.body)))
            sw = r.get_sync_write()
            goals = [int.from_bytes(bytes(m.get_value()), "little") for m in sw.get_motors()]
            self.sent.append((self.i - 1, r.get_target_bus_serial(), sw.get_address(), goals))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--ticks", type=int, default=80)
    p.add_argument("--hand", choices=["dataset", "live"], default="dataset",
                   help="policies con yemas de los dedos: 'dataset' usa las columnas hand_* del Parquet (lo que "
                        "vio el entrenamiento con --hand dataset); 'live' las calcula con --hand-source")
    p.add_argument("--hand-source", default=None, help="con --hand live: wilor | mediapipe")
    p.add_argument("--wilor-root", default=None)
    p.add_argument("--drop-camera", type=int, default=None,
                   help="simula que Station no tiene abierta la cámara images[N] (prueba del diagnóstico)")
    sub = p.add_subparsers(dest="mode", required=True)
    pl = sub.add_parser("local"); pl.add_argument("--checkpoint", required=True); pl.add_argument("--device", default=None)
    pr = sub.add_parser("remote"); pr.add_argument("--server", default="localhost:9000"); pr.add_argument("--token", default=None)
    a = p.parse_args()

    fake = FakeStation(a.parquet, a.drop_camera)

    async def _noop(*args, **kw):
        return None

    async def _client(*args, **kw):
        return object()

    control.new_station_client = _client
    control.fetch_fresh_obs = fake.fetch_fresh_obs
    control.send_commands = fake.send_commands
    control.safe_enable_torque = _noop
    control.set_torque = _noop
    if a.hand == "dataset" and "hand_fingertips_2d" in fake.t:
        # fuente falsa: devuelve lo grabado para ese frame (se busca por el contenido de la imagen)
        recorded = {}
        for i in range(fake.n):
            rgb = station_io.decode_rgb(fake.t["images"][i][fake.t["hand_camera"][i]]["jpeg"])
            recorded[rgb.tobytes()] = (np.array(fake.t["hand_fingertips_2d"][i], np.float32)
                                       if fake.t["hand_detected"][i] else None)
        control.HAND_SOURCE_OVERRIDE = lambda rgb: recorded[rgb.tobytes()]

    sys.argv = [sys.argv[0], "--yes", "--max-ticks", str(a.ticks), "--task", fake.t["task"][0]]
    sys.argv += (["--hand-source", a.hand_source] if a.hand_source else []) + (["--wilor-root", a.wilor_root] if a.wilor_root else [])
    if a.mode == "local":
        import run_local
        sys.argv += ["--checkpoint", a.checkpoint] + (["--device", a.device] if a.device else [])
        entry = run_local.main
    else:
        import robot_client
        sys.argv += ["--server", a.server] + (["--token", a.token] if a.token else [])
        entry = robot_client.main
    logging.basicConfig(level=logging.WARNING)
    entry()

    # ── informe ──
    assert fake.sent, "no se envió ningún comando"
    buses = {s[1] for s in fake.sent}; addrs = {s[2] for s in fake.sent}
    err = []
    for i, _, _, goals in fake.sent:
        rec = np.array([j["goal"] for j in fake.t["joints"][i]])
        err.append(np.abs(np.array(goals) - rec))
    err = np.array(err)
    print(f"\nSELFTEST {a.mode}: {len(fake.sent)} sync writes | bus {buses} | registro {[hex(x) for x in addrs]}")
    print("  |goal enviado - goal grabado| ticks, mediana por motor:", np.median(err, 0).astype(int).tolist())


if __name__ == "__main__":
    main()
