"""Protocolo mínimo cliente<->servidor de policy (TCP, sin dependencias).

Mensaje = b"FXN1" | u32 len_cabecera | cabecera JSON | u32 n_blobs | (u32 len | bytes) * n_blobs

  servidor -> cliente  {"type":"hello", "chunk_size", "n_action_steps", "views", "policy", "device"}
  cliente  -> servidor {"type":"auth", "token"}                       (solo si el servidor usa --token)
  cliente  -> servidor {"type":"obs", "seq", "state":[8], "views":[...], "task"} + un JPEG por vista
  servidor -> cliente  {"type":"chunk", "seq", "actions":[[8]*T], "n_action_steps", "infer_ms"}
  servidor -> cliente  {"type":"error", "msg"}
"""

from __future__ import annotations

import asyncio
import json
import struct

MAGIC = b"FXN1"
MAX_HEADER = 1 << 20
MAX_BLOB = 8 << 20


async def send_msg(writer: asyncio.StreamWriter, header: dict, blobs: list[bytes] = ()) -> None:
    h = json.dumps(header).encode()
    parts = [MAGIC, struct.pack("<I", len(h)), h, struct.pack("<I", len(blobs))]
    for b in blobs:
        parts += [struct.pack("<I", len(b)), b]
    writer.write(b"".join(parts))
    await writer.drain()


async def recv_msg(reader: asyncio.StreamReader) -> tuple[dict, list[bytes]]:
    if await reader.readexactly(4) != MAGIC:
        raise ConnectionError("protocolo desconocido (magic incorrecto)")
    (hl,) = struct.unpack("<I", await reader.readexactly(4))
    if hl > MAX_HEADER:
        raise ConnectionError("cabecera demasiado grande")
    header = json.loads(await reader.readexactly(hl))
    (nb,) = struct.unpack("<I", await reader.readexactly(4))
    if nb > 16:
        raise ConnectionError("demasiados blobs")
    blobs = []
    for _ in range(nb):
        (bl,) = struct.unpack("<I", await reader.readexactly(4))
        if bl > MAX_BLOB:
            raise ConnectionError("blob demasiado grande")
        blobs.append(await reader.readexactly(bl))
    return header, blobs
