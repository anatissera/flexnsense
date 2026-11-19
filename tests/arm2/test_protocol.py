"""Wire format between robot_client.py and policy_server.py. Pure asyncio, no network."""

import asyncio
import struct

import pytest

from protocol import MAGIC, recv_msg, send_msg


class _Writer:
    def __init__(self):
        self.buf = b""

    def write(self, data):
        self.buf += data

    async def drain(self):
        pass


def _roundtrip(header, blobs=()):
    async def go():
        w = _Writer()
        await send_msg(w, header, list(blobs))
        r = asyncio.StreamReader()
        r.feed_data(w.buf)
        r.feed_eof()
        return await recv_msg(r)

    return asyncio.run(go())


def test_header_and_blobs_survive_a_roundtrip():
    header = {"type": "obs", "seq": 3, "state": [0.1] * 8, "views": ["side", "gripper", "table"]}
    blobs = [b"\xff\xd8jpeg-one", b"", bytes(range(256))]
    h, b = _roundtrip(header, blobs)
    assert h == header and b == blobs


def test_message_without_blobs():
    h, b = _roundtrip({"type": "hello", "chunk_size": 30})
    assert h["chunk_size"] == 30 and b == []


def _read(raw):
    async def go():
        r = asyncio.StreamReader()
        r.feed_data(raw)
        r.feed_eof()
        return await recv_msg(r)

    return asyncio.run(go())


def test_wrong_magic_is_rejected():
    with pytest.raises(ConnectionError):
        _read(b"HTTP" + b"\x00" * 16)


def test_oversized_header_is_rejected_before_reading_it():
    with pytest.raises(ConnectionError):
        _read(MAGIC + struct.pack("<I", 1 << 30))


def test_truncated_message_raises():
    with pytest.raises(asyncio.IncompleteReadError):
        _read(MAGIC + struct.pack("<I", 100) + b"{}")
