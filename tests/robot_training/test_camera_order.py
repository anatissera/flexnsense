"""Camera slot assignment used by the dataset converter, on synthetic images. No dataset needed."""

import io

import numpy as np
import pytest

pytest.importorskip("PIL")
pytest.importorskip("pyarrow")
from PIL import Image  # noqa: E402

from convert_normacore import build_reference, classify_slots  # noqa: E402


def _jpeg(kind: int, seed: int) -> bytes:
    """Three clearly different synthetic views, with a little per-frame noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:64, 0:64]
    base = [x * 4, y * 4, ((x // 8 + y // 8) % 2) * 255][kind].astype(np.float32)
    img = np.clip(base + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.stack([img] * 3, -1)).save(buf, format="JPEG")
    return buf.getvalue()


def _episode(order, n=20, seed=0):
    return [[{"jpeg": _jpeg(k, seed + i * 3 + s)} for s, k in enumerate(order)] for i in range(n)]


def test_same_order_as_the_reference_episode():
    ref = build_reference(_episode((0, 1, 2)))
    perm, margin = classify_slots(_episode((0, 1, 2), seed=100), ref)
    assert perm == (0, 1, 2) and margin > 0.02


def test_rotated_camera_order_is_recovered():
    ref = build_reference(_episode((0, 1, 2)))
    perm, _ = classify_slots(_episode((2, 0, 1), seed=200), ref)
    assert perm == (2, 0, 1)                    # perm[slot] = view index


def test_ambiguous_episode_aborts_instead_of_guessing():
    ref = build_reference(_episode((0, 1, 2)))
    with pytest.raises(RuntimeError):
        classify_slots(_episode((0, 0, 1), seed=300), ref)   # two slots show the same view
