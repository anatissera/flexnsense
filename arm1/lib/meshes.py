"""The arm's visual meshes, from the same URDF `lib.kinematics` reads for its geometry.

    from arm1.lib.meshes import load_visuals, posed_triangles
    visuals = load_visuals()                       # once; the STLs are 240k triangles
    tris = posed_triangles(chain, visuals, ticks)  # (N, 3, 3) in the base frame, metres

Why this exists: `lib.kinematics` answers where the joints are, which is enough to drive the
arm and enough to draw it as a stick figure, but not enough to recognise. A person checking
whether a simulated pull matches what they watched on the bench is comparing against Station's
3D view, and a line with dots on it is not the same picture.

Pure geometry and file reading: no renderer, no hardware. What comes out is triangles, and
what draws them is the caller's problem.

The STLs ship with the hardware in `hardware/elrobot/simulation/assets`, in millimetres, with
the URDF carrying the 0.001 scale and a per-visual origin. Both are applied here, so the
triangles come out in the same metres-in-the-base-frame that `Chain.tip_position` returns.
"""
from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arm1.lib.kinematics import DEFAULT_URDF, _rpy_matrix


@dataclass(frozen=True)
class Visual:
    """One link's mesh, already scaled and placed in its link's own frame."""

    link: str
    triangles: np.ndarray      # (N, 3, 3), metres, in the link frame


def read_binary_stl(path: Path) -> np.ndarray:
    """Triangles from a binary STL, as `(N, 3, 3)` in the file's own units.

    Binary only, which is what NormaCore ships. An ASCII STL starts with the word "solid", so
    it is detected and rejected rather than parsed as garbage.
    """
    data = path.read_bytes()
    if data[:5].lower() == b"solid" and len(data) < 84:
        raise ValueError(f"{path} looks like an ASCII STL; only binary is supported")
    count = struct.unpack("<I", data[80:84])[0]
    body = np.frombuffer(data[84: 84 + count * 50], dtype=np.uint8).reshape(count, 50)
    # 50 bytes per triangle: 12 of normal, 36 of three vertices, 2 of attributes.
    return body[:, 12:48].copy().view("<f4").reshape(count, 3, 3).astype(np.float64)


def simplify(triangles: np.ndarray, cell_m: float) -> np.ndarray:
    """Fewer triangles, by snapping vertices to a grid and dropping the ones that collapse.

    Vertex clustering rather than a real mesh decimation: it is a dozen lines of numpy, it
    needs no topology, and it cannot fail in a way that opens holes, because a triangle either
    survives with slightly moved corners or collapses to nothing because it was smaller than
    the grid. For these parts, mostly servo cases that are boxes with rounded corners, the
    result at a few millimetres is visually the same object with a tenth of the triangles.

    `cell_m` of 0 returns the mesh untouched.
    """
    if cell_m <= 0:
        return triangles
    snapped = np.round(triangles / cell_m) * cell_m
    # A triangle is degenerate once two of its corners land in the same cell.
    a, b, c = snapped[:, 0], snapped[:, 1], snapped[:, 2]
    keep = ~((a == b).all(1) | (b == c).all(1) | (a == c).all(1))
    return snapped[keep]


def load_visuals(urdf_path: str | Path = DEFAULT_URDF,
                 cell_m: float = 0.003) -> list[Visual]:
    """Every link's visual mesh, scaled, placed in its link frame and simplified.

    `cell_m` is passed to `simplify`; the default takes the arm from 240k triangles to a size
    an animation can redraw per frame. Pass 0 for the full meshes.
    """
    path = Path(urdf_path)
    root = ET.parse(path).getroot()
    out: list[Visual] = []
    for link in root.findall("link"):
        name = link.get("name") or ""
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                continue
            stl = path.parent / mesh.get("filename")
            if not stl.is_file():
                continue
            scale = np.array([float(v) for v in (mesh.get("scale") or "1 1 1").split()])
            tris = read_binary_stl(stl) * scale

            origin = visual.find("origin")
            if origin is not None:
                xyz = np.array([float(v) for v in (origin.get("xyz") or "0 0 0").split()])
                rpy = np.array([float(v) for v in (origin.get("rpy") or "0 0 0").split()])
                tris = tris @ _rpy_matrix(rpy).T + xyz

            out.append(Visual(name, simplify(tris, cell_m)))
    return out


def posed_triangles(chain, visuals: list[Visual], ticks: dict[int, float]
                    ) -> tuple[np.ndarray, np.ndarray]:
    """`(triangles, link_index)` for the whole arm at one pose, in the base frame.

    `link_index` says which visual each triangle came from, so a caller can colour the parts
    without re-deriving the grouping.
    """
    chunks, owners = [], []
    for i, visual in enumerate(visuals):
        R, p = chain.link_transform(visual.link, ticks)
        chunks.append(visual.triangles @ R.T + p)
        owners.append(np.full(len(visual.triangles), i))
    if not chunks:
        return np.zeros((0, 3, 3)), np.zeros(0, dtype=int)
    return np.concatenate(chunks), np.concatenate(owners)
