"""The arm's visual meshes: loaded, placed, and cheap enough to animate.

These are drawing-only, so the bar is different from `test_kinematics.py`: nothing here decides
where the arm goes. What is worth pinning is that the meshes land in the same frame as the
kinematics (a picture that disagrees with the geometry is worse than no picture) and that the
simplification stays honest about the shape it is simplifying.
"""
import struct
from pathlib import Path

import numpy as np
import pytest

from arm1.lib.kinematics import load_chain
from arm1.lib.meshes import load_visuals, posed_triangles, read_binary_stl, simplify

REPO_ROOT = Path(__file__).resolve().parents[2]
URDF = REPO_ROOT / "norma-core/hardware/elrobot/simulation/elrobot_follower.urdf"

POSE = {1: 2045, 2: 1468, 3: 2458, 4: 2775, 5: 1877, 6: 1524, 7: 3274, 8: 1301}

pytestmark = pytest.mark.skipif(
    not URDF.is_file(),
    reason="needs the norma-core clone (hardware/elrobot/simulation); see SETUP.md",
)


@pytest.fixture(scope="module")
def chain():
    return load_chain(URDF)


@pytest.fixture(scope="module")
def visuals():
    return load_visuals(URDF)


# ---------------------------------------------------------------- reading the files
def test_an_ascii_stl_is_rejected_rather_than_misread(tmp_path):
    """Binary is what NormaCore ships. Parsing ASCII as binary yields plausible-looking
    nonsense rather than an error, so it has to be caught."""
    f = tmp_path / "ascii.stl"
    f.write_bytes(b"solid thing\nfacet normal 0 0 1\n")
    with pytest.raises(ValueError, match="ASCII"):
        read_binary_stl(f)


def test_a_binary_stl_round_trips(tmp_path):
    tri = np.array([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype="<f4")
    payload = b"\0" * 80 + struct.pack("<I", 1)
    payload += np.zeros(3, dtype="<f4").tobytes() + tri.tobytes() + b"\0\0"
    f = tmp_path / "one.stl"
    f.write_bytes(payload)
    assert np.allclose(read_binary_stl(f), tri)


# ---------------------------------------------------------------- simplification
def test_simplify_keeps_the_object_the_same_size(visuals):
    """It may drop triangles; it may not shrink or move the part. A bounding box that moved
    would put the drawn arm somewhere the real one is not."""
    full = load_visuals(URDF, cell_m=0.0)
    for coarse, fine in zip(visuals, full):
        assert coarse.link == fine.link
        lo_c, hi_c = coarse.triangles.reshape(-1, 3).min(0), coarse.triangles.reshape(-1, 3).max(0)
        lo_f, hi_f = fine.triangles.reshape(-1, 3).min(0), fine.triangles.reshape(-1, 3).max(0)
        assert np.allclose(lo_c, lo_f, atol=0.004), coarse.link
        assert np.allclose(hi_c, hi_f, atol=0.004), coarse.link


def test_simplify_actually_removes_triangles(visuals):
    full = sum(len(v.triangles) for v in load_visuals(URDF, cell_m=0.0))
    coarse = sum(len(v.triangles) for v in visuals)
    assert coarse < full / 5, "an animation redraws every triangle on every frame"


def test_simplify_with_no_grid_changes_nothing():
    tris = np.random.default_rng(0).normal(size=(10, 3, 3))
    assert np.array_equal(simplify(tris, 0.0), tris)


def test_a_collapsed_triangle_is_dropped_not_kept_as_a_sliver():
    """Two corners in the same cell is a zero-area triangle. Keeping it would hand the
    renderer a degenerate face with no normal."""
    tris = np.array([[[0.0, 0, 0], [0.0001, 0, 0], [0, 1, 0]]])
    assert len(simplify(tris, 0.01)) == 0


# ---------------------------------------------------------------- agreeing with the kinematics
def test_the_gripper_mesh_surrounds_the_tip_the_kinematics_reports(chain, visuals):
    """The one that makes the picture trustworthy. If the meshes were in a different frame, or
    scaled wrongly, the drawn gripper would not sit where `tip_position` says it does."""
    tris, owners = posed_triangles(chain, visuals, POSE)
    tip = chain.tip_position(POSE)

    gripper = [i for i, v in enumerate(visuals) if v.link == "Gripper_Base_v1_1"]
    assert gripper, "no gripper visual found"
    points = tris[np.isin(owners, gripper)].reshape(-1, 3)
    assert (points.min(0) <= tip + 1e-6).all()
    assert (points.max(0) >= tip - 1e-6).all()


def test_the_whole_arm_is_the_size_of_the_real_one(chain, visuals):
    """The documented reach is about 25 cm. A metre-scale or millimetre-scale mistake in the
    mesh units would show up here as a factor of 1000."""
    tris, _ = posed_triangles(chain, visuals, POSE)
    span = tris.reshape(-1, 3).max(0) - tris.reshape(-1, 3).min(0)
    assert (span < 0.5).all(), span
    assert span.max() > 0.15, span


def test_moving_a_joint_moves_only_what_hangs_below_it(chain, visuals):
    """A serial chain drawn correctly: turning motor 4 must leave the base alone and move the
    gripper. If every link moved, the per-link transforms would be wrong."""
    turned = {**POSE, 4: POSE[4] + 300}
    before, owners = posed_triangles(chain, visuals, POSE)
    after, _ = posed_triangles(chain, visuals, turned)

    by_link = {v.link: i for i, v in enumerate(visuals)}
    base = np.isin(owners, [by_link["base_link"]])
    gripper = np.isin(owners, [by_link["Gripper_Base_v1_1"]])
    assert np.allclose(before[base], after[base])
    assert not np.allclose(before[gripper], after[gripper])
