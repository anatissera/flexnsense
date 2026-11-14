import pytest

from lib.geometry import Axis


def test_axis_project_and_back():
    ax = Axis((10.0, 20.0), (10.0, 120.0))              # pointing down the image
    assert ax.length == pytest.approx(100.0)
    assert ax.angle_deg == pytest.approx(90.0)
    t, n = ax.project((15.0, 70.0))
    assert (t, n) == (pytest.approx(50.0), pytest.approx(-5.0))
    assert ax.point_at(t, n) == (pytest.approx(15.0), pytest.approx(70.0))


def test_axis_dict_roundtrip_and_degenerate():
    ax = Axis((1.5, 2.5), (30.0, 40.0))
    assert Axis.from_dict(ax.to_dict()) == ax
    with pytest.raises(ValueError):
        _ = Axis((1.0, 1.0), (1.0, 1.0)).unit
