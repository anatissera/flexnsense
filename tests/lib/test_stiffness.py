import math

import pytest

from arm1.lib.stiffness import compare_directions, fit_stiffness
from arm1.lib.units import grams_force_to_newtons


def span_for(grams: float, k_n_per_m: float, x0_mm: float = 20.0) -> float:
    """The ruler reading a perfect spring of stiffness `k` would give at `grams` gf."""
    return x0_mm + grams_force_to_newtons(grams) / k_n_per_m * 1000.0


class TestFitStiffness:
    def test_recovers_a_known_stiffness(self):
        k = 53.5  # the estimate in docs/admittance-design.md, 600 gf over 110 mm
        points = [(g, span_for(g, k)) for g in (50, 100, 200, 300, 400, 500)]
        fit = fit_stiffness(points)
        assert fit.k_n_per_m == pytest.approx(k, rel=1e-6)
        assert fit.x0_mm == pytest.approx(20.0, abs=1e-6)
        assert fit.r2 == pytest.approx(1.0)
        assert fit.residual_max_mm < 1e-6
        assert fit.n_points == 6

    def test_datasheet_shortcut_agrees(self):
        """600 gf over 110 mm is 53.5 N/m: the one-minute version of this measurement."""
        fit = fit_stiffness([(0.0, 0.0), (600.0, 110.0)])
        assert fit.k_n_per_m == pytest.approx(53.5, abs=0.1)

    def test_k_gf_per_mm_is_the_same_number_in_measurement_units(self):
        fit = fit_stiffness([(0.0, 0.0), (600.0, 110.0)])
        assert fit.k_gf_per_mm == pytest.approx(600.0 / 110.0, rel=1e-9)

    def test_displacement_m_inverts_the_spring_law(self):
        fit = fit_stiffness([(g, span_for(g, 50.0)) for g in (100, 200, 300)])
        # Half a newton against 50 N/m is 10 mm, whatever the reference marks were.
        assert fit.displacement_m(0.5) == pytest.approx(0.01, rel=1e-6)

    def test_residual_is_reported_in_millimetres_of_ruler_error(self):
        k = 50.0
        points = [(g, span_for(g, k)) for g in (100, 200, 300, 400)]
        points[2] = (points[2][0], points[2][1] + 3.0)  # one reading misread by 3 mm
        fit = fit_stiffness(points)
        assert 1.0 < fit.residual_max_mm < 3.0  # the fit tilts, so it absorbs part of the error
        assert fit.r2 < 1.0

    def test_noise_does_not_move_k_much(self):
        k = 53.5
        jitter = [0.4, -0.5, 0.3, -0.2, 0.5, -0.4]  # +/- half a millimetre of ruler reading
        points = [(g, span_for(g, k) + j) for g, j in zip((50, 100, 200, 300, 400, 500), jitter)]
        assert fit_stiffness(points).k_n_per_m == pytest.approx(k, rel=0.05)

    def test_rejects_fewer_than_two_points(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            fit_stiffness([(100.0, 30.0)])

    def test_rejects_a_sweep_with_no_movement(self):
        with pytest.raises(ValueError, match="identical"):
            fit_stiffness([(100.0, 30.0), (200.0, 30.0), (300.0, 30.0)])

    def test_rejects_a_negative_slope_rather_than_returning_it(self):
        """More force must mean more stretch; the other sign is marks read the wrong way round."""
        with pytest.raises(ValueError, match="not positive"):
            fit_stiffness([(100.0, 40.0), (200.0, 30.0), (300.0, 20.0)])


class TestCompareDirections:
    def test_pairs_levels_measured_in_both_directions(self):
        up = [(100.0, 30.0), (200.0, 40.0), (300.0, 50.0)]
        down = [(300.0, 51.0), (200.0, 41.5), (100.0, 31.0)]
        pairs = compare_directions(up, down)
        assert [p.grams for p in pairs] == [100.0, 200.0, 300.0]
        assert [p.gap_mm for p in pairs] == pytest.approx([1.0, 1.5, 1.0])

    def test_ignores_levels_measured_only_one_way(self):
        pairs = compare_directions([(100.0, 30.0), (150.0, 35.0)], [(100.0, 31.0)])
        assert [p.grams for p in pairs] == [100.0]

    def test_gap_is_signed_so_a_stiffer_return_is_visible(self):
        pairs = compare_directions([(200.0, 40.0)], [(200.0, 39.0)])
        assert pairs[0].gap_mm == pytest.approx(-1.0)

    def test_no_overlap_is_empty_not_an_error(self):
        assert compare_directions([(100.0, 30.0)], [(200.0, 40.0)]) == []

    def test_hysteresis_in_millimetres_converts_to_a_force_uncertainty(self):
        """What the gap means downstream: 1 mm against 53.5 N/m is ~5 gf of ambiguity."""
        pairs = compare_directions([(200.0, 40.0)], [(200.0, 41.0)])
        gap_n = pairs[0].gap_mm / 1000.0 * 53.5
        assert math.isclose(gap_n, 0.0535, rel_tol=1e-9)
