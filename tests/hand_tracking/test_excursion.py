import math

import pytest

from arm1.hand_tracking.excursion import fit_excursion, tip_minus_wrist


def make_21(tip: tuple[float, float, float], wrist: tuple[float, float, float] = (0.0, 0.0, 0.0)):
    """21 landmarks with only wrist (0) and pinky tip (20) set to anything meaningful."""
    points = [(0.0, 0.0, 0.0)] * 21
    points[0] = wrist
    points[20] = tip
    return points


class TestTipMinusWrist:
    def test_subtracts_wrist_from_tip(self):
        points = make_21(tip=(0.05, 0.02, 0.0), wrist=(0.01, 0.01, 0.0))
        assert tip_minus_wrist(points, "pinky") == pytest.approx((0.04, 0.01, 0.0))

    def test_cancels_a_shared_origin_shift(self):
        """The whole point of tip-minus-wrist: a drifting hand-centred origin cancels out."""
        shift = (0.1, -0.05, 0.02)
        a = tip_minus_wrist(make_21(tip=(0.05, 0.0, 0.0), wrist=(0.0, 0.0, 0.0)), "pinky")
        b = tip_minus_wrist(
            make_21(tip=(0.05 + shift[0], shift[1], shift[2]), wrist=shift), "pinky"
        )
        assert a == pytest.approx(b)

    def test_rejects_unknown_finger(self):
        with pytest.raises(ValueError, match="unknown finger"):
            tip_minus_wrist(make_21(tip=(0.0, 0.0, 0.0)), "little")

    def test_rejects_too_few_landmarks(self):
        with pytest.raises(ValueError, match="21 landmarks"):
            tip_minus_wrist([(0.0, 0.0, 0.0)] * 5, "pinky")


class TestFitExcursion:
    def test_a_straight_line_gives_its_own_length(self):
        """Motion straight along one axis: excursion equals the end-to-end distance exactly."""
        vectors = [(0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.02, 0.0, 0.0), (0.03, 0.0, 0.0)]
        result = fit_excursion(vectors)
        assert result.excursion_m == pytest.approx(0.03, abs=1e-9)
        assert result.variance_explained == pytest.approx(1.0, abs=1e-6)
        assert result.perpendicular_rms_m == pytest.approx(0.0, abs=1e-9)
        assert result.n_samples == 4

    def test_a_tilted_line_is_still_recovered(self):
        """The axis need not be one of the three world axes."""
        direction = (3.0, 4.0, 0.0)  # length 5, deliberately not normalised going in
        norm = math.sqrt(sum(d * d for d in direction))
        unit = tuple(d / norm for d in direction)
        vectors = [tuple(u * t for u in unit) for t in (0.0, 0.01, 0.02, 0.03, 0.04)]
        result = fit_excursion(vectors)
        assert result.excursion_m == pytest.approx(0.04, abs=1e-6)
        assert result.variance_explained == pytest.approx(1.0, abs=1e-5)

    def test_off_axis_wobble_is_reported_not_hidden(self):
        """A real recording wanders a little; the wobble shows up as non-zero perpendicular RMS."""
        vectors = [
            (0.0, 0.001, 0.0), (0.01, -0.001, 0.0), (0.02, 0.001, 0.0),
            (0.03, -0.001, 0.0), (0.04, 0.0, 0.0),
        ]
        result = fit_excursion(vectors)
        assert result.excursion_m == pytest.approx(0.04, abs=0.001)
        assert 0.0 < result.perpendicular_rms_m < 0.002
        assert 0.9 < result.variance_explained < 1.0

    def test_pure_noise_with_no_dominant_direction_has_low_variance_explained(self):
        """Motion with no real 1-D structure should not be reported as a clean axis."""
        vectors = [
            (0.001, 0.0, 0.0), (0.0, 0.001, 0.0), (0.0, 0.0, 0.001),
            (-0.001, 0.0, 0.0), (0.0, -0.001, 0.0), (0.0, 0.0, -0.001),
        ]
        result = fit_excursion(vectors)
        assert result.variance_explained < 0.6

    def test_sign_of_axis_does_not_affect_excursion(self):
        vectors = [(0.0, 0.0, 0.0), (-0.01, 0.0, 0.0), (-0.02, 0.0, 0.0)]
        result = fit_excursion(vectors)
        assert result.excursion_m == pytest.approx(0.02, abs=1e-9)

    def test_rejects_fewer_than_two_samples(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            fit_excursion([(0.0, 0.0, 0.0)])

    def test_realistic_scale_is_centimetres_not_metres(self):
        """Sanity: a plausible finger excursion is a few cm, i.e. a few hundredths of a metre."""
        vectors = [(0.0, 0.0, 0.0), (0.015, 0.002, -0.001), (0.03, 0.0, 0.0)]
        result = fit_excursion(vectors)
        assert 0.01 < result.excursion_m < 0.1
