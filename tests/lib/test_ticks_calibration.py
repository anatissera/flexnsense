import pytest

from lib.ticks_calibration import fit_ticks_per_metre, infer_yield_sign


class TestFitTicksPerMetre:
    def test_recovers_a_known_rate(self):
        rate = 3300.0  # ticks per metre
        points = [(round(rate * mm / 1000.0), mm) for mm in (10.0, 20.0, 30.0, 40.0)]
        fit = fit_ticks_per_metre(points)
        assert fit.ticks_per_metre == pytest.approx(rate, rel=1e-3)
        assert fit.n_points == 4

    def test_negative_ticks_with_negative_displacement_also_fit(self):
        """Commanding the opposite direction should move the mark the opposite way too."""
        rate = 3300.0
        points = [(round(rate * mm / 1000.0), mm) for mm in (10.0, -10.0, 20.0, -20.0)]
        fit = fit_ticks_per_metre(points)
        assert fit.ticks_per_metre == pytest.approx(rate, rel=1e-3)

    def test_an_inverted_convention_still_gives_a_positive_magnitude(self):
        """If positive ticks happened to shorten the measured span, ticks_per_metre is still
        reported as a positive magnitude - the direction itself is yield_sign's job, not this
        fit's."""
        rate = 3300.0
        points = [(round(rate * mm / 1000.0), -mm) for mm in (10.0, 20.0, 30.0)]
        fit = fit_ticks_per_metre(points)
        assert fit.ticks_per_metre == pytest.approx(rate, rel=1e-3)

    def test_small_ruler_noise_does_not_move_the_fit_much(self):
        rate = 3300.0
        points = [
            (round(rate * 10 / 1000), 10.0 + 0.4),
            (round(rate * 20 / 1000), 20.0 - 0.3),
            (round(rate * 30 / 1000), 30.0 + 0.2),
            (round(rate * 45 / 1000), 45.0 - 0.5),
        ]
        fit = fit_ticks_per_metre(points)
        assert fit.ticks_per_metre == pytest.approx(rate, rel=0.05)

    def test_rejects_fewer_than_two_points(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            fit_ticks_per_metre([(50, 15.0)])

    def test_rejects_a_trial_with_zero_measured_displacement(self):
        with pytest.raises(ValueError, match="zero displacement"):
            fit_ticks_per_metre([(50, 15.0), (100, 0.0)])

    def test_rejects_a_trial_that_disagrees_in_sign_with_the_rest(self):
        points = [(50, 15.0), (100, 30.0), (150, -45.0)]  # third trial moved the wrong way
        with pytest.raises(ValueError, match="opposite way"):
            fit_ticks_per_metre(points)

    def test_per_point_rates_are_reported_for_inspection(self):
        points = [(33, 10.0), (66, 20.0), (99, 30.0)]
        fit = fit_ticks_per_metre(points)
        assert len(fit.per_point_ticks_per_metre) == 3
        assert fit.max_relative_deviation < 0.01


class TestInferYieldSign:
    def test_positive_ticks_that_reduce_force_give_plus_one(self):
        trials = [(50, 200.0, 150.0), (100, 150.0, 80.0)]
        assert infer_yield_sign(trials) == 1

    def test_positive_ticks_that_increase_force_give_minus_one(self):
        trials = [(50, 100.0, 180.0), (100, 180.0, 260.0)]
        assert infer_yield_sign(trials) == -1

    def test_ignores_trials_within_the_noise_floor(self):
        trials = [(50, 200.0, 199.0), (100, 200.0, 120.0)]  # first barely moved, second real
        assert infer_yield_sign(trials) == 1

    def test_ignores_zero_delta_trials(self):
        trials = [(0, 200.0, 350.0), (50, 200.0, 140.0)]
        assert infer_yield_sign(trials) == 1

    def test_raises_when_nothing_cleared_the_noise_floor(self):
        trials = [(50, 200.0, 198.0), (-50, 200.0, 202.0)]
        with pytest.raises(ValueError, match="noise floor"):
            infer_yield_sign(trials)

    def test_raises_when_trials_disagree(self):
        trials = [(50, 200.0, 120.0), (50, 200.0, 280.0)]  # same delta, opposite outcome
        with pytest.raises(ValueError, match="disagree"):
            infer_yield_sign(trials)
