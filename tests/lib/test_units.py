from arm1.lib.units import STANDARD_GRAVITY_M_S2, grams_force_to_newtons, newtons_to_grams_force


def test_known_value():
    # 200 gf is the number used throughout the docs as the therapy force cap.
    assert grams_force_to_newtons(200) == 200 / 1000 * STANDARD_GRAVITY_M_S2
    assert round(grams_force_to_newtons(200), 4) == 1.9613


def test_zero():
    assert grams_force_to_newtons(0) == 0.0
    assert newtons_to_grams_force(0) == 0.0


def test_round_trip():
    for g in (0.0, 1.0, 50.0, 200.0, 600.0, 1234.5):
        assert round(newtons_to_grams_force(grams_force_to_newtons(g)), 9) == round(g, 9)
