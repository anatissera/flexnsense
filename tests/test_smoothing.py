import math

import numpy as np
import pytest

from vision.smoothing import FilteredReading, ReadingFilter

FPS = 30.0
DT = 1.0 / FPS
SEEDS = range(10)


def run(f, values, t0=0.0):
    return [f.update(None if v is None else float(v), t0 + i * DT) for i, v in enumerate(values)]


@pytest.mark.parametrize("seed", SEEDS)
def test_constant_load_noise_reduced_and_stable(seed):
    rng = np.random.default_rng(seed)
    out = run(ReadingFilter(), 250.0 + rng.normal(0.0, 2.0, 90))
    assert all(isinstance(o, FilteredReading) for o in out)
    settled = out[30:]                                   # after 1 s
    g = np.array([o.grams for o in settled])
    assert g.std() < 1.0                                 # raw noise is 2 g
    assert abs(g.mean() - 250.0) < 1.0
    assert out[0].stable is False                        # not enough history yet
    assert not any(o.stable for o in out[:int(0.4 * FPS)])   # window not yet 80 % covered
    # Stable once the 0.5 s window has filled and the velocity estimate has settled (~0.5-0.8 s).
    # With 2 g noise the 3 g std / 5 g/s velocity limits are occasionally crossed by chance (a 300-seed
    # sweep gives >= 98 % stable frames for 95 % of seeds), so check the fraction, not every frame.
    assert np.mean([o.stable for o in out[int(0.8 * FPS):]]) >= 0.9
    assert out[-1].stable
    assert math.isnan(out[0].std_g) and 1.0 < out[-1].std_g < 4.0


def test_constant_load_noise_reduction_over_many_seeds():
    ratios = []
    for seed in range(50):
        rng = np.random.default_rng(1000 + seed)
        out = run(ReadingFilter(), 250.0 + rng.normal(0.0, 2.0, 90))
        ratios.append(np.std([o.grams for o in out[30:]]) / 2.0)
    assert np.mean(ratios) < 0.35                        # ~3.5x less jitter than the raw readings


@pytest.mark.parametrize("seed", SEEDS)
def test_step_followed_fast(seed):
    rng = np.random.default_rng(seed)
    n0, n1 = 60, 60
    raw = np.r_[np.zeros(n0), np.full(n1, 300.0)] + rng.normal(0.0, 2.0, n0 + n1)
    out = run(ReadingFilter(), raw)
    err = np.abs(np.array([o.grams for o in out[n0:]]) - 300.0)
    k = int(0.4 * FPS)                                   # 12 frames = 0.4 s after the step
    assert int(np.argmax(err < 2.0)) < k                 # within 2 g in under 0.4 s ...
    assert np.mean(err[k:] < 2.0) >= 0.85                # ... and mostly staying there (2 g raw noise)
    assert np.mean(err[k:]) < 1.2
    assert not out[n0 + 3].stable                        # a fresh step is not "stable" yet
    assert out[-1].stable


def test_step_response_time_noise_free():
    out = run(ReadingFilter(), [0.0] * 30 + [300.0] * 30)
    g = np.array([o.grams for o in out[30:]])
    first = int(np.argmax(np.abs(g - 300.0) < 2.0))
    assert first * DT <= 0.1                             # outlier hold of 3 frames, then re-init
    assert np.all(np.abs(g[first:] - 300.0) < 2.0)


def test_small_step_below_outlier_gate_is_followed():
    out = run(ReadingFilter(), [100.0] * 60 + [110.0] * 30)
    g = np.array([o.grams for o in out[60:]])
    assert np.all(np.abs(g[int(0.3 * FPS):] - 110.0) < 1.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_not_stable_during_ramp(seed):
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0.0, 500.0, int(2 * FPS))         # 0 -> 500 g in 2 s
    raw = np.r_[np.zeros(30), ramp, np.full(45, 500.0)] + rng.normal(0.0, 2.0, 30 + ramp.size + 45)
    out = run(ReadingFilter(), raw)
    assert out[29].stable                                # at rest before the ramp
    during = out[30 + 3:30 + ramp.size]                  # allow 0.1 s detection latency
    assert not any(o.stable for o in during)
    assert all(o.grams is not None and o.velocity_g_s > 100 for o in out[30 + 15:30 + ramp.size])
    assert out[-1].stable and abs(out[-1].grams - 500.0) < 3.0


def test_short_gap_keeps_value():
    rng = np.random.default_rng(1)
    vals = list(200.0 + rng.normal(0.0, 2.0, 30)) + [None] * 6 + list(200.0 + rng.normal(0.0, 2.0, 10))
    out = run(ReadingFilter(), vals)
    for o in out[30:36]:                                 # 0.2 s of no detection
        assert o.grams is not None and abs(o.grams - 200.0) < 2.0
        assert o.raw is None
    assert abs(out[-1].grams - 200.0) < 2.0


def test_long_gap_gives_none_then_recovers():
    vals = [200.0] * 30 + [None] * 30 + [350.0] * 5      # 1 s gap, load changed meanwhile
    out = run(ReadingFilter(), vals)
    gap = out[30:60]
    assert gap[0].grams is not None                      # just lost: still predicting
    assert all(o.grams is None and not o.stable for o in gap[int(0.6 * FPS):])
    assert out[60].grams == pytest.approx(350.0)         # re-initialised on the next measurement
    assert out[-1].grams == pytest.approx(350.0, abs=0.5)


@pytest.mark.parametrize("seed", SEEDS)
def test_single_outlier_spike_rejected(seed):
    rng = np.random.default_rng(seed)
    raw = 250.0 + rng.normal(0.0, 2.0, 60)
    raw[45] += 80.0
    out = run(ReadingFilter(), raw)
    g = np.array([o.grams for o in out])
    assert np.max(np.abs(g[45:] - g[44])) < 5.0
    assert out[45].raw == pytest.approx(raw[45])
    assert out[50].stable                                # a dropped spike does not break stability


def test_irregular_timestamps():
    rng = np.random.default_rng(3)
    f = ReadingFilter()
    t = 0.0
    out = []
    for _ in range(120):
        t += rng.uniform(0.02, 0.05)
        out.append(f.update(150.0 + rng.normal(0.0, 1.0), t))
    assert abs(out[-1].grams - 150.0) < 1.0 and out[-1].stable


def test_reset():
    f = ReadingFilter()
    run(f, [100.0] * 20)
    f.reset()
    o = f.update(None, 1.0)
    assert o.grams is None and not o.stable
    assert f.update(42.0, 1.1).grams == pytest.approx(42.0)


def test_stable_drops_on_first_frames_of_a_ramp():
    """Regression: a resting load starts ramping at 300 g/s; no frame may be 'stable' while it lags > 3 g."""
    from vision.smoothing import ReadingFilter

    f = ReadingFilter()
    fps = 30.0
    for i in range(30):                       # 1 s at rest, noise-free
        f.update(100.0, i / fps)
    bad = []
    for k in range(1, 31):                    # ramp up 10 g per frame
        truth = 100.0 + 10.0 * k
        r = f.update(truth, (30 + k) / fps)
        if r.stable and abs(r.grams - truth) > 3.0:
            bad.append((k, round(r.grams - truth, 2)))
    assert not bad, f"stable while lagging: {bad}"
