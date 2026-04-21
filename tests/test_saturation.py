import numpy as np

from paid_media.saturation import hill_saturation, log_saturation


def test_hill_half_at_k():
    x = np.array([100.0])
    y = hill_saturation(x, k=100.0, s=1.5)
    assert abs(y[0] - 0.5) < 1e-6


def test_hill_monotone():
    x = np.linspace(0, 1000, 50)
    y = hill_saturation(x, k=200.0, s=1.5)
    assert np.all(np.diff(y) >= -1e-12)
    assert y[0] == 0.0
    assert y[-1] < 1.0 and y[-1] > 0.9


def test_log_saturation_at_zero():
    x = np.array([0.0, 10.0])
    y = log_saturation(x, k=5.0)
    assert y[0] == 0.0
    assert y[1] > 0


def test_saturation_rejects_bad_params():
    x = np.ones(3)
    try:
        hill_saturation(x, k=0.0, s=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError
    try:
        hill_saturation(x, k=1.0, s=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError
