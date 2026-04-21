import numpy as np

from paid_media.adstock import delayed_adstock, geometric_adstock


def test_geometric_adstock_steady_state_unit_input():
    x = np.ones(200)
    y = geometric_adstock(x, decay=0.7)
    # Normalised: steady state of a unit input is 1.
    assert abs(y[-1] - 1.0) < 1e-6


def test_geometric_adstock_respects_carryover():
    x = np.zeros(20)
    x[0] = 1.0
    y = geometric_adstock(x, decay=0.5, normalize=False)
    # y[t] = 0.5 ** t
    assert abs(y[0] - 1.0) < 1e-9
    assert abs(y[1] - 0.5) < 1e-9
    assert abs(y[5] - 0.5**5) < 1e-9


def test_delayed_adstock_has_delayed_peak():
    x = np.zeros(30)
    x[5] = 1.0
    y = delayed_adstock(x, decay=0.6, theta=2)
    # Peak should land somewhere in [5+0, 5+4] given theta=2 and symmetric decay.
    assert 5 <= int(np.argmax(y)) <= 5 + 4


def test_geometric_adstock_rejects_bad_decay():
    x = np.ones(5)
    try:
        geometric_adstock(x, decay=1.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
