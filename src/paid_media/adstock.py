"""Adstock transforms.

Adstock captures the fact that advertising effect persists beyond the week the
impression was served.  Two families are commonly used in paid-media modelling:

- **Geometric adstock**  ``a_t = x_t + lam * a_{t-1}``.  One-parameter, memoryless
  decay.  Good first-order model for always-on paid-search spend.
- **Delayed adstock** (Jin et al. 2017)  peaks ``theta`` weeks after exposure before
  decaying geometrically.  Good for campaigns with a consideration lag — branded
  display, video, OOH.

Both functions accept a 1-D ``numpy`` array and return a transformed array of
the same length.  Normalisation is optional so the transform can be interpreted
as "effective GRPs".
"""

from __future__ import annotations

import numpy as np


def geometric_adstock(
    x: np.ndarray,
    decay: float,
    normalize: bool = True,
) -> np.ndarray:
    """Geometric (Koyck) adstock.

    Parameters
    ----------
    x : np.ndarray
        Non-negative media vector (e.g. weekly spend or impressions).
    decay : float
        Carry-over rate in ``[0, 1)``.  ``0`` = no carryover, ``0.8`` = 80% of
        last week's effect carries into this week.
    normalize : bool
        If True, divide by ``1/(1 - decay)`` so the steady-state of a unit input
        is 1.  This keeps the scale interpretable when comparing across channels.
    """
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0, 1)")
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    run = 0.0
    for i, xi in enumerate(x):
        run = xi + decay * run
        out[i] = run
    if normalize:
        out = out * (1.0 - decay)
    return out


def delayed_adstock(
    x: np.ndarray,
    decay: float,
    theta: int = 0,
    max_lag: int = 12,
) -> np.ndarray:
    """Delayed-peak adstock (Jin et al. 2017).

    Weight at lag ``l`` is ``decay ** ((l - theta) ** 2)``; the series is
    convolved with these weights and truncated at ``max_lag``.
    """
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0, 1)")
    if theta < 0:
        raise ValueError("theta must be non-negative")
    x = np.asarray(x, dtype=float)
    lags = np.arange(max_lag + 1)
    w = decay ** ((lags - theta) ** 2)
    w = w / w.sum()
    padded = np.concatenate([np.zeros(max_lag), x])
    out = np.empty_like(x)
    for i in range(len(x)):
        window = padded[i : i + max_lag + 1][::-1]
        out[i] = float(np.dot(w, window))
    return out
