"""Saturation response curves.

Diminishing returns are the second pillar of media-mix modelling (adstock being
the first).  We ship two shapes:

- **Hill (sigmoidal)**  ``y = x^s / (k^s + x^s)``.  S-shape; slow start, inflection
  at ``k``, saturates at 1.  Appropriate for channels with an awareness threshold
  (video, display).
- **Logarithmic**  ``y = log1p(x / k)``.  Concave from the origin.  Appropriate for
  channels that convert the moment they're seen (paid search, branded keywords).

Both return unit-scaled response (multiply by a channel ``beta`` to get revenue
or conversions).
"""

from __future__ import annotations

import numpy as np


def hill_saturation(x: np.ndarray, k: float, s: float = 1.5) -> np.ndarray:
    """Hill response.

    Parameters
    ----------
    x : np.ndarray
        Non-negative transformed spend (post-adstock).
    k : float
        Half-saturation point; response is 0.5 when ``x == k``.
    s : float
        Shape; ``s=1`` is hyperbolic, ``s>1`` is S-shaped.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if s <= 0:
        raise ValueError("s must be positive")
    x = np.asarray(x, dtype=float)
    xs = np.power(np.clip(x, 0, None), s)
    return xs / (np.power(k, s) + xs)


def log_saturation(x: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Log-concave response ``log1p(x / k)``; never saturates fully but slope drops."""
    if k <= 0:
        raise ValueError("k must be positive")
    x = np.asarray(x, dtype=float)
    return np.log1p(np.clip(x, 0, None) / k)
