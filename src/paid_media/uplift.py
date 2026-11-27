"""Shared helpers for uplift and targeting case studies.

Ridge-penalized logistic regression fit by Newton's method (numpy only), and a
top-share mask for ranking users by a score. Used by case 07 (Criteo) and
case 08 (Hillstrom).
"""

from __future__ import annotations

import numpy as np


def fit_logistic(
    x: np.ndarray, y: np.ndarray, l2: float = 1e-4, max_iter: int = 25, tol: float = 1e-7
) -> np.ndarray:
    """Ridge-penalized logistic regression by Newton's method (IRLS).

    ``l2`` is per observation, so the penalty is ``0.5 * l2 * n * ||b||^2``
    (the intercept is penalized too, which is harmless at large ``n``).
    """
    n, k = x.shape
    b = np.zeros(k)
    pen = l2 * n
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ b, -30, 30)))
        w = p * (1 - p)
        hess = (x * w[:, None]).T @ x + pen * np.eye(k)
        step = np.linalg.solve(hess, x.T @ (y - p) - pen * b)
        b += step
        if np.max(np.abs(step)) < tol:
            break
    return b


def predict(b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x @ b, -30, 30)))


def top_mask(score: np.ndarray, reach: float) -> np.ndarray:
    """Boolean mask of the top ``reach`` share of ``score``."""
    if not 0 < reach <= 1:
        raise ValueError("reach must be in (0, 1]")
    k = int(round(reach * len(score)))
    mask = np.zeros(len(score), dtype=bool)
    if k == len(score):
        mask[:] = True
    elif k > 0:
        mask[np.argpartition(-score, k - 1)[:k]] = True
    return mask
