"""Case 04 — Media mix modeling (numpy / scipy MAP reference implementation).

A lightweight MAP point estimate for a channel-level adstock + Hill saturation
regression:

$$y_t = \\alpha + \\beta_t \\text{trend}_t + \\gamma_t \\text{season}_t + \\sum_c \\beta_c\\, \\text{Hill}(\\text{Adstock}(x_{c,t}; \\lambda_c); k_c, s_c) + \\varepsilon_t.$$

We fit $\\lambda_c$, $k_c$, and $\\beta_c$ per channel by minimising squared
error under simple priors.  No posterior uncertainty — for that see
:mod:`mmm_bayes` (PyMC, optional).

The purpose of shipping both is to show:

1. The MAP answer recovers the ground-truth ROAS curves in this DGP, so the
   extra machinery in PyMC is not compulsory for a reasonable first read.
2. Where the MAP answer is brittle (adstock/saturation trade off against each
   other, and a single point estimate hides that), the Bayesian posterior
   makes the identification problem legible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from paid_media.adstock import geometric_adstock
from paid_media.saturation import hill_saturation


@dataclass
class MMMFit:
    channels: list[str]
    decay: dict[str, float]
    halfsat: dict[str, float]
    beta: dict[str, float]
    intercept: float
    trend_coef: float
    season_coef: float
    r2: float

    def predict(
        self,
        spend: pd.DataFrame,
        trend: np.ndarray,
        season: np.ndarray,
    ) -> np.ndarray:
        out = self.intercept + self.trend_coef * trend + self.season_coef * season
        for ch in self.channels:
            x = spend[ch].to_numpy()
            adstk = geometric_adstock(x, self.decay[ch])
            sat = hill_saturation(adstk, self.halfsat[ch], s=1.4)
            out = out + self.beta[ch] * sat * spend[ch].mean()
        return out

    def roas_curve(self, ch: str, mean_spend: float, n_points: int = 60) -> pd.DataFrame:
        """Marginal-spend response curve for a channel."""
        grid = np.linspace(0.2 * mean_spend, 2.5 * mean_spend, n_points)
        x = np.full(52, grid[0])  # steady-state at each grid point
        rows = []
        for s in grid:
            x = np.full(104, s)
            adstk = geometric_adstock(x, self.decay[ch])
            sat = hill_saturation(adstk, self.halfsat[ch], s=1.4)
            steady = self.beta[ch] * sat[-1] * mean_spend
            rows.append({"spend": s, "expected_outcome_per_week": steady})
        return pd.DataFrame(rows)


def fit_mmm_map(
    data: pd.DataFrame,
    channels: list[str],
    outcome: str = "trials",
) -> MMMFit:
    """Minimise squared error over (decay_c, halfsat_c, beta_c) per channel + base terms.

    The per-channel parameters are stacked into one vector; we use Nelder-Mead
    for robustness (the loss surface is not smooth in ``halfsat``).
    """
    trend = data["trend"].to_numpy() if "trend" in data else np.arange(len(data), dtype=float)
    season = data["season"].to_numpy() if "season" in data else np.zeros(len(data))
    y = data[outcome].to_numpy()
    spend_means = {ch: float(data[ch].mean()) for ch in channels}

    # Parameter layout: [intercept, trend_coef, season_coef,
    #                    decay_c0, k_c0, beta_c0, decay_c1, k_c1, beta_c1, ...]
    def unpack(theta: np.ndarray) -> tuple[float, float, float, dict, dict, dict]:
        a, bt, bs = theta[:3]
        decay, k, beta = {}, {}, {}
        for i, ch in enumerate(channels):
            d = float(_sigmoid(theta[3 + 3 * i]) * 0.95)  # in (0, 0.95)
            kk = float(np.exp(theta[3 + 3 * i + 1]))      # > 0
            bb = float(theta[3 + 3 * i + 2])
            decay[ch] = d
            k[ch] = kk
            beta[ch] = bb
        return float(a), float(bt), float(bs), decay, k, beta

    def predict_vec(theta: np.ndarray) -> np.ndarray:
        a, bt, bs, decay, k, beta = unpack(theta)
        mu = a + bt * trend + bs * season
        for ch in channels:
            x = data[ch].to_numpy()
            adstk = geometric_adstock(x, decay[ch])
            sat = hill_saturation(adstk, k[ch], s=1.4)
            mu = mu + beta[ch] * sat * spend_means[ch]
        return mu

    def loss(theta: np.ndarray) -> float:
        mu = predict_vec(theta)
        resid = y - mu
        # Mild L2 on beta to discourage adstock/saturation from swapping roles.
        a, bt, bs, decay, k, beta = unpack(theta)
        reg = 0.01 * sum(b * b for b in beta.values())
        return float(np.dot(resid, resid) + reg)

    # Reasonable initialisation: decay logit ~ 0 (-> 0.45), k log ~ log(mean_spend), beta ~ 0.02
    theta0 = [float(y.mean()), 0.0, 0.0]
    for ch in channels:
        theta0 += [0.0, float(np.log(spend_means[ch])), 0.02]
    theta0 = np.array(theta0)

    res = minimize(loss, theta0, method="Nelder-Mead", options={"maxiter": 20_000, "xatol": 1e-5})
    a, bt, bs, decay, k, beta = unpack(res.x)

    mu_hat = predict_vec(res.x)
    ss_res = float(np.sum((y - mu_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return MMMFit(
        channels=channels,
        decay=decay,
        halfsat=k,
        beta=beta,
        intercept=a,
        trend_coef=bt,
        season_coef=bs,
        r2=r2,
    )


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))
