"""Case 04 — Bayesian media mix modeling (PyMC; optional).

Full posterior over (decay, halfsat, beta) per channel + baseline terms.
Requires ``pip install -r requirements-mmm.txt``.

This module is imported lazily by the reproducer so that ``pytest`` on the
main requirements set doesn't need PyMC installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from paid_media.adstock import geometric_adstock  # noqa: F401 (used in doc)
from paid_media.saturation import hill_saturation  # noqa: F401


@dataclass
class MMMBayesFit:
    trace: Any                      # arviz.InferenceData
    channels: list[str]
    summary: pd.DataFrame           # posterior means + 94% HDI per parameter


def fit_mmm_bayes(
    data: pd.DataFrame,
    channels: list[str],
    outcome: str = "trials",
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    random_seed: int = 0,
) -> MMMBayesFit:
    """Fit a Bayesian MMM with PyMC.

    Priors (weakly informative, chosen to be realistic for SaaS paid media):

    - ``decay_c``   ~ Beta(2, 2)                     (carryover in (0, 1))
    - ``halfsat_c`` ~ HalfNormal(mean channel spend) (half-saturation point)
    - ``beta_c``    ~ HalfNormal(0.05)               (non-negative response coef)
    - ``intercept`` ~ Normal(data mean, 1000)
    - ``sigma``     ~ HalfNormal(data std)
    """
    import pymc as pm  # lazy import
    import pytensor.tensor as pt

    weeks = len(data)
    y = data[outcome].to_numpy()
    trend = data["trend"].to_numpy() if "trend" in data else np.arange(weeks, dtype=float)
    season = data["season"].to_numpy() if "season" in data else np.zeros(weeks)
    spend = {ch: data[ch].to_numpy() for ch in channels}
    spend_means = {ch: float(data[ch].mean()) for ch in channels}

    with pm.Model():
        intercept = pm.Normal("intercept", mu=float(y.mean()), sigma=1000.0)
        trend_coef = pm.Normal("trend_coef", mu=0.0, sigma=50.0)
        season_coef = pm.Normal("season_coef", mu=0.0, sigma=50.0)
        sigma = pm.HalfNormal("sigma", sigma=float(y.std()))

        mu = intercept + trend_coef * trend + season_coef * season

        for ch in channels:
            decay = pm.Beta(f"decay_{ch}", alpha=2.0, beta=2.0)
            k = pm.HalfNormal(f"halfsat_{ch}", sigma=spend_means[ch])
            beta = pm.HalfNormal(f"beta_{ch}", sigma=0.05)

            # Adstock via scan (geometric carryover).
            def _adstock_step(x_t, prev, d=decay):  # noqa: B023
                return x_t + d * prev

            adstk, _ = pt.scan(
                fn=_adstock_step,
                sequences=pt.as_tensor_variable(spend[ch]),
                outputs_info=pt.as_tensor_variable(np.float64(0.0)),
            )
            adstk = adstk * (1.0 - decay)
            sat = adstk ** 1.4 / (k ** 1.4 + adstk ** 1.4)
            mu = mu + beta * sat * spend_means[ch]

        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        trace = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=random_seed,
                          progressbar=False, target_accept=0.9)

    import arviz as az  # noqa: WPS433
    summary = az.summary(trace, hdi_prob=0.94)
    return MMMBayesFit(trace=trace, channels=channels, summary=summary)


def posterior_roas(
    fit: MMMBayesFit,
    channels: list[str],
    mean_spend: dict[str, float],
) -> pd.DataFrame:
    """Posterior mean + 94% HDI of steady-state incremental trials per $ spent."""
    import arviz as az  # noqa: WPS433

    post = fit.trace.posterior
    rows = []
    for ch in channels:
        beta = post[f"beta_{ch}"].to_numpy().reshape(-1)
        k = post[f"halfsat_{ch}"].to_numpy().reshape(-1)
        x = mean_spend[ch]
        sat = x ** 1.4 / (k ** 1.4 + x ** 1.4)
        contrib = beta * sat * x
        roas = contrib / x  # trials per $ (steady-state)
        hdi = az.hdi(roas.reshape(1, -1), hdi_prob=0.94).squeeze()
        rows.append(
            {
                "channel": ch,
                "roas_post_mean": float(roas.mean()),
                "roas_hdi_low": float(hdi[0]),
                "roas_hdi_high": float(hdi[1]),
            }
        )
    return pd.DataFrame(rows)
