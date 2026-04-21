"""Case 01 — geo lift via synthetic control.

Estimate the causal effect of a paid-media brand campaign run in a single
target DMA, using a weighted blend of untreated donor DMAs as the counterfactual.

This module is deliberately self-contained (no import from causal-inference-
playbook) so the playbook reads end-to-end.  For the full treatment of synthetic
control (placebo inference, block bootstrap CIs, California Prop 99 replication),
see that sibling repo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class GeoLiftResult:
    att: float                        # average treatment effect on the treated
    att_pct: float                    # as a % of counterfactual mean
    weights: pd.Series                # donor DMA -> weight in (0, 1)
    y_treated: np.ndarray
    y_synth: np.ndarray
    pre_rmspe: float
    post_rmspe: float

    @property
    def rmspe_ratio(self) -> float:
        return self.post_rmspe / max(self.pre_rmspe, 1e-9)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        tops = self.weights.sort_values(ascending=False).head(3)
        return (
            f"GeoLift(ATT={self.att:+.2f} ({self.att_pct:+.1%}), "
            f"RMSPE pre={self.pre_rmspe:.2f} post={self.post_rmspe:.2f} "
            f"ratio={self.rmspe_ratio:.1f}, "
            f"top donors: {', '.join(f'{i}={v:.2f}' for i, v in tops.items())})"
        )


def fit_geo_lift(
    panel: pd.DataFrame,
    treated_dma: str,
    treatment_week: int,
    outcome: str = "trials",
) -> GeoLiftResult:
    """Fit synthetic control for a single treated DMA.

    Parameters
    ----------
    panel : pd.DataFrame
        Long panel with columns ``dma``, ``week``, and the outcome.
    treated_dma : str
        DMA that received the campaign.
    treatment_week : int
        First week of treatment (inclusive).
    outcome : str
        Outcome column.
    """
    wide = panel.pivot(index="week", columns="dma", values=outcome).sort_index()
    weeks = wide.index.to_numpy()
    pre_mask = weeks < treatment_week

    y = wide[treated_dma].to_numpy()
    donors = [c for c in wide.columns if c != treated_dma]
    X = wide[donors].to_numpy()

    y_pre = y[pre_mask]
    X_pre = X[pre_mask]

    n = len(donors)

    def loss(w: np.ndarray) -> float:
        return float(np.sum((y_pre - X_pre @ w) ** 2))

    w0 = np.full(n, 1.0 / n)
    cons = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
    bounds = [(0.0, 1.0)] * n
    res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x, 0, 1)
    w = w / w.sum()

    y_synth = X @ w
    gap = y - y_synth
    pre_rmspe = float(np.sqrt(np.mean(gap[pre_mask] ** 2)))
    post_rmspe = float(np.sqrt(np.mean(gap[~pre_mask] ** 2)))
    att = float(np.mean(gap[~pre_mask]))
    counter = float(np.mean(y_synth[~pre_mask]))
    att_pct = att / counter if counter > 0 else float("nan")

    return GeoLiftResult(
        att=att,
        att_pct=att_pct,
        weights=pd.Series(w, index=donors).sort_values(ascending=False),
        y_treated=y,
        y_synth=y_synth,
        pre_rmspe=pre_rmspe,
        post_rmspe=post_rmspe,
    )


def placebo_p_value(
    panel: pd.DataFrame,
    treated_dma: str,
    treatment_week: int,
    outcome: str = "trials",
    pre_rmspe_cap: float = 20.0,
) -> float:
    """Permutation p-value from placebo-in-space (Abadie et al. 2010).

    For each donor DMA, re-fit SC pretending it was treated.  Rank the true
    RMSPE ratio against the placebo distribution.  Donors with pre-RMSPE more
    than ``pre_rmspe_cap`` times the treated's are dropped (standard practice).
    """
    true = fit_geo_lift(panel, treated_dma, treatment_week, outcome=outcome)
    ratios = [true.rmspe_ratio]
    donors = [d for d in panel["dma"].unique() if d != treated_dma]
    for d in donors:
        try:
            placebo = fit_geo_lift(panel, d, treatment_week, outcome=outcome)
        except Exception:  # pragma: no cover - QP failure on pathological donor
            continue
        if placebo.pre_rmspe > pre_rmspe_cap * true.pre_rmspe:
            continue
        ratios.append(placebo.rmspe_ratio)
    ratios_arr = np.array(ratios)
    rank = (ratios_arr >= true.rmspe_ratio).sum()
    return float(rank / len(ratios_arr))


def power_analysis(
    base_std: float,
    n_pre: int,
    n_post: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Minimum detectable ATT given counterfactual-noise RMSE and design length.

    Approximation: treats the post-period mean gap as normal with SD
    ``base_std / sqrt(n_post)``; ignores finite-donor variance (usually small).
    """
    from scipy.stats import norm

    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    se = base_std / np.sqrt(n_post)
    return float((z_a + z_b) * se)
