"""Case 03 — Conversion lift / incrementality for low base-rate paid media.

A paid-media conversion-lift test exposes a holdout (``control``) and an
exposed (``treated``) arm to matched user populations.  The exposed arm sees
the real ad; the control sees a placebo / ghost ad / PSA.  The difference in
conversion rates, scaled, is the **incremental lift**.

The binding constraint on conversion-lift tests in practice is **statistical
power** at low base rates.  Trial-start rates in the 0.3–0.8% range mean even a
15% relative lift needs ~200k users per arm to detect at 80% power.  This
module makes the power calc the first-class citizen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta, norm


@dataclass
class LiftResult:
    rate_control: float
    rate_treated: float
    abs_lift: float
    rel_lift: float
    ci_rel: tuple[float, float]
    p_value: float
    n_per_arm: int

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        lo, hi = self.ci_rel
        return (
            f"Lift(rel={self.rel_lift:+.2%} [{lo:+.2%}, {hi:+.2%}], "
            f"abs={self.abs_lift:+.4f}, p={self.p_value:.4f}, n={self.n_per_arm:,}/arm)"
        )


def two_prop_z_test(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """Two-proportion pooled z-test.  Returns (z, two-sided p)."""
    p1 = x1 / n1 if n1 else 0.0
    p2 = x2 / n2 if n2 else 0.0
    p = (x1 + x2) / (n1 + n2) if (n1 + n2) else 0.0
    se = np.sqrt(p * (1 - p) * (1 / max(n1, 1) + 1 / max(n2, 1)))
    if se == 0:
        return 0.0, 1.0
    z = (p2 - p1) / se
    p_two = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p_two)


def relative_lift_ci(
    x_ctrl: int,
    n_ctrl: int,
    x_trt: int,
    n_trt: int,
    alpha: float = 0.05,
    n_boot: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """Parametric-bootstrap CI on the relative lift ``p_trt / p_ctrl - 1``.

    Samples ``p_ctrl`` and ``p_trt`` from their Beta(α+x, β+n-x) posteriors and
    takes the quantiles of the implied relative lift.  Avoids the delta-method
    breakdown at low base rates.
    """
    rng = np.random.default_rng(seed)
    pc = beta.rvs(1 + x_ctrl, 1 + n_ctrl - x_ctrl, size=n_boot, random_state=rng)
    pt = beta.rvs(1 + x_trt, 1 + n_trt - x_trt, size=n_boot, random_state=rng)
    rel = pt / np.maximum(pc, 1e-12) - 1.0
    lo, hi = np.quantile(rel, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def analyze_lift(control: np.ndarray, treated: np.ndarray, alpha: float = 0.05) -> LiftResult:
    """Headline analysis: rates, lift, z-test p-value, bootstrap relative-lift CI."""
    n_c, n_t = len(control), len(treated)
    x_c, x_t = int(control.sum()), int(treated.sum())
    p_c = x_c / n_c if n_c else 0.0
    p_t = x_t / n_t if n_t else 0.0
    abs_lift = p_t - p_c
    rel = (p_t - p_c) / p_c if p_c > 0 else float("nan")
    _, p = two_prop_z_test(x_c, n_c, x_t, n_t)
    lo, hi = relative_lift_ci(x_c, n_c, x_t, n_t, alpha=alpha)
    return LiftResult(
        rate_control=p_c,
        rate_treated=p_t,
        abs_lift=abs_lift,
        rel_lift=rel,
        ci_rel=(lo, hi),
        p_value=p,
        n_per_arm=min(n_c, n_t),
    )


def required_sample_size(
    base_rate: float,
    rel_lift: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Per-arm n for a two-proportion test given base rate and target relative lift.

    Formula: standard two-proportion z-test sample size with pooled variance.
    """
    if base_rate <= 0 or base_rate >= 1:
        raise ValueError("base_rate must be in (0,1)")
    p1 = base_rate
    p2 = base_rate * (1 + rel_lift)
    if p2 <= 0 or p2 >= 1:
        raise ValueError("target p2 out of (0,1); implausible lift")
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    p_bar = (p1 + p2) / 2
    term_a = z_a * np.sqrt(2 * p_bar * (1 - p_bar))
    term_b = z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    num = (term_a + term_b) ** 2
    den = (p2 - p1) ** 2
    return int(np.ceil(num / den))
