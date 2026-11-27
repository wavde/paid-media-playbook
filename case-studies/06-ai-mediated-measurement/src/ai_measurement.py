"""Measurement when targeting, creative delivery and spend pacing are model-driven.

Three mechanisms. Each has a known answer because the simulator knows every
user's baseline and treated conversion probability.

1. **Algorithmic targeting.** The platform ranks users by a noisy read of
   *baseline* conversion propensity (``sharpness`` = correlation with the true
   driver) and serves the top share ``q``. Platform credit counts every
   conversion among served users. A ghost-ads holdout among the users the
   platform *would* serve estimates true incrementality for the policy as
   deployed. Sharper targeting makes the average incremental conversion per
   served user diverge from the marginal one, so a lift test run at one
   budget does not transport to another.
2. **Divergent delivery** (Braun & Schwartz 2023). Two creatives have
   *identical* per-person effects. The platform delivers each one to a
   different audience, so an in-platform "A/B test" compares audiences, not
   creatives.
3. **Algorithm-paced spend.** The platform raises spend when latent demand is
   high. An MMM that cannot see demand attributes demand to media. A
   user-level holdout run for a few weeks gives an unbiased coefficient for
   calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paid_media.adstock import geometric_adstock
from paid_media.saturation import hill_saturation

Z_95 = 1.959963984540054


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------
# 1. Algorithmic targeting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Users:
    """Per-user baseline (``p0``) and treated (``p1``) conversion probabilities."""

    z: np.ndarray
    noise: np.ndarray
    p0: np.ndarray
    p1: np.ndarray

    @property
    def uplift(self) -> np.ndarray:
        return self.p1 - self.p0


def simulate_users(
    n: int,
    rng: np.random.Generator,
    base_logit: float = -3.5,
    effect0: float = 0.7,
    effect_slope: float = 0.2,
    effect_sd: float = 0.2,
) -> Users:
    """Users whose baseline intent ``z`` raises conversion but slightly lowers lift.

    Log-odds lift is ``max(0, effect0 - effect_slope * z + N(0, effect_sd))``.
    High-intent users convert most often, with or without the ad.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    z = rng.standard_normal(n)
    effect = np.clip(effect0 - effect_slope * z + effect_sd * rng.standard_normal(n), 0.0, None)
    eta = base_logit + z
    return Users(z=z, noise=rng.standard_normal(n), p0=sigmoid(eta), p1=sigmoid(eta + effect))


def targeting_score(users: Users, sharpness: float) -> np.ndarray:
    """Platform score with correlation ``sharpness`` to the baseline driver ``z``."""
    if not 0.0 <= sharpness <= 1.0:
        raise ValueError("sharpness must be in [0, 1]")
    return sharpness * users.z + np.sqrt(1.0 - sharpness**2) * users.noise


def top_share(score: np.ndarray, q: float, eligible: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask of the top ``q`` share of ``score`` (within ``eligible``)."""
    if not 0.0 < q <= 1.0:
        raise ValueError("q must be in (0, 1]")
    pool = np.arange(len(score)) if eligible is None else np.flatnonzero(eligible)
    k = int(round(q * len(pool)))
    mask = np.zeros(len(score), dtype=bool)
    if k == 0:
        return mask
    if k == len(pool):
        mask[pool] = True
        return mask
    idx = np.argpartition(-score[pool], k - 1)[:k]
    mask[pool[idx]] = True
    return mask


@dataclass(frozen=True)
class LiftRead:
    """One ghost-ads read on the users a policy selects.

    ``credited`` is the number of conversions the platform would credit if it
    served every selected user. ``lift`` / ``lift_se`` estimate incremental
    conversions over all selected users. ``truth`` is the exact expected
    incremental conversions, ``sum(p1 - p0)``.
    """

    n_selected: int
    credited: float
    lift: float
    lift_se: float
    truth: float


def ghost_ads_read(
    users: Users,
    selected: np.ndarray,
    rng: np.random.Generator,
    holdout_share: float = 0.5,
) -> LiftRead:
    """Randomize a holdout among selected users and read incremental conversions."""
    if not 0.0 < holdout_share < 1.0:
        raise ValueError("holdout_share must be in (0, 1)")
    p0, p1 = users.p0[selected], users.p1[selected]
    n = len(p0)
    if n < 4:
        raise ValueError("need at least 4 selected users")
    u = rng.random(n)
    y0, y1 = u < p0, u < p1
    held = rng.random(n) < holdout_share
    m_t, m_h = float(y1[~held].mean()), float(y0[held].mean())
    se = np.sqrt(m_t * (1 - m_t) / (~held).sum() + m_h * (1 - m_h) / held.sum())
    return LiftRead(
        n_selected=n,
        credited=float(y1.sum()),
        lift=(m_t - m_h) * n,
        lift_se=float(se * n),
        truth=float((p1 - p0).sum()),
    )


def average_and_marginal_uplift(
    users: Users, sharpness: float, q_low: float, q_high: float
) -> tuple[float, float]:
    """Exact average uplift per served user at ``q_low`` and marginal uplift per
    extra served user when moving from ``q_low`` to ``q_high``."""
    if not q_low < q_high:
        raise ValueError("need q_low < q_high")
    score = targeting_score(users, sharpness)
    low, high = top_share(score, q_low), top_share(score, q_high)
    u = users.uplift
    extra = high & ~low
    return float(u[low].mean()), float(u[extra].mean())


@dataclass(frozen=True)
class BudgetSplitRead:
    """Two-cell budget-split test: random user cells run ``q_low`` vs ``q_high``.

    All rates are per served user. ``platform_*`` use credited conversions.
    ``lift_*`` use ghost-ads holdouts within each cell.
    """

    platform_avg_low: float
    platform_marginal: float
    lift_avg_low: float
    lift_avg_high: float
    lift_marginal: float
    lift_marginal_se: float
    true_avg_low: float
    true_marginal: float


def budget_split_read(
    users: Users,
    sharpness: float,
    q_low: float,
    q_high: float,
    rng: np.random.Generator,
    holdout_share: float = 0.5,
) -> BudgetSplitRead:
    """Run both budget levels in randomized halves of the user base."""
    score = targeting_score(users, sharpness)
    cell_high = rng.random(len(score)) < 0.5
    sel_low = top_share(score, q_low, eligible=~cell_high)
    sel_high = top_share(score, q_high, eligible=cell_high)
    r_low = ghost_ads_read(users, sel_low, rng, holdout_share)
    r_high = ghost_ads_read(users, sel_high, rng, holdout_share)
    n_low, n_high = int((~cell_high).sum()), int(cell_high.sum())
    dq = q_high - q_low
    marginal = (r_high.lift / n_high - r_low.lift / n_low) / dq
    marginal_se = np.hypot(r_high.lift_se / n_high, r_low.lift_se / n_low) / dq
    true_avg, true_marg = average_and_marginal_uplift(users, sharpness, q_low, q_high)
    return BudgetSplitRead(
        platform_avg_low=r_low.credited / r_low.n_selected,
        platform_marginal=(r_high.credited / n_high - r_low.credited / n_low) / dq,
        lift_avg_low=r_low.lift / r_low.n_selected,
        lift_avg_high=r_high.lift / r_high.n_selected,
        lift_marginal=float(marginal),
        lift_marginal_se=float(marginal_se),
        true_avg_low=true_avg,
        true_marginal=true_marg,
    )


# --------------------------------------------------------------------------
# 2. Divergent delivery in creative tests
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CreativePopulation:
    """Two segments. The creative does not change a person's conversion odds:
    ``p1`` is the same whichever creative they see."""

    segment: np.ndarray
    z: np.ndarray
    p0: np.ndarray
    p1: np.ndarray

    @property
    def uplift(self) -> np.ndarray:
        return self.p1 - self.p0


def simulate_creative_population(
    n: int,
    rng: np.random.Generator,
    base_logit: float = -3.5,
    segment_shift: float = 1.0,
    effect_seg0: float = 0.6,
    effect_seg1: float = 0.15,
) -> CreativePopulation:
    """Segment 1 converts more at baseline but responds less to ads."""
    seg = rng.random(n) < 0.5
    z = rng.standard_normal(n)
    eta = base_logit + z + segment_shift * seg
    effect = np.where(seg, effect_seg1, effect_seg0)
    return CreativePopulation(segment=seg, z=z, p0=sigmoid(eta), p1=sigmoid(eta + effect))


def delivery_score(
    pop: CreativePopulation,
    affinity: float,
    rng: np.random.Generator,
    z_weight: float = 0.5,
    noise_sd: float = 0.5,
) -> np.ndarray:
    """Platform delivery score for one creative.

    ``affinity > 0`` means the platform's model learned that this creative
    "works" in segment 1, ``< 0`` in segment 0, and ``0`` is creative-neutral.
    """
    return (
        z_weight * pop.z
        + affinity * pop.segment
        + noise_sd * rng.standard_normal(len(pop.z))
    )


@dataclass(frozen=True)
class CreativeTestRead:
    """Estimated A minus B difference and its SE under one design, plus the truth."""

    diff: float
    se: float
    truth: float

    @property
    def z(self) -> float:
        return self.diff / self.se if self.se > 0 else 0.0

    def pick(self) -> str:
        if self.z > Z_95:
            return "A"
        if self.z < -Z_95:
            return "B"
        return "none"


def _conversions(
    pop: CreativePopulation, idx: np.ndarray, treated: bool, rng: np.random.Generator
) -> np.ndarray:
    p = pop.p1[idx] if treated else pop.p0[idx]
    return rng.random(len(idx)) < p


def platform_creative_test(
    pop: CreativePopulation,
    q: float,
    affinity: float,
    rng: np.random.Generator,
    holdout_share: float = 0.5,
) -> tuple[CreativeTestRead, CreativeTestRead]:
    """In-platform split test: users are randomized to the A or B campaign,
    and within each arm the platform serves its top-``q`` users for that
    creative.

    Returns two reads:
    - **Platform CVR**: conversion rate among served users (what the platform
      reports). Truth is 0 for the creative-effect estimand.
    - **Per-arm ghost-ads lift**: incremental conversions per served user for
      each campaign as deployed. Truth is the exact difference in mean uplift
      between the two delivered audiences.
    """
    arm_a = rng.random(len(pop.z)) < 0.5
    cvr, cvr_var, inc, inc_var, truth_inc = [], [], [], [], []
    for in_arm, aff in ((arm_a, affinity), (~arm_a, -affinity)):
        score = delivery_score(pop, aff, rng)
        served = np.flatnonzero(top_share(score, q, eligible=in_arm))
        held = rng.random(len(served)) < holdout_share
        y_t = _conversions(pop, served[~held], True, rng)
        y_h = _conversions(pop, served[held], False, rng)
        m_t, m_h = float(y_t.mean()), float(y_h.mean())
        cvr.append(m_t)
        cvr_var.append(m_t * (1 - m_t) / len(y_t))
        inc.append(m_t - m_h)
        inc_var.append(m_t * (1 - m_t) / len(y_t) + m_h * (1 - m_h) / len(y_h))
        truth_inc.append(float(pop.uplift[served].mean()))
    platform = CreativeTestRead(
        diff=cvr[0] - cvr[1], se=float(np.sqrt(cvr_var[0] + cvr_var[1])), truth=0.0
    )
    lift = CreativeTestRead(
        diff=inc[0] - inc[1],
        se=float(np.sqrt(inc_var[0] + inc_var[1])),
        truth=truth_inc[0] - truth_inc[1],
    )
    return platform, lift


def audience_randomized_creative_test(
    pop: CreativePopulation, q: float, rng: np.random.Generator
) -> CreativeTestRead:
    """Fix one creative-neutral audience first, then randomize each user to A or B
    with forced delivery. The estimand is the creative effect, which is 0 by
    construction."""
    score = delivery_score(pop, 0.0, rng)
    audience = np.flatnonzero(top_share(score, q))
    to_a = rng.random(len(audience)) < 0.5
    y_a = _conversions(pop, audience[to_a], True, rng)
    y_b = _conversions(pop, audience[~to_a], True, rng)
    m_a, m_b = float(y_a.mean()), float(y_b.mean())
    se = np.sqrt(m_a * (1 - m_a) / len(y_a) + m_b * (1 - m_b) / len(y_b))
    return CreativeTestRead(diff=m_a - m_b, se=float(se), truth=0.0)


# --------------------------------------------------------------------------
# 3. Algorithm-paced spend and MMM
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PacedPanel:
    """Weekly MMM panel. Media transforms are known exactly (the MMM's best case);
    only the latent demand series is hidden from the model."""

    spend: np.ndarray
    media: np.ndarray
    trend: np.ndarray
    season: np.ndarray
    y: np.ndarray
    beta: float
    test_media: np.ndarray
    test_base: np.ndarray

    @property
    def media_per_dollar(self) -> float:
        return float(self.media.sum() / self.spend.sum())


def simulate_paced_panel(
    demand_corr: float,
    rng: np.random.Generator,
    n_weeks: int = 104,
    test_weeks: int = 8,
    burn_in: int = 20,
    mean_spend: float = 100_000.0,
    spend_sd: float = 0.35,
    decay: float = 0.4,
    demand_ar: float = 0.7,
    demand_scale: float = 60.0,
    noise_sd: float = 50.0,
    beta: float = 0.006,
) -> PacedPanel:
    """Weekly sales with AR(1) latent demand. Log spend has correlation
    ``demand_corr`` with demand (the platform bids up when intent is high).
    The ``test_weeks`` after the modelling window are held back for a
    user-level lift test."""
    if not 0.0 <= demand_corr < 1.0:
        raise ValueError("demand_corr must be in [0, 1)")
    total = burn_in + n_weeks + test_weeks
    shocks = rng.standard_normal(total)
    d = np.empty(total)
    d[0] = shocks[0]
    for t in range(1, total):
        d[t] = demand_ar * d[t - 1] + np.sqrt(1 - demand_ar**2) * shocks[t]
    e = rng.standard_normal(total)
    log_spend = (
        np.log(mean_spend)
        - 0.5 * spend_sd**2
        + spend_sd * (demand_corr * d + np.sqrt(1 - demand_corr**2) * e)
    )
    spend = np.exp(log_spend)
    media = hill_saturation(geometric_adstock(spend, decay), k=mean_spend, s=1.0) * mean_spend
    weeks = np.arange(total - burn_in, dtype=float)
    trend = 4.0 * weeks
    season = np.sin(2 * np.pi * weeks / 52)
    base = 3000.0 + trend + 300.0 * season + demand_scale * d[burn_in:]
    m = media[burn_in:]
    y = base + beta * m + rng.normal(0.0, noise_sd, total - burn_in)
    fit, test = slice(0, n_weeks), slice(n_weeks, n_weeks + test_weeks)
    return PacedPanel(
        spend=spend[burn_in:][fit],
        media=m[fit],
        trend=trend[fit],
        season=season[fit],
        y=y[fit],
        beta=beta,
        test_media=m[test],
        test_base=base[test],
    )


def fit_mmm(panel: PacedPanel, hac_lags: int = 4) -> tuple[float, float]:
    """OLS of sales on trend, season and (known-transform) media.

    Returns the media coefficient and its Newey-West (HAC) standard error.
    """
    x = np.column_stack([np.ones_like(panel.y), panel.trend, panel.season, panel.media])
    n, k = x.shape
    xtx_inv = np.linalg.inv(x.T @ x)
    coef = xtx_inv @ x.T @ panel.y
    resid = panel.y - x @ coef
    xe = x * resid[:, None]
    s = xe.T @ xe
    for lag in range(1, hac_lags + 1):
        w = 1.0 - lag / (hac_lags + 1)
        g = xe[lag:].T @ xe[:-lag]
        s += w * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv * n / (n - k)
    return float(coef[-1]), float(np.sqrt(cov[-1, -1]))


def lift_window_beta(
    panel: PacedPanel, rng: np.random.Generator, holdout_share: float = 0.2
) -> tuple[float, float]:
    """User-level holdout during the test weeks. The holdout's sales scale the
    baseline, so the treated-minus-scaled-holdout gap estimates
    ``beta * media``, whatever demand did."""
    h = holdout_share
    t_obs = rng.poisson((1 - h) * (panel.test_base + panel.beta * panel.test_media))
    h_obs = rng.poisson(h * panel.test_base)
    inc = t_obs / (1 - h) - h_obs / h
    var = t_obs / (1 - h) ** 2 + h_obs / h**2
    denom = float(panel.test_media.sum())
    return float(inc.sum() / denom), float(np.sqrt(var.sum()) / denom)


def calibrate(
    b_mmm: float, se_mmm: float, b_lift: float, se_lift: float
) -> dict[str, tuple[float, float]]:
    """Three ways to combine the MMM and lift-test coefficients.

    - ``uncalibrated``: MMM alone.
    - ``prior``: precision-weighted average, i.e. the lift test used as a
      Gaussian prior.
    - ``lift_only``: replace the MMM coefficient with the lift-test estimate.
    """
    w_m, w_l = 1.0 / se_mmm**2, 1.0 / se_lift**2
    return {
        "uncalibrated": (b_mmm, se_mmm),
        "prior": ((w_m * b_mmm + w_l * b_lift) / (w_m + w_l), float(np.sqrt(1.0 / (w_m + w_l)))),
        "lift_only": (b_lift, se_lift),
    }


def conflict_flag(b_mmm: float, se_mmm: float, b_lift: float, se_lift: float) -> bool:
    """True when the MMM and lift coefficients disagree at the 5% level."""
    return abs(b_mmm - b_lift) > Z_95 * float(np.hypot(se_mmm, se_lift))
