import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "case-studies" / "06-ai-mediated-measurement" / "src"))

from ai_measurement import (  # noqa: E402
    Z_95,
    audience_randomized_creative_test,
    average_and_marginal_uplift,
    budget_split_read,
    calibrate,
    conflict_flag,
    fit_mmm,
    ghost_ads_read,
    lift_window_beta,
    platform_creative_test,
    simulate_creative_population,
    simulate_paced_panel,
    simulate_users,
    targeting_score,
    top_share,
)

# -- targeting ---------------------------------------------------------------


def test_top_share_selects_exact_count_within_eligible():
    rng = np.random.default_rng(0)
    score = rng.standard_normal(1000)
    eligible = rng.random(1000) < 0.5
    mask = top_share(score, 0.2, eligible=eligible)
    assert mask.sum() == round(0.2 * eligible.sum())
    assert not (mask & ~eligible).any()
    assert score[mask].min() >= score[eligible & ~mask].max()


def test_platform_credit_inflation_grows_with_sharpness():
    users = simulate_users(400_000, np.random.default_rng(1))
    ratios = []
    for s in (0.0, 0.8):
        sel = top_share(targeting_score(users, s), 0.2)
        ratios.append(users.p1[sel].sum() / users.uplift[sel].sum())
    assert ratios[0] > 2.0  # baseline converters get credited even with random targeting
    assert ratios[1] > ratios[0] + 0.5


def test_ghost_ads_lift_is_unbiased_and_covers():
    rng = np.random.default_rng(2)
    errors, covered = [], []
    for _ in range(40):
        users = simulate_users(100_000, rng)
        sel = top_share(targeting_score(users, 0.8), 0.2)
        r = ghost_ads_read(users, sel, rng)
        errors.append((r.lift - r.truth) / r.lift_se)
        covered.append(abs(r.lift - r.truth) <= Z_95 * r.lift_se)
    assert abs(np.mean(errors)) < 0.5
    assert np.mean(covered) >= 0.85


def test_marginal_equals_average_only_for_random_targeting():
    users = simulate_users(1_000_000, np.random.default_rng(3))
    avg0, marg0 = average_and_marginal_uplift(users, 0.0, 0.1, 0.4)
    avg8, marg8 = average_and_marginal_uplift(users, 0.8, 0.1, 0.4)
    assert marg0 == pytest.approx(avg0, rel=0.03)
    assert marg8 < 0.9 * avg8


def test_budget_split_recovers_marginal_uplift():
    rng = np.random.default_rng(4)
    errors = []
    for _ in range(10):
        users = simulate_users(1_000_000, rng)
        r = budget_split_read(users, 0.8, 0.1, 0.4, rng)
        errors.append((r.lift_marginal - r.true_marginal) / r.lift_marginal_se)
        assert r.platform_marginal > 2 * r.true_marginal
    assert abs(np.mean(errors)) < 1.0


# -- divergent delivery ------------------------------------------------------


def test_platform_ab_picks_a_while_incrementality_favours_b():
    rng = np.random.default_rng(5)
    pop = simulate_creative_population(400_000, rng)
    platform, lift = platform_creative_test(pop, 0.3, 2.0, rng)
    assert platform.truth == 0.0
    assert platform.pick() == "A"
    assert lift.truth < 0
    assert lift.pick() == "B"
    assert abs(lift.diff - lift.truth) < 4 * lift.se


def test_audience_randomized_test_holds_false_positive_rate():
    rng = np.random.default_rng(6)
    picks = [
        audience_randomized_creative_test(
            simulate_creative_population(100_000, rng), 0.3, rng
        ).pick()
        for _ in range(100)
    ]
    assert np.mean([p != "none" for p in picks]) < 0.12


# -- MMM ---------------------------------------------------------------------


def _mmm_means(corr, reps, seed):
    rng = np.random.default_rng(seed)
    b_m, b_l, flags = [], [], []
    beta = None
    for _ in range(reps):
        panel = simulate_paced_panel(corr, rng)
        beta = panel.beta
        m, sm = fit_mmm(panel)
        lift, sl = lift_window_beta(panel, rng)
        b_m.append(m)
        b_l.append(lift)
        flags.append(conflict_flag(m, sm, lift, sl))
    return np.mean(b_m), np.mean(b_l), np.mean(flags), beta


def test_mmm_unbiased_without_pacing_and_biased_with_it():
    m0, l0, f0, beta = _mmm_means(0.0, 150, 7)
    m9, l9, f9, _ = _mmm_means(0.9, 150, 8)
    assert m0 == pytest.approx(beta, rel=0.1)
    assert m9 > 1.8 * beta
    assert l0 == pytest.approx(beta, rel=0.1)
    assert l9 == pytest.approx(beta, rel=0.1)
    assert f0 < 0.15
    assert f9 > 0.9


def test_calibrate_prior_is_between_inputs():
    out = calibrate(2.0, 0.5, 1.0, 0.5)
    assert out["uncalibrated"] == (2.0, 0.5)
    assert out["lift_only"] == (1.0, 0.5)
    assert out["prior"][0] == pytest.approx(1.5)
    assert out["prior"][1] == pytest.approx(0.5 / np.sqrt(2))


def test_validation():
    with pytest.raises(ValueError):
        top_share(np.zeros(10), 0.0)
    with pytest.raises(ValueError):
        targeting_score(simulate_users(10, np.random.default_rng(0)), 1.5)
    with pytest.raises(ValueError):
        simulate_paced_panel(1.0, np.random.default_rng(0))
