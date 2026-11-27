import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CASE = _ROOT / "case-studies" / "08-hillstrom-email-uplift" / "src"
sys.path.insert(0, str(_CASE))

from hillstrom_uplift import (  # noqa: E402
    COLUMNS,
    _ratio,
    design_matrix,
    fetch,
    marginal_ratio,
    synthetic_sample,
    targeting_stats,
)


def _data(n=60_000, seed=0):
    rng = np.random.default_rng(seed)
    df = synthetic_sample(n, rng)
    t = df["segment"].ne("No E-Mail").to_numpy()
    y = df["visit"].to_numpy(float)
    boot = rng.poisson(1.0, size=(100, n)).astype(np.float32)
    return df, y, t, boot, rng


def test_synthetic_sample_has_real_schema():
    df = synthetic_sample(500, np.random.default_rng(0))
    assert tuple(df.columns) == COLUMNS
    assert set(df["segment"]) <= {"Mens E-Mail", "Womens E-Mail", "No E-Mail"}


def test_design_matrix_shape():
    x = design_matrix(synthetic_sample(500, np.random.default_rng(1)))
    assert x.shape == (500, 10)
    assert np.allclose(x[:, 0], 1.0)


def test_random_targeting_reaches_its_share_and_matches_overall_uplift():
    _, y, t, boot, rng = _data()
    res = targeting_stats(rng.random(len(y)), y, t, (0.2, 1.0), boot)
    overall = 1000 * (y[t].mean() - y[~t].mean())
    assert res[1.0]["uplift"].value == pytest.approx(overall)
    assert res[1.0]["share"].value == pytest.approx(1.0)
    s = res[0.2]["share"]
    assert abs(s.value - 0.2) < 4 * s.se


def test_bootstrap_se_matches_analytic_for_overall_uplift():
    _, y, t, boot, rng = _data()
    se = targeting_stats(rng.random(len(y)), y, t, (1.0,), boot)[1.0]["uplift"].se
    r1, r0 = y[t].mean(), y[~t].mean()
    analytic = 1000 * np.sqrt(r1 * (1 - r1) / t.sum() + r0 * (1 - r0) / (~t).sum())
    assert se == pytest.approx(analytic, rel=0.25)


def test_propensity_ranking_concentrates_uplift_in_synthetic_data():
    # Constant log-odds lift: absolute uplift rises with baseline propensity,
    # which falls with recency.
    df, y, t, boot, _ = _data()
    score = -df["recency"].to_numpy(float) + 1e-6 * np.arange(len(y))
    m = marginal_ratio(score, y, t, 0.1, 0.4, boot)
    assert m["average"].value > m["marginal"].value


def test_ratio_not_estimable_with_noisy_denominator():
    rng = np.random.default_rng(3)
    den = np.concatenate([[0.1], rng.normal(0.1, 0.2, 200)])
    assert not _ratio(np.ones(201), den).estimable
    den = np.concatenate([[1.0], rng.normal(1.0, 0.05, 200)])
    assert _ratio(np.ones(201), den).estimable


def test_fetch_rejects_wrong_hash(tmp_path):
    bad = tmp_path / "hillstrom.csv"
    bad.write_text("not the dataset")
    with pytest.raises(ValueError, match="SHA-256"):
        fetch(bad)


def test_run_smoke():
    out = subprocess.run([sys.executable, str(_CASE / "run.py"), "--smoke"],
                         capture_output=True, text=True, check=True)
    assert "wrote" in out.stdout
