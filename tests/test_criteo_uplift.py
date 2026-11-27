import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CASE = _ROOT / "case-studies" / "07-criteo-uplift" / "src"
sys.path.insert(0, str(_CASE))

from criteo_uplift import (  # noqa: E402
    FEATURES,
    Standardizer,
    evaluate_targeting,
    fetch,
    fit_logistic,
    marginal_uplift,
    predict,
    synthetic_sample,
    top_mask,
)


def test_fit_logistic_recovers_coefficients():
    rng = np.random.default_rng(0)
    x = np.column_stack([np.ones(200_000), rng.standard_normal((200_000, 2))])
    beta = np.array([-2.0, 0.8, -0.5])
    y = (rng.random(200_000) < predict(beta, x)).astype(float)
    b = fit_logistic(x, y, l2=0.0)
    assert b == pytest.approx(beta, abs=0.05)


def test_top_mask_counts():
    s = np.arange(100.0)
    m = top_mask(s, 0.1)
    assert m.sum() == 10
    assert s[m].min() == 90
    assert top_mask(s, 1.0).all()


def _synthetic_scores(n=400_000, seed=1):
    rng = np.random.default_rng(seed)
    df = synthetic_sample(n, rng)
    x = df[list(FEATURES)].to_numpy(np.float64)
    y = df["visit"].to_numpy(np.float64)
    t = df["treatment"].to_numpy() == 1
    folds = rng.integers(0, 10, size=n)
    return x, y, t, folds, rng


def test_random_targeting_matches_overall_uplift():
    x, y, t, folds, rng = _synthetic_scores()
    overall = 1000 * (y[t].mean() - y[~t].mean())
    reads = evaluate_targeting(rng.random(len(y)), y, t, (0.2, 1.0), folds)
    assert reads[1].uplift == pytest.approx(overall)
    assert reads[1].share_of_total == pytest.approx(1.0)
    assert abs(reads[0].uplift - overall) < 4 * reads[0].uplift_se
    assert reads[0].share_of_total == pytest.approx(0.2, abs=4 * reads[0].share_se)


def test_propensity_targeting_concentrates_uplift_and_inflates_credit():
    # In the synthetic sample the lift is constant in log-odds, so absolute
    # uplift rises with baseline propensity: the true score is f0.
    x, y, t, folds, rng = _synthetic_scores()
    score = x[:, 0]
    top, everyone = evaluate_targeting(score, y, t, (0.1, 1.0), folds)
    assert top.uplift > 2 * everyone.uplift
    assert top.share_of_total > 0.2
    assert top.inflation > everyone.inflation
    avg, marg = marginal_uplift(score, y, t, 0.1, 0.4)
    assert marg < 0.7 * avg
    r_avg, r_marg = marginal_uplift(rng.random(len(y)), y, t, 0.1, 0.4)
    assert r_marg == pytest.approx(r_avg, rel=0.35)


def test_standardizer_design_shape():
    x = np.random.default_rng(2).standard_normal((50, 12))
    d = Standardizer.fit(x).design(x)
    assert d.shape == (50, 25)
    assert np.allclose(d[:, 0], 1.0)


def test_fetch_rejects_wrong_hash(tmp_path):
    bad = tmp_path / "criteo.csv.gz"
    bad.write_bytes(b"not the dataset")
    with pytest.raises(ValueError, match="SHA-256"):
        fetch(bad)


def test_run_smoke(tmp_path):
    out = subprocess.run(
        [sys.executable, str(_CASE / "run.py"), "--smoke"],
        capture_output=True, text=True, check=True,
    )
    assert "wrote" in out.stdout
