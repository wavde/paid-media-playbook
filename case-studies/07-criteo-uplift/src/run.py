"""Reproducer for case 07: propensity vs uplift targeting on the Criteo uplift dataset.

Writes case-studies/07-criteo-uplift/results.md. On first run it downloads the
public dataset (~300 MB) into case-studies/07-criteo-uplift/data/ (gitignored)
and verifies its SHA-256.

Run:        python case-studies/07-criteo-uplift/src/run.py
Own copy:   python case-studies/07-criteo-uplift/src/run.py --data path/to/file.csv.gz
CI smoke:   python case-studies/07-criteo-uplift/src/run.py --smoke   (synthetic data, temp output)
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

import numpy as np  # noqa: E402
from criteo_uplift import (  # noqa: E402
    FEATURES,
    N_ROWS,
    Standardizer,
    evaluate_targeting,
    fetch,
    fit_logistic,
    load,
    marginal_uplift,
    predict,
    synthetic_sample,
)

SEED = 20261127
CASE = _HERE.parents[1]
DEFAULT_DATA = CASE / "data" / "criteo-research-uplift-v2.1.csv.gz"
TEST_SHARE = 0.5
N_FOLDS = 20
MAX_TREATED_TRAIN = 2_000_000
REACHES = (0.05, 0.10, 0.20, 0.30, 0.50, 1.00)
MARGINAL = (0.10, 0.40)
CHUNK = 1_000_000
OUTCOMES = ("visit", "conversion")


def score(b: np.ndarray, std: Standardizer, x: np.ndarray) -> np.ndarray:
    out = np.empty(len(x))
    for s in range(0, len(x), CHUNK):
        out[s : s + CHUNK] = predict(b, std.design(x[s : s + CHUNK].astype(np.float64)))
    return out


def arm_summary(df) -> list[str]:
    t = df["treatment"].to_numpy() == 1
    lines = [
        "| Outcome | Control rate | Treated rate | ITT uplift (SE) | Relative lift |",
        "|---|---:|---:|---:|---:|",
    ]
    for y_name in OUTCOMES:
        y = df[y_name].to_numpy(np.float64)
        r1, r0 = y[t].mean(), y[~t].mean()
        se = np.sqrt(r1 * (1 - r1) / t.sum() + r0 * (1 - r0) / (~t).sum())
        lines.append(
            f"| {y_name} | {r0:.5f} | {r1:.5f} | {r1 - r0:+.5f} ({se:.5f}) | {r1 / r0 - 1:+.1%} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=None, help="path to the .csv.gz file")
    parser.add_argument("--smoke", action="store_true",
                        help="synthetic data with the same schema, output to a temp dir (CI)")
    args = parser.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    if args.smoke:
        df = synthetic_sample(200_000, rng)
        out = Path(tempfile.mkdtemp()) / "results.md"
        source = "**SMOKE RUN on synthetic data; not results.**"
    else:
        df = load(fetch(args.data or DEFAULT_DATA))
        if len(df) != N_ROWS:
            raise ValueError(f"expected {N_ROWS:,} rows, got {len(df):,}")
        out = CASE / "results.md"
        source = "Criteo uplift dataset v2.1 (SHA-256 verified)."

    n = len(df)
    is_test = rng.random(n) < TEST_SHARE
    x = df[list(FEATURES)].to_numpy(np.float32)
    t = df["treatment"].to_numpy() == 1
    std = Standardizer.fit(x[~is_test].astype(np.float64))
    te = np.flatnonzero(is_test)
    folds = rng.integers(0, N_FOLDS, size=len(te))
    tr_t = np.flatnonzero(~is_test & t)
    if len(tr_t) > MAX_TREATED_TRAIN:
        tr_t = np.sort(rng.choice(tr_t, MAX_TREATED_TRAIN, replace=False))
    tr_c = np.flatnonzero(~is_test & ~t)

    sections: list[str] = []
    for y_name in OUTCOMES:
        y = df[y_name].to_numpy(np.float64)
        b1 = fit_logistic(std.design(x[tr_t].astype(np.float64)), y[tr_t])
        b0 = fit_logistic(std.design(x[tr_c].astype(np.float64)), y[tr_c])
        p1, p0 = score(b1, std, x[te]), score(b0, std, x[te])
        scores = {
            "Random": rng.random(len(te)),
            "Propensity (P(outcome | treated, x))": p1,
            "Uplift (T-learner: P1 - P0)": p1 - p0,
        }
        yt, tt = y[te], t[te]
        rows = [
            "| Targeting | Reach | Incremental per 1k targeted (SE) | Platform-credited per 1k "
            "| Credited / incremental (SE) | Share of all incremental reached (SE) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        marg = [
            f"| Targeting | Avg incremental per 1k, top {MARGINAL[0]:.0%} "
            f"| Marginal per 1k, {MARGINAL[0]:.0%} to {MARGINAL[1]:.0%} | Marginal / average |",
            "|---|---:|---:|---:|",
        ]
        for name, s in scores.items():
            for r in evaluate_targeting(s, yt, tt, REACHES, folds):
                rows.append(
                    f"| {name} | {r.reach:.0%} | {r.uplift:.2f} ({r.uplift_se:.2f}) "
                    f"| {r.credited:.2f} | {r.inflation:.2f}x ({r.inflation_se:.2f}) "
                    f"| {r.share_of_total:.1%} ({r.share_se:.1%}) |"
                )
            avg, m = marginal_uplift(s, yt, tt, *MARGINAL)
            marg.append(f"| {name} | {avg:.2f} | {m:.2f} | {m / avg:.2f} |")
        sections += [
            f"## Outcome: {y_name}",
            "",
            f"Models fit on the training half; every number below is computed on the "
            f"{len(te):,}-user test half, using its randomized treatment. SEs come from "
            f"{N_FOLDS} random folds of the test half and reflect test-set sampling only, "
            "not model-fitting noise.",
            "",
            *rows,
            "",
            "Average vs marginal incremental return:",
            "",
            *marg,
            "",
        ]

    md = [
        "# Case 07 results: propensity vs uplift targeting on the Criteo uplift dataset",
        "",
        "<!-- Generated by case-studies/07-criteo-uplift/src/run.py. Do not edit by hand. -->",
        "",
        "Reproduce (downloads ~300 MB on first run):",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "python case-studies/07-criteo-uplift/src/run.py",
        "```",
        "",
        f"Seed `{SEED}`. Data: {source} {n:,} users, treatment randomized "
        f"({t.mean():.0%} treated). Intent-to-treat throughout: treatment means eligible to be "
        f"targeted, and only {df['exposure'].mean():.1%} of all users were actually exposed. "
        "Train/test split 50/50. Each model is a ridge logistic regression on the 12 "
        "standardized features and their squares. The propensity model is fit on "
        f"{len(tr_t):,} treated training users, which is what a platform optimizing for "
        f"outcomes learns. The control-arm model uses all {len(tr_c):,} control training users.",
        "",
        "## Randomized baseline (all users)",
        "",
        *arm_summary(df),
        "",
        *sections,
    ]
    out.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
