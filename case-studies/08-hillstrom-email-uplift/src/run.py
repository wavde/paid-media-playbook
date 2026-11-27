"""Reproducer for case 08: propensity vs uplift targeting on the Hillstrom e-mail experiment.

Writes case-studies/08-hillstrom-email-uplift/results.md. On first run it downloads
the public dataset (~4 MB) into case-studies/08-hillstrom-email-uplift/data/
(gitignored) and verifies its SHA-256.

Run:        python case-studies/08-hillstrom-email-uplift/src/run.py
Own copy:   python case-studies/08-hillstrom-email-uplift/src/run.py --data path/to/file.csv
CI smoke:   python case-studies/08-hillstrom-email-uplift/src/run.py --smoke   (synthetic data)
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "src"))
sys.path.insert(0, str(_HERE.parent))

import numpy as np  # noqa: E402
from hillstrom_uplift import (  # noqa: E402
    CAMPAIGNS,
    CONTROL,
    N_ROWS,
    design_matrix,
    fetch,
    load,
    marginal_ratio,
    synthetic_sample,
    targeting_stats,
)

from paid_media.uplift import fit_logistic, predict  # noqa: E402

SEED = 20261127
CASE = _HERE.parents[1]
DEFAULT_DATA = CASE / "data" / "hillstrom.csv"
N_FOLDS = 5
N_BOOT = 300
REACHES = (0.05, 0.10, 0.20, 0.30, 0.50, 1.00)
MARGINAL = (0.10, 0.40)
OUTCOMES = ("visit", "conversion")
SCORES = ("Random", "Propensity", "Uplift (T-learner)")


def cross_fit_scores(x, y, t, folds, rng):
    """Out-of-fold propensity P(y | treated, x) and T-learner uplift for every user."""
    p1 = np.empty(len(y))
    p0 = np.empty(len(y))
    for f in range(N_FOLDS):
        train, test = folds != f, folds == f
        b1 = fit_logistic(x[train & t], y[train & t])
        b0 = fit_logistic(x[train & ~t], y[train & ~t])
        p1[test], p0[test] = predict(b1, x[test]), predict(b0, x[test])
    return {"Random": rng.random(len(y)), "Propensity": p1, "Uplift (T-learner)": p1 - p0}


def fmt(s, spec, pct=False):
    if not s.estimable:
        return "n/e"
    if pct:
        return f"{s.value:.1%} ({s.se:.1%})"
    return f"{format(s.value, spec)} ({format(s.se, spec)})"


def incremental_conversions(df) -> float:
    """ITT conversion uplift of any e-mail times the number of customers e-mailed."""
    t = df["segment"].ne(CONTROL).to_numpy()
    y = df["conversion"].to_numpy(float)
    return float((y[t].mean() - y[~t].mean()) * t.sum())


def baseline_table(df):
    lines = [
        "| Campaign | Outcome | Control rate | Treated rate | ITT uplift (SE) | Relative lift |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, segs in CAMPAIGNS.items():
        sub = df[df["segment"].isin((*segs, CONTROL))]
        t = sub["segment"].ne(CONTROL).to_numpy()
        for o in OUTCOMES:
            y = sub[o].to_numpy(float)
            r1, r0 = y[t].mean(), y[~t].mean()
            se = np.sqrt(r1 * (1 - r1) / t.sum() + r0 * (1 - r0) / (~t).sum())
            lines.append(f"| {name} | {o} | {r0:.4f} | {r1:.4f} | {r1 - r0:+.4f} ({se:.4f}) "
                         f"| {r1 / r0 - 1:+.1%} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=None, help="path to the Hillstrom CSV")
    parser.add_argument("--smoke", action="store_true",
                        help="synthetic data with the real schema, output to a temp dir (CI)")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    if args.smoke:
        df = synthetic_sample(6_000, rng)
        out = Path(tempfile.mkdtemp()) / "results.md"
        n_boot = 20
        source = "**SMOKE RUN on synthetic data; not results.**"
    else:
        df = load(fetch(args.data or DEFAULT_DATA))
        if len(df) != N_ROWS:
            raise ValueError(f"expected {N_ROWS:,} rows, got {len(df):,}")
        out = CASE / "results.md"
        n_boot = N_BOOT
        source = "Hillstrom MineThatData e-mail experiment (SHA-256 verified)."

    sections, summary = [], []
    for name, segs in CAMPAIGNS.items():
        sub = df[df["segment"].isin((*segs, CONTROL))].reset_index(drop=True)
        t = sub["segment"].ne(CONTROL).to_numpy()
        x = design_matrix(sub)
        folds = rng.integers(0, N_FOLDS, size=len(sub))
        boot = rng.poisson(1.0, size=(n_boot, len(sub))).astype(np.float32)
        for o in OUTCOMES:
            y = sub[o].to_numpy(float)
            scores = cross_fit_scores(x, y, t, folds, rng)
            res = {s: targeting_stats(scores[s], y, t, REACHES, boot) for s in SCORES}
            marg = {s: marginal_ratio(scores[s], y, t, *MARGINAL, boot) for s in SCORES}
            p, u = res["Propensity"], res["Uplift (T-learner)"]
            summary.append(
                f"| {name} | {o} | {fmt(p[0.10]['share'], '', True)} "
                f"| {fmt(u[0.10]['share'], '', True)} | {fmt(p[0.20]['share'], '', True)} "
                f"| {fmt(u[0.20]['share'], '', True)} | {fmt(marg['Propensity']['ratio'], '.2f')} "
                f"| {fmt(p[1.00]['inflation'], '.2f')} | {fmt(p[0.10]['inflation'], '.2f')} |"
            )
            if name != "Any e-mail":
                continue
            rows = [
                "| Targeting | Reach | Incremental per 1k targeted (SE) | Platform-credited per 1k "
                "| Credited / incremental (SE) | Share of all incremental reached (SE) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for s in SCORES:
                for r in REACHES:
                    st = res[s][r]
                    rows.append(
                        f"| {s} | {r:.0%} | {fmt(st['uplift'], '.2f')} "
                        f"| {st['credited'].value:.2f} | {fmt(st['inflation'], '.2f')} "
                        f"| {fmt(st['share'], '', True)} |"
                    )
            mrows = [
                f"| Targeting | Avg incremental per 1k, top {MARGINAL[0]:.0%} "
                f"| Marginal per 1k, {MARGINAL[0]:.0%} to {MARGINAL[1]:.0%} "
                "| Marginal / average |",
                "|---|---:|---:|---:|",
            ]
            for s in SCORES:
                m = marg[s]
                mrows.append(f"| {s} | {fmt(m['average'], '.2f')} | {fmt(m['marginal'], '.2f')} "
                             f"| {fmt(m['ratio'], '.2f')} |")
            sections += [f"## Any e-mail vs no e-mail: {o}", "", *rows, "",
                         "Average vs marginal incremental return:", "", *mrows, ""]

    md = [
        "# Case 08 results: propensity vs uplift targeting on the Hillstrom e-mail experiment",
        "",
        "<!-- Generated by case-studies/08-hillstrom-email-uplift/src/run.py. "
        "Do not edit by hand. -->",
        "",
        "Reproduce (downloads ~4 MB on first run):",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "python case-studies/08-hillstrom-email-uplift/src/run.py",
        "```",
        "",
        f"Seed `{SEED}`. Data: {source} {len(df):,} customers, randomized one third each to a "
        "men's e-mail, a women's e-mail or no e-mail; outcomes over two weeks. Intent-to-treat: "
        "treated means sent the e-mail. Scores are cross-fitted over "
        f"{N_FOLDS} folds, so every customer is scored by models that never saw them. The "
        "propensity model is P(outcome | treated, x); the uplift model is a T-learner "
        "(treated model minus control model). Both are ridge logistic regressions on recency, "
        "log past spend, men's/women's purchase flags, new-customer flag, zip type and channel. "
        f"SEs are from a {n_boot}-replicate Poisson bootstrap over customers with the ranking "
        "held fixed, so they exclude model-fitting noise. **n/e** (not estimable) marks a "
        "ratio whose denominator has a bootstrap coefficient of variation above 0.3, where the "
        "ratio's SE is not meaningful; "
        f"with about {incremental_conversions(df):,.0f} incremental conversions from sending "
        "any e-mail, most conversion ratios are n/e.",
        "",
        "## Randomized baseline (all customers in each comparison)",
        "",
        *baseline_table(df),
        "",
        "## Summary across campaigns",
        "",
        "Share of all incremental outcomes reached when targeting the top 10% or 20%, the "
        f"marginal / average ratio from {MARGINAL[0]:.0%} to {MARGINAL[1]:.0%} reach under "
        "propensity targeting, and credited / incremental for everyone and for the top 10% by "
        "propensity. SEs in parentheses.",
        "",
        "| Campaign | Outcome | Propensity, top 10% | Uplift, top 10% | Propensity, top 20% "
        "| Uplift, top 20% | Marginal / average (propensity) | Credited / incremental, everyone "
        "| Credited / incremental, top 10% propensity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *summary,
        "",
        *sections,
    ]
    out.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
