"""Reproducer for case 06: measurement when targeting, creative and pacing are model-driven.

Writes case-studies/06-ai-mediated-measurement/results.md. Every simulated
number appears next to the true value the simulator knows by construction.

Run:       python case-studies/06-ai-mediated-measurement/src/run.py
CI smoke:  python case-studies/06-ai-mediated-measurement/src/run.py --smoke   (tiny, temp output)
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

SEED = 20261127
OUT = _HERE.parents[1] / "results.md"

# Part A: targeting
SHARPNESS = (0.0, 0.5, 0.8, 0.95)
N_USERS = 400_000
Q_SERVE = 0.20
REPS_A = 200
# Part A2: budget transport
Q_LOW, Q_HIGH = 0.10, 0.40
TRANSPORT_SHARPNESS = 0.8
N_REFERENCE = 2_000_000
N_SPLIT = 2_000_000
REPS_SPLIT = 100
# Part B: creative tests
N_CREATIVE = 600_000
Q_CREATIVE = 0.30
AFFINITY = 2.0
REPS_B = 200
# Part C: MMM
DEMAND_CORR = (0.0, 0.3, 0.6, 0.9)
REPS_C = 500
VALUE_PER_SALE = 240.0  # makes the true ROI about 0.71, below break-even 1.0


def _signed(x: float, spec: str) -> str:
    """Signed format without a stray '-0'."""
    text = format(x, spec)
    stripped = text.lstrip("+-").replace("0", "").replace(".", "").replace("%", "")
    return text.replace("-", "+", 1) if text.startswith("-") and not stripped else text


def part_a(rng: np.random.Generator) -> list[str]:
    rows = {s: {"ratio": [], "bias": [], "cover": [], "cvr": [], "inc": []} for s in SHARPNESS}
    for _ in range(REPS_A):
        users = simulate_users(N_USERS, rng)
        for s in SHARPNESS:
            sel = top_share(targeting_score(users, s), Q_SERVE)
            r = ghost_ads_read(users, sel, rng)
            rows[s]["ratio"].append(r.credited / r.truth)
            rows[s]["bias"].append((r.lift - r.truth) / r.truth)
            rows[s]["cover"].append(abs(r.lift - r.truth) <= Z_95 * r.lift_se)
            rows[s]["cvr"].append(r.credited / r.n_selected)
            rows[s]["inc"].append(1000 * r.truth / r.n_selected)
    lines = [
        "| Targeting sharpness | Served CVR | True incremental per 1k served "
        "| Platform-credited / true incremental | Lift-test bias | Lift-test 95% CI coverage |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for s in SHARPNESS:
        r = {k: np.asarray(v, dtype=float) for k, v in rows[s].items()}
        lines.append(
            f"| {s:.2f} | {r['cvr'].mean():.4f} | {r['inc'].mean():.1f} "
            f"| {r['ratio'].mean():.2f}x | {_signed(r['bias'].mean(), '+.1%')} "
            f"| {r['cover'].mean():.1%} |"
        )
    return lines


def part_a2(rng: np.random.Generator) -> tuple[list[str], list[str], float]:
    ref = simulate_users(N_REFERENCE, rng)
    exact = {s: average_and_marginal_uplift(ref, s, Q_LOW, Q_HIGH) for s in SHARPNESS}
    avg_t, marg_t = exact[TRANSPORT_SHARPNESS]
    cost = 0.5 * (avg_t + marg_t)  # cost per served user, in units of conversion value

    ref_lines = [
        f"| Targeting sharpness | True avg iROAS at {Q_LOW:.0%} reach "
        f"| True marginal iROAS, {Q_LOW:.0%} to {Q_HIGH:.0%} | Marginal / average |",
        "|---:|---:|---:|---:|",
    ]
    for s in SHARPNESS:
        a, m = exact[s]
        ref_lines.append(f"| {s:.2f} | {a / cost:.2f} | {m / cost:.2f} | {m / a:.2f} |")

    calls = {k: [] for k in ("platform_avg", "platform_marg", "lift_low", "lift_high", "split")}
    cover, est, truth = [], [], []
    for _ in range(REPS_SPLIT):
        users = simulate_users(N_SPLIT, rng)
        r = budget_split_read(users, TRANSPORT_SHARPNESS, Q_LOW, Q_HIGH, rng)
        calls["platform_avg"].append(r.platform_avg_low > cost)
        calls["platform_marg"].append(r.platform_marginal > cost)
        calls["lift_low"].append(r.lift_avg_low > cost)
        calls["lift_high"].append(r.lift_avg_high > cost)
        calls["split"].append(r.lift_marginal > cost)
        cover.append(abs(r.lift_marginal - r.true_marginal) <= Z_95 * r.lift_marginal_se)
        est.append(r.lift_marginal / cost)
        truth.append(r.true_marginal / cost)
    labels = {
        "platform_avg": f"Platform-credited ROAS at {Q_LOW:.0%} reach",
        "platform_marg": "Platform-credited marginal ROAS (two-cell)",
        "lift_low": f"Lift test at {Q_LOW:.0%} reach (average iROAS)",
        "lift_high": f"Lift test at {Q_HIGH:.0%} reach (average iROAS)",
        "split": "Two-cell budget-split lift test (marginal iROAS)",
    }
    dec_lines = [
        "| Evidence used for \"scale up?\" | Says scale up (truth: don't) |",
        "|---|---:|",
    ]
    for key, label in labels.items():
        dec_lines.append(f"| {label} | {np.mean(calls[key]):.0%} |")
    dec_lines += [
        "",
        f"Two-cell marginal iROAS estimate: mean {np.mean(est):.2f} against a true "
        f"{np.mean(truth):.2f}, with 95% CI coverage {np.mean(cover):.0%} "
        f"({REPS_SPLIT} replications, {N_SPLIT:,} users each).",
    ]
    return ref_lines, dec_lines, cost


def part_b(rng: np.random.Generator) -> list[str]:
    picks = {"platform": [], "lift": [], "audience": []}
    diffs = {"platform": [], "lift": [], "audience": []}
    truths = {"platform": [], "lift": [], "audience": []}
    for _ in range(REPS_B):
        pop = simulate_creative_population(N_CREATIVE, rng)
        platform, lift = platform_creative_test(pop, Q_CREATIVE, AFFINITY, rng)
        audience = audience_randomized_creative_test(
            simulate_creative_population(N_CREATIVE, rng), Q_CREATIVE, rng
        )
        for key, read in (("platform", platform), ("lift", lift), ("audience", audience)):
            picks[key].append(read.pick())
            diffs[key].append(read.diff)
            truths[key].append(read.truth)
    rows = [
        ("platform", "In-platform A/B, served CVR", "Creative effect (truth 0)"),
        ("lift", "In-platform A/B, per-arm ghost-ads lift",
         "Incrementality of each campaign as delivered"),
        ("audience", "Audience fixed first, creative randomized", "Creative effect (truth 0)"),
    ]
    lines = [
        "| Design | Estimand | A minus B: estimate (truth) | Picks A | Picks B | No winner |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, design, estimand in rows:
        p = np.asarray(picks[key])
        lines.append(
            f"| {design} | {estimand} | {_signed(np.mean(diffs[key]), '+.4f')} "
            f"({_signed(np.mean(truths[key]), '+.4f')}) "
            f"| {np.mean(p == 'A'):.0%} | {np.mean(p == 'B'):.0%} | {np.mean(p == 'none'):.0%} |"
        )
    return lines


def part_c(rng: np.random.Generator) -> list[str]:
    lines = [
        "| Corr(spend, demand) | Method | Mean ROI (true) | ROI bias | 95% CI coverage "
        "| Says ROI > 1 (truth: no) | MMM vs lift conflict flagged |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for corr in DEMAND_CORR:
        out = {k: {"roi": [], "cover": [], "over": []} for k in ("uncalibrated", "prior",
                                                                 "lift_only")}
        true_roi, flags = [], []
        for _ in range(REPS_C):
            panel = simulate_paced_panel(corr, rng)
            scale = VALUE_PER_SALE * panel.media_per_dollar
            b_m, se_m = fit_mmm(panel)
            b_l, se_l = lift_window_beta(panel, rng)
            flags.append(conflict_flag(b_m, se_m, b_l, se_l))
            true_roi.append(scale * panel.beta)
            for key, (b, se) in calibrate(b_m, se_m, b_l, se_l).items():
                out[key]["roi"].append(scale * b)
                out[key]["cover"].append(abs(b - panel.beta) <= Z_95 * se)
                out[key]["over"].append(scale * b > 1.0)
        t = float(np.mean(true_roi))
        labels = {"uncalibrated": "MMM alone", "prior": "MMM with lift test as prior",
                  "lift_only": "Lift-test coefficient"}
        for key, label in labels.items():
            roi = float(np.mean(out[key]["roi"]))
            flag = f"{np.mean(flags):.0%}" if key == "uncalibrated" else ""
            lines.append(
                f"| {corr:.1f} | {label} | {roi:.2f} ({t:.2f}) | {_signed(roi / t - 1, '+.0%')} "
                f"| {np.mean(out[key]['cover']):.0%} | {np.mean(out[key]['over']):.0%} | {flag} |"
            )
    return lines


def main() -> None:
    global OUT, N_USERS, REPS_A, N_REFERENCE, N_SPLIT, REPS_SPLIT, N_CREATIVE, REPS_B, REPS_C
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true",
                        help="tiny sizes, output to a temp dir (for CI)")
    if parser.parse_args().smoke:
        N_USERS, REPS_A, N_REFERENCE, N_SPLIT, REPS_SPLIT = 20_000, 2, 50_000, 50_000, 2
        N_CREATIVE, REPS_B, REPS_C = 20_000, 2, 3
        OUT = Path(tempfile.mkdtemp()) / "results.md"

    seeds = np.random.SeedSequence(SEED).spawn(4)
    a_lines = part_a(np.random.default_rng(seeds[0]))
    ref_lines, dec_lines, cost = part_a2(np.random.default_rng(seeds[1]))
    b_lines = part_b(np.random.default_rng(seeds[2]))
    c_lines = part_c(np.random.default_rng(seeds[3]))

    md = [
        "# Case 06 results: measurement when targeting, creative and pacing are model-driven",
        "",
        "<!-- Generated by case-studies/06-ai-mediated-measurement/src/run.py. "
        "Do not edit by hand. -->",
        "",
        f"Seed `{SEED}`. All data is synthetic and the truth is known by construction. "
        "Reproduce with:",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "python case-studies/06-ai-mediated-measurement/src/run.py",
        "```",
        "",
        "## 1. Algorithmic targeting: platform credit vs incrementality",
        "",
        f"The platform serves the top {Q_SERVE:.0%} of {N_USERS:,} users by a score whose "
        "correlation with *baseline* conversion intent is the targeting sharpness. A 50% "
        "ghost-ads holdout runs among the users it would serve. Platform credit counts every "
        f"conversion among served users. {REPS_A} replications.",
        "",
        *a_lines,
        "",
        "## 2. Budget transport: a lift test at one budget doesn't price the next dollar",
        "",
        f"Exact values from a {N_REFERENCE:,}-user reference population. iROAS = incremental "
        "conversion value / spend, break-even 1.0. The cost per served user is set to "
        f"{cost:.4f} conversions, halfway between the average and marginal uplift at sharpness "
        f"{TRANSPORT_SHARPNESS}. At that sharpness, scaling from {Q_LOW:.0%} to {Q_HIGH:.0%} "
        "reach is profitable on average and unprofitable at the margin.",
        "",
        *ref_lines,
        "",
        f"Decision at sharpness {TRANSPORT_SHARPNESS}: scale reach from {Q_LOW:.0%} to "
        f"{Q_HIGH:.0%}? Each method decides on its point estimate. {REPS_SPLIT} replications "
        f"of a {N_SPLIT:,}-user two-cell test (random halves run each reach level, each with a "
        "50% ghost-ads holdout).",
        "",
        *dec_lines,
        "",
        "## 3. Divergent delivery in creative tests",
        "",
        f"Creatives A and B have identical per-person effects. The platform delivers A to the "
        f"high-baseline, low-lift segment and B to the low-baseline, high-lift segment "
        f"(affinity {AFFINITY}). Reach is {Q_CREATIVE:.0%}, with {N_CREATIVE:,} users per test "
        f"and {REPS_B} replications. A winner requires |z| > 1.96.",
        "",
        *b_lines,
        "",
        "## 4. MMM when the platform paces spend with demand",
        "",
        f"104 weeks of sales with AR(1) latent demand that the MMM cannot see. Media transforms "
        "are known exactly, which is the MMM's best case. The MMM is OLS with Newey-West SEs. "
        "The lift test is an 8-week, 20% user-level holdout. The lift-test-as-prior row is a "
        f"precision-weighted average. Value per sale is {VALUE_PER_SALE:.0f}, so the true ROI is "
        f"about 0.71. {REPS_C} replications per row.",
        "",
        *c_lines,
        "",
    ]
    OUT.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
