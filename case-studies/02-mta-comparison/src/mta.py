"""Case 02 — Multi-touch attribution: rule-based vs data-driven.

Implements the five attribution methods that show up on real paid-media teams:

- Last-touch (LTA) and first-touch (FTA)
- Position-based (U-shape 40/20/40)
- Time-decay (exponential with configurable half-life)
- Markov-chain removal effect (Anderl et al. 2016)
- Shapley value attribution (Berman 2018; Dalessandro et al. 2012)

All methods share the same input format: a list of converting users'
``(channel, interaction)`` touch paths, where ``interaction`` is ``"click"`` or
``"view"``.  Views are credit-haircut via :mod:`paid_media.credit` before the
method-specific allocation runs.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd

from paid_media.credit import apply_credit_weights, total_channel_credit


# ---------------------------------------------------------------------------
# Rule-based methods (credit-weighted, click>view)
# ---------------------------------------------------------------------------
def last_touch(paths: Iterable[Iterable[tuple[str, str]]]) -> pd.Series:
    return total_channel_credit(apply_credit_weights(paths), rule="last")


def first_touch(paths: Iterable[Iterable[tuple[str, str]]]) -> pd.Series:
    return total_channel_credit(apply_credit_weights(paths), rule="first")


def linear(paths: Iterable[Iterable[tuple[str, str]]]) -> pd.Series:
    return total_channel_credit(apply_credit_weights(paths), rule="linear")


def position_based(paths: Iterable[Iterable[tuple[str, str]]]) -> pd.Series:
    return total_channel_credit(apply_credit_weights(paths), rule="position")


def time_decay(
    paths: Iterable[Iterable[tuple[str, str]]],
    half_life_touches: float = 2.0,
) -> pd.Series:
    """Exponential time-decay on touch *index*.

    Touches closer to conversion get more credit; decay is indexed in touches
    (not wall-clock) since paths don't carry timestamps here.  Applies the
    click/view haircut before the decay weights.
    """
    weighted = apply_credit_weights(paths)
    agg: dict[str, float] = defaultdict(float)
    for path in weighted:
        if not path:
            continue
        n = len(path)
        # weight[i] = 0.5 ** ((n-1 - i) / half_life_touches)  (most recent = 1)
        gap = np.arange(n - 1, -1, -1)
        w = 0.5 ** (gap / half_life_touches)
        cw = np.array([c for (_, c) in path])
        combined = w * cw
        s = combined.sum()
        if s <= 0:
            continue
        combined = combined / s
        for (ch, _), credit in zip(path, combined, strict=False):
            agg[ch] += float(credit)
    return pd.Series(agg).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Markov-chain removal effect
# ---------------------------------------------------------------------------
def markov_chain(
    paths: Iterable[Iterable[tuple[str, str]]],
    converters: Iterable[bool] | None = None,
    non_converting: Iterable[Iterable[tuple[str, str]]] | None = None,
) -> pd.Series:
    """Markov removal-effect attribution.

    Builds a first-order Markov chain over ``(start)`` -> channels -> ``(conv)``
    and ``(null)``.  Each channel's credit is its *removal effect*: the drop in
    overall conversion probability if that channel is removed (transitions
    rerouted to ``null``).

    Accepts either:
    - ``paths`` as converting paths only (then pass ``non_converting`` separately), or
    - ``paths`` as all paths + ``converters`` mask.
    """
    if non_converting is None:
        assert converters is not None, "pass converters when paths contains non-converters"
        all_paths = list(paths)
        conv_flags = list(converters)
        conv_paths = [p for p, c in zip(all_paths, conv_flags, strict=False) if c]
        non_conv_paths = [p for p, c in zip(all_paths, conv_flags, strict=False) if not c]
    else:
        conv_paths = list(paths)
        non_conv_paths = list(non_converting)

    # Collect channel universe and build transition counts.
    channels: set[str] = set()
    for path in itertools.chain(conv_paths, non_conv_paths):
        for ch, _ in path:
            channels.add(ch)
    nodes = ["(start)", *sorted(channels), "(conv)", "(null)"]
    idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)
    counts = np.zeros((n_nodes, n_nodes))

    def _walk(path: list[tuple[str, str]], converted: bool) -> None:
        seq = ["(start)", *(ch for ch, _ in path), "(conv)" if converted else "(null)"]
        for a, b in zip(seq[:-1], seq[1:], strict=False):
            counts[idx[a], idx[b]] += 1.0

    for p in conv_paths:
        _walk(p, converted=True)
    for p in non_conv_paths:
        _walk(p, converted=False)

    # Row-normalise, treating zero-out rows as absorbing.
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    T = counts / row_sums
    T[idx["(conv)"]] = 0.0
    T[idx["(conv)"], idx["(conv)"]] = 1.0
    T[idx["(null)"]] = 0.0
    T[idx["(null)"], idx["(null)"]] = 1.0

    p_conv_base = _absorbing_prob(T, idx["(start)"], idx["(conv)"])
    total_conversions = float(len(conv_paths))

    credit = {}
    for ch in sorted(channels):
        T_removed = T.copy()
        # Route all transitions into ch -> (null) instead.
        ch_idx = idx[ch]
        incoming = T_removed[:, ch_idx].copy()
        T_removed[:, ch_idx] = 0.0
        T_removed[:, idx["(null)"]] = T_removed[:, idx["(null)"]] + incoming
        T_removed[ch_idx] = 0.0
        T_removed[ch_idx, idx["(null)"]] = 1.0
        p_conv_removed = _absorbing_prob(T_removed, idx["(start)"], idx["(conv)"])
        removal_effect = max(0.0, p_conv_base - p_conv_removed) / max(p_conv_base, 1e-12)
        credit[ch] = removal_effect

    # Normalise removal effects to share of 1, then scale to total conversions.
    s = sum(credit.values())
    if s > 0:
        credit = {k: v / s * total_conversions for k, v in credit.items()}
    return pd.Series(credit).sort_values(ascending=False)


def _absorbing_prob(T: np.ndarray, start: int, target: int, max_iter: int = 2000) -> float:
    """Probability of reaching ``target`` starting from ``start``; iterative solve."""
    n = T.shape[0]
    p = np.zeros(n)
    p[start] = 1.0
    for _ in range(max_iter):
        p_new = p @ T
        if np.allclose(p_new, p, atol=1e-10):
            break
        p = p_new
    return float(p[target])


# ---------------------------------------------------------------------------
# Shapley value attribution
# ---------------------------------------------------------------------------
def shapley(
    paths: Iterable[Iterable[tuple[str, str]]],
    converters: Iterable[bool],
) -> pd.Series:
    """Shapley-value attribution over channel **sets** (coalitions).

    For each coalition of channels, we compute the conversion rate among paths
    whose channel set equals the coalition.  Channel credit is the weighted
    average marginal contribution across all coalitions containing it.

    This is the Berman (2018) / Dalessandro et al. (2012) formulation.
    """
    all_paths = list(paths)
    conv_flags = list(converters)

    coalitions: dict[frozenset[str], list[int]] = defaultdict(list)
    for p, c in zip(all_paths, conv_flags, strict=False):
        key = frozenset(ch for ch, _ in p)
        if not key:
            continue
        coalitions[key].append(int(c))

    channels = sorted({ch for key in coalitions for ch in key})
    n = len(channels)
    conv_rate: dict[frozenset[str], float] = {}
    for key, flags in coalitions.items():
        conv_rate[key] = float(np.mean(flags)) if flags else 0.0
    conv_rate[frozenset()] = 0.0

    # For coalitions that didn't appear in the data, back off to the max of any
    # strict subset that did appear (prevents undefined marginals).
    def v(coal: frozenset[str]) -> float:
        if coal in conv_rate:
            return conv_rate[coal]
        best = 0.0
        for r in range(len(coal), -1, -1):
            for sub in itertools.combinations(coal, r):
                f = frozenset(sub)
                if f in conv_rate:
                    best = max(best, conv_rate[f])
        conv_rate[coal] = best
        return best

    credit: dict[str, float] = {ch: 0.0 for ch in channels}
    # Pre-compute factorials to weight coalition sizes.
    fac = [math.factorial(k) for k in range(n + 1)]

    # Restrict Shapley to channels actually observed in each coalition-path to
    # keep combinatorics tractable at n up to ~10.
    for ch in channels:
        total = 0.0
        others = [c for c in channels if c != ch]
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                S = frozenset(subset)
                with_ch = S | {ch}
                marg = v(with_ch) - v(S)
                weight = fac[r] * fac[n - r - 1] / fac[n]
                total += weight * marg
        credit[ch] = max(0.0, total)

    # Scale so the total matches the observed conversion count.
    total_conv = float(sum(conv_flags))
    s = sum(credit.values())
    if s > 0:
        credit = {k: v / s * total_conv for k, v in credit.items()}
    return pd.Series(credit).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Scoring against ground truth
# ---------------------------------------------------------------------------
def score_against_truth(
    attribution: pd.Series,
    true_weights: dict[str, float],
) -> pd.DataFrame:
    """Compare a method's channel credit (share) to a ground-truth share."""
    share = attribution / attribution.sum() if attribution.sum() > 0 else attribution
    truth_total = sum(true_weights.values())
    truth_share = {k: v / truth_total for k, v in true_weights.items()}
    rows = []
    for ch in sorted(set(share.index) | set(truth_share)):
        rows.append(
            {
                "channel": ch,
                "credit_share": float(share.get(ch, 0.0)),
                "truth_share": float(truth_share.get(ch, 0.0)),
                "over_credit_pp": float(share.get(ch, 0.0) - truth_share.get(ch, 0.0)) * 100,
            }
        )
    return pd.DataFrame(rows).sort_values("truth_share", ascending=False).reset_index(drop=True)
