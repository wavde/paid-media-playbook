"""Click-vs-view credit weighting.

The single most consequential rule in multi-touch attribution at any paid-media
team that actually serves display is the **click-vs-view credit weight**.  A
view (served ad, no click) is evidence of exposure; a click is evidence of
intent.  Treating them equally inflates display's apparent contribution.

The default here mirrors the real-world convention used on paid-media teams at
large tech companies: clicks get full credit, views get a 40% haircut (0.6×).
The canonical factor in this playbook is **clicks=1.0, views=0.4** — a 60%
haircut on views — which is deliberately more conservative because:

1. View-through conversions have a higher share of organic/direct demand.
2. Without the haircut, display systematically steals credit from SEM.
3. It matches the split the author used in production.

The helpers here compute touchpoint weights before they flow into LTA, FTA,
position-based, time-decay, Markov, or Shapley attributions.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

CLICK_WEIGHT: float = 1.0
VIEW_WEIGHT: float = 0.4


def credit_weight(interaction: str) -> float:
    """Return the credit weight for a single interaction type.

    Anything not explicitly recognised gets ``1.0`` (organic / branded / direct
    defaults to full credit; calling code can override).
    """
    i = interaction.strip().lower()
    if i in {"view", "impression", "vtr", "view-through"}:
        return VIEW_WEIGHT
    if i in {"click", "ctr", "click-through"}:
        return CLICK_WEIGHT
    return 1.0


def apply_credit_weights(
    paths: Iterable[Iterable[tuple[str, str]]],
) -> list[list[tuple[str, float]]]:
    """Apply click/view credit to a list of converting paths.

    Parameters
    ----------
    paths : iterable of iterables of ``(channel, interaction)`` tuples
        Each inner iterable is one converting user's ordered touch list.

    Returns
    -------
    list of lists of ``(channel, weight)``
        Same structure with interaction strings replaced by their credit.
    """
    out: list[list[tuple[str, float]]] = []
    for path in paths:
        out.append([(ch, credit_weight(inter)) for (ch, inter) in path])
    return out


def total_channel_credit(
    weighted_paths: Iterable[Iterable[tuple[str, float]]],
    rule: str = "linear",
) -> pd.Series:
    """Aggregate per-channel credit across converting paths under a simple rule.

    Parameters
    ----------
    weighted_paths :
        Output of :func:`apply_credit_weights`.
    rule : {"linear", "last", "first", "position"}
        Touch-level attribution rule applied *after* the click/view weighting.
        ``position`` uses the 40/20/40 U-shape (first 40%, last 40%, middle 20%).

    Returns
    -------
    pd.Series
        Index = channel, value = total credit (in conversions-equivalent units).
    """
    paths = [list(p) for p in weighted_paths]
    agg: dict[str, float] = {}
    for path in paths:
        if not path:
            continue
        n = len(path)
        if rule == "last":
            w = np.zeros(n)
            w[-1] = 1.0
        elif rule == "first":
            w = np.zeros(n)
            w[0] = 1.0
        elif rule == "linear":
            w = np.full(n, 1.0 / n)
        elif rule == "position":
            if n == 1:
                w = np.array([1.0])
            elif n == 2:
                w = np.array([0.5, 0.5])
            else:
                w = np.full(n, 0.2 / (n - 2))
                w[0] = 0.4
                w[-1] = 0.4
        else:
            raise ValueError(f"unknown rule: {rule}")

        # Combine touch-rule weights with click/view weights.
        totals = w * np.array([cw for (_, cw) in path])
        # Renormalise so the conversion is still worth 1 conversion of credit.
        s = totals.sum()
        if s > 0:
            totals = totals / s
        for (ch, _), credit in zip(path, totals, strict=False):
            agg[ch] = agg.get(ch, 0.0) + float(credit)
    return pd.Series(agg, dtype=float).sort_values(ascending=False)
