"""Small, dependency-light matplotlib helpers used by case reproducers.

These are deliberately minimal — the case studies are the feature, not the
charting library.  All helpers accept an ``ax`` to support notebook composition
and return the axes for chaining.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

PLAYBOOK_PALETTE = {
    "sem": "#2b6cb0",
    "paid_search": "#2b6cb0",
    "brand_search": "#2c5282",
    "display": "#d97706",
    "video": "#7c3aed",
    "brand": "#059669",
    "email": "#0f766e",
    "organic": "#6b7280",
    "direct": "#111827",
    "treated": "#b91c1c",
    "control": "#1f2937",
    "synthetic": "#0ea5e9",
}


def styled_ax(ax: plt.Axes | None = None, title: str | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    if title:
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    return ax


def bar_channel(series, ax: plt.Axes | None = None, title: str | None = None) -> plt.Axes:
    ax = styled_ax(ax, title=title)
    colors = [PLAYBOOK_PALETTE.get(ch, "#4b5563") for ch in series.index]
    ax.bar(np.arange(len(series)), series.values, color=colors)
    ax.set_xticks(np.arange(len(series)))
    ax.set_xticklabels(series.index, rotation=30, ha="right")
    return ax
