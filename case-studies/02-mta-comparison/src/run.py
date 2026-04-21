"""Reproducer for case 02: MTA comparison."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "src"))

import pandas as pd  # noqa: E402
from mta import (  # noqa: E402
    first_touch,
    last_touch,
    linear,
    markov_chain,
    position_based,
    score_against_truth,
    shapley,
    time_decay,
)

from paid_media.simulate import simulate_user_paths  # noqa: E402


def main() -> None:
    sim = simulate_user_paths(n_users=10_000, seed=0)
    conv_paths = sim.converting_paths
    print(f"Paths: {len(sim.paths):,}  |  converters: {sim.converters.sum():,}")
    print()

    methods = {
        "Last-touch": last_touch(conv_paths),
        "First-touch": first_touch(conv_paths),
        "Linear": linear(conv_paths),
        "Position-based (40/20/40)": position_based(conv_paths),
        "Time-decay (hl=2)": time_decay(conv_paths, half_life_touches=2.0),
        "Markov removal": markov_chain(sim.paths, converters=sim.converters),
        "Shapley": shapley(sim.paths, sim.converters),
    }

    print("Channel credit share by method (rows = methods, cols = channels):")
    shares = pd.DataFrame(
        {
            name: (s / s.sum()).reindex(list(sim.channel_true_contribution))
            for name, s in methods.items()
        }
    ).T
    truth_share = pd.Series(
        {
            k: v / sum(sim.channel_true_contribution.values())
            for k, v in sim.channel_true_contribution.items()
        }
    )
    shares.loc["** truth **"] = truth_share
    print(shares.fillna(0).round(3).to_string())
    print()

    print("How much does last-touch over-credit / under-credit each channel (pp)?")
    score = score_against_truth(
        methods["Last-touch"], sim.channel_true_contribution,
    )
    print(score.to_string(index=False))


if __name__ == "__main__":
    main()
