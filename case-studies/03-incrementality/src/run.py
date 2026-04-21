"""Reproducer for case 03: conversion lift."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "src"))

from lift import analyze_lift, required_sample_size  # noqa: E402

from paid_media.simulate import simulate_conversion_lift  # noqa: E402


def main() -> None:
    # Design
    base_rate = 0.005
    target_lift = 0.10
    n_needed = required_sample_size(base_rate=base_rate, rel_lift=target_lift)
    print("Planning")
    print("--------")
    print(f"Base rate:        {base_rate:.3%}")
    print(f"Target rel lift:  {target_lift:+.0%}")
    print(f"Required n/arm:   {n_needed:,} (alpha=0.05, power=0.80)")

    # Readout at planned n
    sim = simulate_conversion_lift(
        n_per_arm=n_needed,
        base_rate=base_rate,
        true_lift_rel=target_lift,
        seed=0,
    )
    res = analyze_lift(sim.control, sim.treated)
    print()
    print("Readout")
    print("-------")
    print(res)
    print()

    # Sensitivity: what if the true lift is half what we planned for?
    sim_half = simulate_conversion_lift(
        n_per_arm=n_needed,
        base_rate=base_rate,
        true_lift_rel=target_lift / 2,
        seed=1,
    )
    res_half = analyze_lift(sim_half.control, sim_half.treated)
    print("Sensitivity: true lift = half of target")
    print(res_half)


if __name__ == "__main__":
    main()
