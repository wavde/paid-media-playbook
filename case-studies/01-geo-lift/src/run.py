"""Reproducer for case 01: geo lift."""

from __future__ import annotations

import sys
from pathlib import Path

# Repo-local import setup so the script runs without `pip install -e .`
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "src"))

from geo_lift import fit_geo_lift, placebo_p_value, power_analysis  # noqa: E402

from paid_media.simulate import simulate_geo_panel  # noqa: E402


def main() -> None:
    sim = simulate_geo_panel(seed=0)
    res = fit_geo_lift(sim.panel, sim.treated_dma, sim.treatment_week, outcome="trials")
    p = placebo_p_value(sim.panel, sim.treated_dma, sim.treatment_week, outcome="trials")
    n_post = sim.panel["week"].max() + 1 - sim.treatment_week

    true_att_pct = sim.true_lift_pct

    donors_panel = sim.panel[sim.panel["dma"] != sim.treated_dma]
    donor_std = float(
        donors_panel.groupby("dma")["trials"].std().mean()
    )
    mde_abs = power_analysis(base_std=donor_std, n_pre=sim.treatment_week, n_post=n_post)

    print(f"Panel: {sim.panel['dma'].nunique()} DMAs x {sim.panel['week'].max()+1} weeks")
    print(f"Treated: {sim.treated_dma}   True lift: {true_att_pct:+.1%}")
    print()
    print(res)
    print()
    print(f"Placebo-in-space p-value (RMSPE-ratio rank): {p:.3f}")
    print(f"Donor pool avg weekly SD: {donor_std:.1f}")
    print(f"MDE (80% power, alpha=0.05, {n_post}w post): {mde_abs:.1f} trials/wk")


if __name__ == "__main__":
    main()
