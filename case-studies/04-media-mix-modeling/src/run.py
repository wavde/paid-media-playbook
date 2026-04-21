"""Reproducer for case 04: MMM (MAP + optional Bayesian)."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "src"))

import pandas as pd  # noqa: E402
from mmm_map import fit_mmm_map  # noqa: E402

from paid_media.simulate import simulate_mmm_panel  # noqa: E402


def _summarise(sim, fit) -> pd.DataFrame:
    rows = []
    for ch in sim.channels:
        mean_spend = float(sim.data[ch].mean())
        fit.beta[ch] * 0.5  # saturation at own half-sat is 0.5 by def
        true_roas = sim.true_roas[ch]
        rows.append(
            {
                "channel": ch,
                "true_decay": sim.true_decay[ch],
                "fit_decay": round(fit.decay[ch], 2),
                "true_halfsat": sim.true_halfsat[ch],
                "fit_halfsat": round(fit.halfsat[ch], 0),
                "true_roas_shape_scale": round(true_roas, 4),
                "fit_beta": round(fit.beta[ch], 4),
                "mean_spend": round(mean_spend, 0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    sim = simulate_mmm_panel(seed=0)
    print(f"Panel: {len(sim.data)} weeks x {len(sim.channels)} channels")
    fit = fit_mmm_map(sim.data, sim.channels, outcome="trials")
    print(f"MAP fit R² = {fit.r2:.3f}")
    print()
    print(_summarise(sim, fit).to_string(index=False))
    print()

    try:
        import pymc  # noqa: F401
        from mmm_bayes import fit_mmm_bayes, posterior_roas  # noqa: E402

        print("Running Bayesian MMM (smaller draws for demo)...")
        bfit = fit_mmm_bayes(sim.data, sim.channels, draws=500, tune=500, chains=2, random_seed=0)
        mean_spend = {ch: float(sim.data[ch].mean()) for ch in sim.channels}
        print(posterior_roas(bfit, sim.channels, mean_spend).to_string(index=False))
    except ImportError:
        print("PyMC not installed — skipping Bayesian fit.")
        print("Install with:  pip install -r requirements-mmm.txt")


if __name__ == "__main__":
    main()
