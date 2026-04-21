import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "case-studies" / "04-media-mix-modeling" / "src"))

from mmm_map import fit_mmm_map  # noqa: E402

from paid_media.simulate import simulate_mmm_panel  # noqa: E402


def test_mmm_map_high_r2():
    sim = simulate_mmm_panel(seed=0)
    fit = fit_mmm_map(sim.data, sim.channels)
    # MAP with a tough Nelder-Mead on 18 params and real noise.  We require a
    # solid explanatory fit without demanding near-perfect recovery.
    assert fit.r2 > 0.60


def test_mmm_map_beta_ordering_matches_roas():
    """MAP fit should rank channels' fit_beta in roughly the same order as true ROAS."""
    sim = simulate_mmm_panel(seed=0)
    fit = fit_mmm_map(sim.data, sim.channels)
    # Email has the highest ROAS in the DGP; SEM is high-spend high-ROAS dollar;
    # display is lowest ROAS.  We only require the extreme ends to rank right.
    assert fit.beta["email"] > fit.beta["display"]
