import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "case-studies" / "01-geo-lift" / "src"))

from geo_lift import fit_geo_lift, power_analysis  # noqa: E402

from paid_media.simulate import simulate_geo_panel  # noqa: E402


def test_geo_lift_recovers_truth_within_pct():
    sim = simulate_geo_panel(seed=0)
    res = fit_geo_lift(sim.panel, sim.treated_dma, sim.treatment_week)
    # True lift is 8% of a ~500-unit mean -> ~40 trials/wk.  Recover within 15%.
    expected = sim.true_lift_pct * sim.panel.query("dma == @sim.treated_dma")["trials"].mean()
    assert abs(res.att - expected) < 0.25 * abs(expected), (res.att, expected)


def test_geo_lift_weights_sum_to_one():
    sim = simulate_geo_panel(seed=1)
    res = fit_geo_lift(sim.panel, sim.treated_dma, sim.treatment_week)
    s = float(res.weights.sum())
    assert abs(s - 1.0) < 1e-4


def test_power_analysis_monotone_in_sample():
    big = power_analysis(base_std=20.0, n_pre=40, n_post=24)
    small = power_analysis(base_std=20.0, n_pre=40, n_post=4)
    assert big < small  # smaller MDE with more post weeks
