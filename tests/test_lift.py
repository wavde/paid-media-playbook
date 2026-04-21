import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "case-studies" / "03-incrementality" / "src"))

from lift import analyze_lift, required_sample_size  # noqa: E402

from paid_media.simulate import simulate_conversion_lift  # noqa: E402


def test_required_sample_size_scales_with_lift():
    n_small = required_sample_size(base_rate=0.005, rel_lift=0.05)
    n_big = required_sample_size(base_rate=0.005, rel_lift=0.20)
    assert n_small > n_big


def test_required_sample_size_low_base_rate():
    # At p=0.005, +10% rel lift, two-prop pooled z requires ~325k per arm.
    n = required_sample_size(base_rate=0.005, rel_lift=0.10)
    assert 250_000 < n < 400_000, n


def test_analyze_lift_detects_large_effect():
    sim = simulate_conversion_lift(n_per_arm=150_000, base_rate=0.005, true_lift_rel=0.15, seed=0)
    res = analyze_lift(sim.control, sim.treated)
    assert res.p_value < 0.05
    assert 0.05 < res.rel_lift < 0.25


def test_analyze_lift_null_not_rejected():
    sim = simulate_conversion_lift(n_per_arm=20_000, base_rate=0.005, true_lift_rel=0.0, seed=1)
    res = analyze_lift(sim.control, sim.treated)
    assert res.p_value > 0.05
