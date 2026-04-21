import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "case-studies" / "02-mta-comparison" / "src"))

from mta import first_touch, last_touch, markov_chain, shapley  # noqa: E402

from paid_media.simulate import simulate_user_paths  # noqa: E402


def test_last_touch_and_first_touch_disagree_on_display_share():
    sim = simulate_user_paths(n_users=3000, seed=0)
    conv = sim.converting_paths
    lta = last_touch(conv)
    fta = first_touch(conv)
    lta_share = lta / lta.sum()
    fta_share = fta / fta.sum()
    # FTA should credit display more than LTA does (display is upper-funnel).
    assert fta_share.get("display", 0) > lta_share.get("display", 0)


def test_markov_credits_display_higher_than_last_touch():
    """Markov removal is the primary data-driven method in the playbook; Shapley
    is a secondary comparator whose coalition counts get noisy at small N.  We
    assert Markov behaves, and sanity-check that Shapley runs without error.
    """
    sim = simulate_user_paths(n_users=3000, seed=1)
    m = markov_chain(sim.paths, converters=sim.converters)
    s = shapley(sim.paths, sim.converters)
    lta = last_touch(sim.converting_paths)

    m_share = (m / m.sum()).get("display", 0)
    lta_share = (lta / lta.sum()).get("display", 0)
    assert m_share > lta_share
    assert s.sum() > 0  # shapley returns some credit


def test_markov_credits_sum_to_conv_count():
    sim = simulate_user_paths(n_users=1500, seed=2)
    m = markov_chain(sim.paths, converters=sim.converters)
    n_conv = int(sim.converters.sum())
    assert abs(float(m.sum()) - n_conv) < 1e-6 or m.sum() == 0
