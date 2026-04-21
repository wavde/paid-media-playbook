import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "case-studies" / "05-web-attribution" / "src"))

from web_attribution import (  # noqa: E402
    attribution_window_sensitivity,
    ios_attribution_loss,
    stitch_sessions,
)

from paid_media.simulate import simulate_event_log  # noqa: E402


def test_stitch_produces_journeys_leq_user_ids():
    log = simulate_event_log(n_users=500, seed=0)
    res = stitch_sessions(log.events)
    # Journeys merge cookie-lost ids, so journey count <= user_id count.
    assert res.n_journeys <= res.n_users
    assert 0.0 <= res.cookie_loss_share <= 1.0


def test_window_sensitivity_1d_vs_28d_differs():
    log = simulate_event_log(n_users=300, seed=1)
    res = stitch_sessions(log.events)
    tab = attribution_window_sensitivity(res, windows_days=(1, 28))
    # At least one channel should have meaningfully different share under 1d vs 28d.
    diffs = (tab[28] - tab[1]).abs()
    assert diffs.max() > 0.02


def test_ios_has_lower_events_per_journey():
    log = simulate_event_log(n_users=500, seed=2)
    res = stitch_sessions(log.events)
    tab = ios_attribution_loss(res)
    if "ios_app" in tab.index and "web_desktop" in tab.index:
        assert tab.loc["ios_app", "events_per_journey"] < tab.loc["web_desktop", "events_per_journey"]
