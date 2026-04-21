"""Reproducer for case 05: web attribution."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "src"))

from web_attribution import (  # noqa: E402
    attribution_window_sensitivity,
    ios_attribution_loss,
    path_depth_distribution,
    stitch_sessions,
)

from paid_media.simulate import simulate_event_log  # noqa: E402


def main() -> None:
    log = simulate_event_log(n_users=5_000, seed=0)
    stitched = stitch_sessions(log.events, session_gap_seconds=30 * 60)

    print("Event log")
    print("---------")
    print(f"Events:           {len(log.events):,}")
    print(f"Simulated cookie-loss rate (ground truth): {log.cookie_loss_pct:.1%}")
    print()

    print("After session stitching")
    print("-----------------------")
    print(f"Distinct user_ids:  {stitched.n_users:,}")
    print(f"Distinct sessions:  {stitched.n_sessions:,}")
    print(f"Distinct journeys:  {stitched.n_journeys:,}")
    print(f"Share of journeys spanning >1 user_id (cookie loss): {stitched.cookie_loss_share:.1%}")
    print()

    print("Attribution-window sensitivity (last-touch share)")
    print("-------------------------------------------------")
    print(attribution_window_sensitivity(stitched, windows_days=(1, 7, 28)).round(3))
    print()

    print("iOS attribution loss (events per journey by device)")
    print("---------------------------------------------------")
    print(ios_attribution_loss(stitched).round(2))
    print()

    print("Journey depth distribution")
    print("--------------------------")
    print(path_depth_distribution(stitched))


if __name__ == "__main__":
    main()
