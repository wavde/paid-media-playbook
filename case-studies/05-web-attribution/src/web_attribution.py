"""Case 05 — Web attribution: session stitching, cookie loss, window sensitivity.

Given a first-party event log with (user_id, device, event_ts, event_type,
channel), this module does the plumbing every production web-attribution stack
has to get right *before* any attribution rule is applied:

- **Session stitching**: gaps > ``session_gap_seconds`` end a session.
- **Device-bridged identity**: stitch sessions that share a ``device_bridge_id``
  (e.g., authenticated user id post-login) into one journey.
- **Cookie-loss diagnostic**: count the fraction of users who appear under
  multiple ids within the window.
- **Attribution-window sensitivity**: compute last-touch share by channel under
  1-day, 7-day and 28-day post-click windows; the delta is the visibility you
  lose when iOS/ATT truncates you to short windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class StitchResult:
    events: pd.DataFrame            # input + session_id + journey_id
    n_users: int                    # distinct user ids
    n_sessions: int
    n_journeys: int
    cookie_loss_share: float        # share of journeys spanning >1 user_id


def stitch_sessions(
    events: pd.DataFrame,
    session_gap_seconds: float = 30 * 60,
) -> StitchResult:
    """Assign session_id and journey_id to events.

    A **session** is a contiguous run of events for a single user_id separated
    by gaps < ``session_gap_seconds``.  A **journey** is a set of sessions
    linked by a device bridge; here we approximate the bridge by the
    common prefix of ``user_id`` (``u_00123``, ``u_00123_b``, ``u_00123_ios``
    all fold to the same journey).  A real system would use a login id, an
    IDFV, or a probabilistic match.
    """
    df = events.sort_values(["user_id", "event_ts"]).copy()
    df["journey_id"] = df["user_id"].str.extract(r"^(u_\d+)")[0].fillna(df["user_id"])
    gap = df.groupby("user_id")["event_ts"].diff().fillna(float("inf"))
    new_session = (gap > session_gap_seconds).astype(int)
    df["session_id"] = df["user_id"] + "|" + new_session.groupby(df["user_id"]).cumsum().astype(str)

    journey_user_counts = df.groupby("journey_id")["user_id"].nunique()
    cookie_loss_share = float((journey_user_counts > 1).mean()) if len(journey_user_counts) else 0.0

    return StitchResult(
        events=df,
        n_users=int(df["user_id"].nunique()),
        n_sessions=int(df["session_id"].nunique()),
        n_journeys=int(df["journey_id"].nunique()),
        cookie_loss_share=cookie_loss_share,
    )


def attribution_window_sensitivity(
    stitched: StitchResult,
    windows_days: tuple[int, ...] = (1, 7, 28),
) -> pd.DataFrame:
    """Last-touch share per channel under different post-click windows.

    We define "conversion" as the last event of each journey (whichever it is),
    then ask: among the touches in the preceding ``W`` days, which channel
    gets last-touch credit?  Shorter windows miss upper-funnel touches and
    concentrate credit on brand-search / direct.
    """
    df = stitched.events.sort_values(["journey_id", "event_ts"]).copy()
    out_rows = []
    for w_days in windows_days:
        w_sec = w_days * 24 * 3600
        credit: dict[str, float] = {}
        for _jid, g in df.groupby("journey_id"):
            g = g.reset_index(drop=True)
            conv_ts = float(g["event_ts"].iloc[-1])
            pre = g[g["event_ts"] >= conv_ts - w_sec]
            pre = pre.iloc[:-1]  # exclude the conversion event itself
            if pre.empty:
                continue
            last = pre.iloc[-1]
            ch = str(last["channel"])
            credit[ch] = credit.get(ch, 0.0) + 1.0
        total = sum(credit.values())
        for ch, v in credit.items():
            out_rows.append(
                {"window_days": w_days, "channel": ch, "share": v / total if total else 0.0}
            )
    return (
        pd.DataFrame(out_rows)
        .pivot(index="channel", columns="window_days", values="share")
        .fillna(0.0)
        .sort_values(1, ascending=False)
    )


def ios_attribution_loss(
    stitched: StitchResult,
) -> pd.DataFrame:
    """Compare what the attribution stack can see on iOS-app vs web.

    On iOS-app sessions, each session looks like a new user (IDFA absent, ATT
    opt-out).  This function quantifies the journey-level visibility loss.
    """
    ev = stitched.events
    by_device = ev.groupby("device").agg(
        events=("event_ts", "size"),
        unique_users=("user_id", "nunique"),
        unique_journeys=("journey_id", "nunique"),
    )
    by_device["events_per_journey"] = by_device["events"] / by_device["unique_journeys"]
    return by_device.sort_values("events", ascending=False)


def path_depth_distribution(stitched: StitchResult) -> pd.Series:
    """Distribution of events-per-journey, capped at 10."""
    depth = stitched.events.groupby("journey_id").size().clip(upper=10)
    return depth.value_counts().sort_index()
