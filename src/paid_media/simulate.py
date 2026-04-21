"""Simulators used across the paid-media-playbook case studies.

All data in this repository is synthetic.  Framing is inspired by productivity-
SaaS paid media — SEM and display as the two primary channels driving trial
starts and seat activations — but no real numbers, segments, or accounts are
used.  These simulators exist so reviewers can regenerate data and verify that
the methods recover a known ground truth.

Functions
---------
- :func:`simulate_geo_panel`      : DMA-level weekly panel for geo-lift (case 01).
- :func:`simulate_user_paths`     : converting-user web paths for MTA (case 02).
- :func:`simulate_conversion_lift`: matched-market ghost-ad / PSA data (case 03).
- :func:`simulate_mmm_panel`      : weekly channel spend -> outcome for MMM (case 04).
- :func:`simulate_event_log`      : first-party web event log for attribution (case 05).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .adstock import geometric_adstock
from .saturation import hill_saturation

RNG_DEFAULT = 42


# ----------------------------------------------------------------------------
# Case 01 — geo panel
# ----------------------------------------------------------------------------
@dataclass
class GeoPanel:
    panel: pd.DataFrame  # long format: dma, week, trials, spend, treated
    treated_dma: str
    treatment_week: int
    true_lift_pct: float


def simulate_geo_panel(
    n_dmas: int = 40,
    n_weeks: int = 52,
    treatment_week: int = 40,
    true_lift_pct: float = 0.08,
    base_trials_mean: float = 500.0,
    seed: int = RNG_DEFAULT,
) -> GeoPanel:
    """Weekly trial-starts panel across DMAs with one treated DMA.

    Each DMA has its own level + mild seasonality + weekly noise.  Treated DMA
    gets a multiplicative lift of ``true_lift_pct`` for all weeks ``>= treatment_week``.
    """
    rng = np.random.default_rng(seed)
    dmas = [f"dma_{i:02d}" for i in range(n_dmas)]
    weeks = np.arange(n_weeks)
    treated = dmas[0]

    # per-DMA intercepts (population proxy)
    intercepts = rng.lognormal(mean=np.log(base_trials_mean), sigma=0.35, size=n_dmas)
    # seasonality shared across DMAs (back-to-school + fiscal year-end bumps)
    season = 1.0 + 0.10 * np.sin(2 * np.pi * weeks / 52) + 0.05 * np.sin(2 * np.pi * weeks / 26)

    rows = []
    for i, dma in enumerate(dmas):
        trend = intercepts[i] * season
        noise = rng.normal(0, 0.05 * intercepts[i], n_weeks)
        trials = trend + noise
        if dma == treated:
            trials[treatment_week:] = trials[treatment_week:] * (1.0 + true_lift_pct)
        spend = rng.uniform(5_000, 20_000, n_weeks)
        for w in weeks:
            rows.append(
                {
                    "dma": dma,
                    "week": int(w),
                    "trials": float(trials[w]),
                    "spend": float(spend[w]),
                    "treated": int(dma == treated and w >= treatment_week),
                }
            )
    panel = pd.DataFrame(rows)
    return GeoPanel(
        panel=panel,
        treated_dma=treated,
        treatment_week=treatment_week,
        true_lift_pct=true_lift_pct,
    )


# ----------------------------------------------------------------------------
# Case 02 — converting user paths
# ----------------------------------------------------------------------------
@dataclass
class UserPaths:
    paths: list[list[tuple[str, str]]]           # list of [(channel, interaction), ...]
    converters: np.ndarray                        # bool array, len == len(paths)
    channel_true_contribution: dict[str, float]   # ground-truth incremental share

    @property
    def converting_paths(self) -> list[list[tuple[str, str]]]:
        return [p for p, c in zip(self.paths, self.converters, strict=False) if c]


CHANNELS_DEFAULT = ("brand_search", "paid_search", "display", "video", "organic", "direct")


def simulate_user_paths(
    n_users: int = 20_000,
    channels: tuple[str, ...] = CHANNELS_DEFAULT,
    seed: int = RNG_DEFAULT,
) -> UserPaths:
    """Simulate pre-conversion touch paths with click/view realism.

    Ground truth: upper-funnel display + video carry most of the *incremental*
    contribution, but paid_search / brand_search get the last click.  This is
    the LTA failure mode under-credited display is designed to expose.

    A user's conversion probability is a sigmoid of a channel-weighted exposure
    sum; the true channel weights are returned so downstream MTA methods can be
    scored against them.
    """
    rng = np.random.default_rng(seed)

    # Ground-truth *incremental* contribution of each channel (not observed).
    true_weights = {
        "paid_search": 0.22,
        "brand_search": 0.14,
        "display": 0.28,
        "video": 0.18,
        "organic": 0.10,
        "direct": 0.08,
    }
    # Click probabilities by channel — display is mostly views.
    click_prob = {
        "paid_search": 0.95,
        "brand_search": 0.97,
        "display": 0.08,
        "video": 0.20,
        "organic": 0.60,
        "direct": 1.00,
    }
    # Channels more likely to appear last in the path (intent-capturing):
    last_bias = {"paid_search": 3.0, "brand_search": 4.0, "direct": 3.0}

    paths: list[list[tuple[str, str]]] = []
    converters = np.zeros(n_users, dtype=bool)

    for u in range(n_users):
        path_len = rng.integers(1, 7)
        touches = list(rng.choice(channels, size=path_len, p=_channel_prior(channels)))
        # nudge one intent-capturing channel to the end with some probability
        if rng.random() < 0.55:
            intent = rng.choice(
                list(last_bias.keys()),
                p=np.array(list(last_bias.values())) / sum(last_bias.values()),
            )
            touches[-1] = intent

        # credit-adjusted exposure (views count less than clicks toward conversion)
        expo = 0.0
        path: list[tuple[str, str]] = []
        for ch in touches:
            interaction = "click" if rng.random() < click_prob[ch] else "view"
            w = true_weights.get(ch, 0.0)
            expo += w * (1.0 if interaction == "click" else 0.4)
            path.append((ch, interaction))
        base = -2.8  # -> ~6% conversion rate at typical exposure
        p_conv = 1.0 / (1.0 + np.exp(-(base + expo)))
        converters[u] = rng.random() < p_conv
        paths.append(path)

    return UserPaths(paths=paths, converters=converters, channel_true_contribution=true_weights)


def _channel_prior(channels: tuple[str, ...]) -> np.ndarray:
    prior = {
        "paid_search": 0.18,
        "brand_search": 0.12,
        "display": 0.30,
        "video": 0.18,
        "organic": 0.14,
        "direct": 0.08,
    }
    p = np.array([prior.get(c, 1.0 / len(channels)) for c in channels], dtype=float)
    return p / p.sum()


# ----------------------------------------------------------------------------
# Case 03 — conversion lift (ghost-ad / PSA)
# ----------------------------------------------------------------------------
@dataclass
class ConversionLiftData:
    control: np.ndarray   # shape (n_control,) of 0/1 conversion flags
    treated: np.ndarray
    true_lift_rel: float  # relative lift (treated_rate / control_rate - 1)


def simulate_conversion_lift(
    n_per_arm: int = 250_000,
    base_rate: float = 0.005,
    true_lift_rel: float = 0.10,
    seed: int = RNG_DEFAULT,
) -> ConversionLiftData:
    """Ghost-ad style lift data: two matched arms, low base rate."""
    rng = np.random.default_rng(seed)
    p_ctrl = base_rate
    p_trt = base_rate * (1.0 + true_lift_rel)
    control = (rng.random(n_per_arm) < p_ctrl).astype(int)
    treated = (rng.random(n_per_arm) < p_trt).astype(int)
    return ConversionLiftData(control=control, treated=treated, true_lift_rel=true_lift_rel)


# ----------------------------------------------------------------------------
# Case 04 — MMM weekly panel
# ----------------------------------------------------------------------------
@dataclass
class MMMPanel:
    data: pd.DataFrame          # columns: week, channels, trend, season, trials
    channels: list[str]
    true_roas: dict[str, float]        # steady-state incremental outcome per $ spent
    true_decay: dict[str, float]
    true_halfsat: dict[str, float]     # k of hill_saturation


def simulate_mmm_panel(
    n_weeks: int = 104,
    seed: int = RNG_DEFAULT,
) -> MMMPanel:
    """Two years of weekly spend across SEM + display + video + brand + email.

    Each channel has known adstock (decay), saturation (Hill k), and ``beta``.
    Trials are the outcome.  Trend + seasonality are baked in so the model has
    to learn to separate media from base.
    """
    rng = np.random.default_rng(seed)
    weeks = np.arange(n_weeks)
    channels = ["sem", "display", "video", "brand", "email"]

    # Spend profiles (roughly realistic for a large SaaS advertiser).
    spend = pd.DataFrame(index=weeks)
    spend["sem"] = rng.uniform(80_000, 180_000, n_weeks)
    spend["display"] = rng.uniform(50_000, 150_000, n_weeks) * (1 + 0.4 * np.sin(weeks / 6))
    spend["video"] = (rng.random(n_weeks) < 0.4).astype(float) * rng.uniform(100_000, 400_000, n_weeks)
    spend["brand"] = rng.uniform(30_000, 90_000, n_weeks)
    spend["email"] = rng.uniform(5_000, 15_000, n_weeks)

    true_decay = {"sem": 0.30, "display": 0.55, "video": 0.70, "brand": 0.60, "email": 0.20}
    true_halfsat = {
        "sem": 120_000.0,
        "display": 120_000.0,
        "video": 250_000.0,
        "brand": 60_000.0,
        "email": 10_000.0,
    }
    true_roas = {"sem": 0.050, "display": 0.018, "video": 0.028, "brand": 0.035, "email": 0.080}

    trend = 2_000 + 8 * weeks
    season = 400 * np.sin(2 * np.pi * weeks / 52) + 200 * np.cos(2 * np.pi * weeks / 26)

    media_contrib = np.zeros(n_weeks)
    for ch in channels:
        x = spend[ch].to_numpy()
        adstk = geometric_adstock(x, decay=true_decay[ch])
        sat = hill_saturation(adstk, k=true_halfsat[ch], s=1.4)
        media_contrib = media_contrib + true_roas[ch] * sat * spend[ch].mean()

    noise = rng.normal(0, 150, n_weeks)
    trials = trend + season + media_contrib + noise

    data = spend.copy()
    data.insert(0, "week", weeks)
    data["trend"] = trend
    data["season"] = season
    data["trials"] = trials
    return MMMPanel(
        data=data.reset_index(drop=True),
        channels=channels,
        true_roas=true_roas,
        true_decay=true_decay,
        true_halfsat=true_halfsat,
    )


# ----------------------------------------------------------------------------
# Case 05 — web event log
# ----------------------------------------------------------------------------
@dataclass
class EventLog:
    events: pd.DataFrame  # user_id, device, event_ts, event_type, channel
    cookie_loss_pct: float


def simulate_event_log(
    n_users: int = 5_000,
    platforms: tuple[str, ...] = ("web_desktop", "web_mobile", "ios_app", "android_app"),
    cookie_loss_pct: float = 0.18,
    window_days: int = 30,
    seed: int = RNG_DEFAULT,
) -> EventLog:
    """First-party event log with realistic device mix and cookie loss.

    ``cookie_loss_pct`` of users lose their first-party cookie mid-journey (user
    ids split across sessions).  iOS app sessions never get the cookie.

    Channels are positioned by funnel stage: display and video skew earlier in
    the journey, brand and direct skew later.  Shorter attribution windows
    will therefore over-count brand/direct and under-count display/video —
    the bias the case-study memo is designed to expose.
    """
    rng = np.random.default_rng(seed)
    # Funnel-stage prior: position within journey (0 = first, 1 = last).
    funnel_stage = {
        "video": 0.12,
        "display": 0.22,
        "organic": 0.45,
        "email": 0.55,
        "sem": 0.65,
        "brand": 0.82,
        "direct": 0.90,
    }
    rows = []

    for u in range(n_users):
        device = rng.choice(platforms, p=[0.35, 0.40, 0.15, 0.10])
        n_events = int(rng.integers(2, 12))
        start = rng.uniform(0, window_days * 24 * 3600)
        ts = np.sort(start + rng.uniform(0, window_days * 24 * 3600, n_events))

        # Draw channels with position-aware bias: for each event index i in the
        # path, compute its normalised position in [0, 1], then sample a channel
        # with probability proportional to exp(-|stage - position| / 0.25).
        path_channels = []
        for i in range(n_events):
            pos = (i + 0.5) / n_events
            logits = {ch: -abs(stage - pos) / 0.25 for ch, stage in funnel_stage.items()}
            # softmax
            mx = max(logits.values())
            weights = {ch: np.exp(v - mx) for ch, v in logits.items()}
            total = sum(weights.values())
            probs = [weights[ch] / total for ch in funnel_stage]
            path_channels.append(str(rng.choice(list(funnel_stage.keys()), p=probs)))

        for i, (t, ch) in enumerate(zip(ts, path_channels, strict=False)):
            uid = f"u_{u:05d}"
            if rng.random() < cookie_loss_pct and i > n_events // 2 and device.startswith("web"):
                uid = f"u_{u:05d}_b"
            if device == "ios_app":
                uid = f"ios_{u:05d}_{i}"  # each iOS session looks like a new user
            # Clicks more likely on intent-capturing channels.
            click_p = {
                "sem": 0.9, "brand": 0.95, "direct": 1.0, "email": 0.7,
                "organic": 0.6, "display": 0.08, "video": 0.2,
            }.get(ch, 0.5)
            rows.append(
                {
                    "user_id": uid,
                    "device": device,
                    "event_ts": float(t),
                    "event_type": "click" if rng.random() < click_p else "view",
                    "channel": ch,
                }
            )

    events = pd.DataFrame(rows).sort_values(["user_id", "event_ts"]).reset_index(drop=True)
    return EventLog(events=events, cookie_loss_pct=cookie_loss_pct)
