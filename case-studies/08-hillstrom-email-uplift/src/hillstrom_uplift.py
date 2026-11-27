"""Propensity vs uplift targeting on the Hillstrom e-mail experiment.

Case 07 found, on the Criteo display-advertising data, that users a propensity
model ranks highest are also the ones the ad moves most in absolute terms, so
propensity targeting captures most of the incremental outcomes and an uplift
model only helps at tight reach. This module repeats the check on a different
advertiser and channel: a retailer's e-mail campaign.

Data: Kevin Hillstrom, "The MineThatData E-Mail Analytics And Data Mining
Challenge" (2008). 64,000 customers who purchased in the last twelve months,
randomized one third each to a men's merchandise e-mail, a women's merchandise
e-mail, or no e-mail. Visits, conversions and spend were tracked for two weeks.
The author published it for public analysis but states no license, so the
file is downloaded at runtime and not committed. It is served over plain HTTP,
so the SHA-256 is verified before use.

Everything is intent-to-treat: "treated" means "sent the e-mail".
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from paid_media.uplift import top_mask

URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
N_ROWS = 64_000
COLUMNS = (
    "recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie",
    "channel", "segment", "visit", "conversion", "spend",
)
CONTROL = "No E-Mail"
CAMPAIGNS = {
    "Any e-mail": ("Mens E-Mail", "Womens E-Mail"),
    "Men's e-mail": ("Mens E-Mail",),
    "Women's e-mail": ("Womens E-Mail",),
}
ZIP_LEVELS = ("Rural", "Surburban", "Urban")
CHANNEL_LEVELS = ("Multichannel", "Phone", "Web")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch(dest: Path) -> Path:
    """Download the dataset to ``dest`` (if missing) and verify its SHA-256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading {URL} (~4 MB) ...")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(dest)
    digest = sha256(dest)
    if digest != SHA256:
        raise ValueError(f"{dest} has SHA-256 {digest}, expected {SHA256}")
    return dest


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if tuple(df.columns) != COLUMNS:
        raise ValueError(f"unexpected columns: {list(df.columns)}")
    return df


def synthetic_sample(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Stand-in with the real schema, for CI smoke runs and tests.

    Visit log-odds rise with past spend and fall with recency, and the e-mail
    adds a constant to the log-odds, so absolute uplift rises with baseline
    propensity.
    """
    recency = rng.integers(1, 13, n)
    history = np.round(rng.lognormal(4.8, 0.9, n), 2)
    seg = rng.choice(["Mens E-Mail", "Womens E-Mail", CONTROL], n)
    treated = seg != CONTROL
    eta = -2.2 + 0.5 * np.log1p(history) / 5 - 0.08 * recency
    visit = rng.random(n) < 1 / (1 + np.exp(-(eta + 0.5 * treated)))
    conv = visit & (rng.random(n) < 0.05 + 0.02 * treated)
    return pd.DataFrame({
        "recency": recency,
        "history_segment": np.where(history < 100, "1) $0 - $100", "2) $100 - $200"),
        "history": history,
        "mens": rng.integers(0, 2, n),
        "womens": rng.integers(0, 2, n),
        "zip_code": rng.choice(list(ZIP_LEVELS), n),
        "newbie": rng.integers(0, 2, n),
        "channel": rng.choice(list(CHANNEL_LEVELS), n),
        "segment": seg,
        "visit": visit.astype(int),
        "conversion": conv.astype(int),
        "spend": np.where(conv, np.round(rng.gamma(2.0, 60.0, n), 2), 0.0),
    })


def design_matrix(df: pd.DataFrame) -> np.ndarray:
    """Intercept, standardized numeric features, binary flags and one-hot categoricals."""
    rec = df["recency"].to_numpy(float)
    hist = np.log1p(df["history"].to_numpy(float))
    cols = [
        np.ones(len(df)),
        (rec - rec.mean()) / rec.std(),
        (hist - hist.mean()) / hist.std(),
        df["mens"].to_numpy(float),
        df["womens"].to_numpy(float),
        df["newbie"].to_numpy(float),
    ]
    cols += [(df["zip_code"] == z).to_numpy(float) for z in ZIP_LEVELS[1:]]
    cols += [(df["channel"] == c).to_numpy(float) for c in CHANNEL_LEVELS[1:]]
    return np.column_stack(cols)


# -- evaluation ------------------------------------------------------------------


@dataclass(frozen=True)
class Stat:
    """Point estimate and bootstrap SE. ``se`` is NaN for a ratio whose
    denominator is not reliably positive (see ``_ratio``)."""

    value: float
    se: float

    @property
    def estimable(self) -> bool:
        return bool(np.isfinite(self.value))


# A ratio is reported only if its denominator is estimated precisely enough:
# bootstrap coefficient of variation at most this value. Above it, the ratio's
# sampling distribution has heavy tails and its SE is not meaningful.
_MAX_DENOMINATOR_CV = 0.3


def _stat(v: np.ndarray) -> Stat:
    return Stat(float(v[0]), float(np.std(v[1:], ddof=1)))


def _ratio(num: np.ndarray, den: np.ndarray) -> Stat:
    if den[0] <= 0 or np.std(den[1:], ddof=1) / den[0] > _MAX_DENOMINATOR_CV:
        return Stat(float("nan"), float("nan"))
    return _stat(num / den)


def _weighted_rates(w: np.ndarray, y: np.ndarray, t: np.ndarray, mask: np.ndarray):
    """Treated and control rates in ``mask`` for each bootstrap weight row in ``w``."""
    wt, wc = w[:, mask & t], w[:, mask & ~t]
    rt = wt @ y[mask & t] / wt.sum(axis=1)
    rc = wc @ y[mask & ~t] / wc.sum(axis=1)
    return rt, rc


def targeting_stats(
    score: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    reaches: tuple[float, ...],
    boot_w: np.ndarray,
) -> dict[float, dict[str, Stat]]:
    """For each reach: incremental and credited outcomes per 1,000 targeted,
    credited / incremental, and the share of all incremental outcomes reached.

    The ranking is fixed. SEs come from a Poisson bootstrap over users
    (``boot_w`` is replicates x users, including a first row of ones for the
    point estimate), so they reflect evaluation sampling only.
    """
    t = t.astype(bool)
    ones = np.ones((1, len(y)))
    w = np.vstack([ones, boot_w])
    all_mask = np.ones(len(y), dtype=bool)
    rt_all, rc_all = _weighted_rates(w, y, t, all_mask)
    total = rt_all - rc_all
    out = {}
    for reach in reaches:
        mask = top_mask(score, reach)
        rt, rc = _weighted_rates(w, y, t, mask)
        up = rt - rc
        out[reach] = {
            "uplift": _stat(1000 * up),
            "credited": _stat(1000 * rt),
            "inflation": _ratio(rt, up),
            "share": _stat(reach * up / total),
        }
    return out


def marginal_ratio(
    score: np.ndarray, y: np.ndarray, t: np.ndarray, low: float, high: float,
    boot_w: np.ndarray,
) -> dict[str, Stat]:
    """Average incremental per 1k in the top ``low``, marginal per 1k from ``low``
    to ``high``, and their ratio, with Poisson-bootstrap SEs."""
    t = t.astype(bool)
    w = np.vstack([np.ones((1, len(y))), boot_w])
    a = top_mask(score, low)
    extra = top_mask(score, high) & ~a
    rt_a, rc_a = _weighted_rates(w, y, t, a)
    rt_m, rc_m = _weighted_rates(w, y, t, extra)
    avg, marg = 1000 * (rt_a - rc_a), 1000 * (rt_m - rc_m)
    return {"average": _stat(avg), "marginal": _stat(marg), "ratio": _ratio(marg, avg)}
