"""Propensity vs uplift targeting on the Criteo uplift dataset (a randomized trial).

Loop 4's simulation (case 06) says that when a platform targets by predicted
conversion propensity:

1. platform-credited conversions overstate incremental conversions, and more so
   as targeting concentrates on high-propensity users; and
2. the incremental return of the marginal user is lower than the average, so a
   lift test at one budget overprices the next dollar.

This module checks both claims on public data with a randomized treatment, and
adds a third question: how much incrementality does ranking by propensity give
up compared with ranking by a (simple) uplift model?

Data: Diemert, Betlei, Renaudin, Amini (2018), "A large scale benchmark for
uplift modeling", Criteo AI Lab. v2.1, 13,979,592 rows, 12 anonymized features,
``treatment`` (85% treated, randomized), ``visit`` and ``conversion`` labels.
License CC-BY-NC-SA 4.0. The file is downloaded on first use and is not
committed.

Everything here is intent-to-treat: ``treatment`` means "eligible to be
targeted", not "saw an ad".
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from paid_media.uplift import fit_logistic, predict, top_mask  # noqa: F401

URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
    "criteo-research-uplift-v2.1.csv.gz"
)
SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
FEATURES = tuple(f"f{i}" for i in range(12))
COLUMNS = (*FEATURES, "treatment", "conversion", "visit", "exposure")
N_ROWS = 13_979_592
_CLIP = 8.0


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
        print(f"downloading {URL} (~300 MB) ...")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(dest)
    digest = sha256(dest)
    if digest != SHA256:
        raise ValueError(f"{dest} has SHA-256 {digest}, expected {SHA256}")
    return dest


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype="float32")
    if tuple(df.columns) != COLUMNS:
        raise ValueError(f"unexpected columns: {list(df.columns)}")
    return df


def synthetic_sample(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Small stand-in with the real schema, for CI smoke runs and tests.

    Baseline visit log-odds rise with ``f0`` and the treatment lifts log-odds
    by a constant, so absolute uplift is highest for high-propensity users.
    """
    x = rng.standard_normal((n, len(FEATURES))).astype("float32")
    t = (rng.random(n) < 0.85).astype("float32")
    eta = -3.2 + 1.2 * x[:, 0] + 0.4 * x[:, 1]
    visit = (rng.random(n) < 1 / (1 + np.exp(-(eta + 0.3 * t)))).astype("float32")
    conv = (visit * (rng.random(n) < 0.06 + 0.02 * t)).astype("float32")
    df = pd.DataFrame(x, columns=list(FEATURES))
    df["treatment"], df["conversion"], df["visit"] = t, conv, visit
    df["exposure"] = (t * visit * (rng.random(n) < 0.6)).astype("float32")
    return df


# -- models ----------------------------------------------------------------------


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> Standardizer:
        sd = x.std(axis=0)
        return cls(x.mean(axis=0), np.where(sd > 0, sd, 1.0))

    def design(self, x: np.ndarray) -> np.ndarray:
        """Intercept, standardized features (clipped at +/-8 SD), and their squares."""
        z = np.clip((x - self.mean) / self.sd, -_CLIP, _CLIP)
        return np.column_stack([np.ones(len(z)), z, z**2])






# -- evaluation ------------------------------------------------------------------


@dataclass(frozen=True)
class TargetingRead:
    """Readout for the top-``reach`` share of users ranked by one score.

    Rates are per 1,000 targeted users. ``uplift`` is the treated-minus-control
    rate within the targeted set (intent-to-treat). ``credited`` is the treated
    rate, i.e. what a platform that credits every outcome among treated users
    would report. ``share_of_total`` is the incremental outcomes reached by
    targeting this set, as a share of the incremental outcomes from targeting
    everyone. ``*_se`` come from K random folds of the test set.
    """

    reach: float
    uplift: float
    uplift_se: float
    credited: float
    inflation: float
    inflation_se: float
    share_of_total: float
    share_se: float




def _uplift(y: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    treated = float(y[t].mean())
    return treated - float(y[~t].mean()), treated


def evaluate_targeting(
    score: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    reaches: tuple[float, ...],
    folds: np.ndarray,
) -> list[TargetingRead]:
    """Rank users by ``score`` and read out each reach level on randomized data.

    The ranking is global. Standard errors come from recomputing each statistic
    within each fold (``folds`` holds a fold id per user) and dividing the
    fold-to-fold SD by sqrt(K). They capture sampling noise in the test set,
    not noise in the fitted model.
    """
    t = t.astype(bool)
    k = int(folds.max()) + 1
    total_up, _ = _uplift(y, t)
    fold_total = np.array([_uplift(y[folds == f], t[folds == f])[0] for f in range(k)])
    out = []
    for reach in reaches:
        sel = top_mask(score, reach)
        up, cred = _uplift(y[sel], t[sel])
        per_fold = []
        for f in range(k):
            m = sel & (folds == f)
            u, c = _uplift(y[m], t[m])
            per_fold.append((u, c / u if u > 0 else np.nan, reach * u / fold_total[f]))
        se = np.nanstd(np.array(per_fold), axis=0, ddof=1) / np.sqrt(k)
        out.append(
            TargetingRead(
                reach=reach,
                uplift=1000 * up,
                uplift_se=1000 * float(se[0]),
                credited=1000 * cred,
                inflation=cred / up if up > 0 else float("nan"),
                inflation_se=float(se[1]),
                share_of_total=reach * up / total_up,
                share_se=float(se[2]),
            )
        )
    return out


def marginal_uplift(
    score: np.ndarray, y: np.ndarray, t: np.ndarray, low: float, high: float
) -> tuple[float, float]:
    """Average uplift per 1k in the top ``low`` and per 1k extra users from
    ``low`` to ``high`` reach."""
    t = t.astype(bool)
    a, b = top_mask(score, low), top_mask(score, high)
    extra = b & ~a
    return 1000 * _uplift(y[a], t[a])[0], 1000 * _uplift(y[extra], t[extra])[0]
