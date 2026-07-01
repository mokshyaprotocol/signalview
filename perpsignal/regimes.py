"""3-regime classifier: trend_up / trend_down / chop.

Decision rule per bar:
  - Trend strength = ADX value and slope of long-EMA.
  - If ADX >= 25 and EMA200 slope > 0 (over last N bars): trend_up
  - If ADX >= 25 and EMA200 slope < 0: trend_down
  - Else: chop

We then smooth the regime label with a short min-dwell to avoid flicker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FactorConfig
from .factors import adx, ema

REGIMES = ("trend_up", "trend_down", "chop")
_REGIME_TO_INT = {"trend_up": 1, "trend_down": -1, "chop": 0}
_INT_TO_REGIME = {v: k for k, v in _REGIME_TO_INT.items()}


def classify(df: pd.DataFrame, cfg: FactorConfig | None = None,
             adx_threshold: float = 25.0,
             slope_lookback: int = 24,
             min_dwell: int = 6) -> pd.Series:
    """Return a Series of regime labels aligned with df.index."""
    cfg = cfg or FactorConfig()
    if df.empty:
        # A market can have NO bars at all on the slower regime TF (e.g. a
        # days-old builder-dex listing with zero 1d candles). Returning an
        # empty series lets the caller surface a clean "not enough history"
        # instead of an IndexError out of the dwell smoother below.
        return pd.Series(dtype=object)
    e = ema(df["close"], cfg.ema_trend)
    slope = e.diff(slope_lookback) / e.shift(slope_lookback)
    a = adx(df["high"], df["low"], df["close"], cfg.adx_period)

    raw = pd.Series("chop", index=df.index)
    trending = a >= adx_threshold
    raw[trending & (slope > 0)] = "trend_up"
    raw[trending & (slope < 0)] = "trend_down"

    # Min-dwell smoothing: a new regime label must persist for `min_dwell` bars
    # before we accept the switch — otherwise we keep the previous regime.
    if min_dwell <= 1:
        return raw

    arr = raw.map(_REGIME_TO_INT).to_numpy()
    out = arr.copy()
    cur = arr[0]
    run_len = 1
    for i in range(1, len(arr)):
        if arr[i] == cur:
            run_len += 1
            out[i] = cur
        else:
            # candidate switch — peek forward to see if run sustains
            j = i
            new_val = arr[i]
            while j < len(arr) and arr[j] == new_val:
                j += 1
            if (j - i) >= min_dwell:
                cur = new_val
                out[i] = cur
                run_len = 1
            else:
                out[i] = cur
    return pd.Series(out, index=df.index).map(_INT_TO_REGIME)


def regime_stats(regimes: pd.Series) -> pd.DataFrame:
    counts = regimes.value_counts().reindex(REGIMES, fill_value=0)
    pct = (counts / len(regimes) * 100.0).round(1)
    return pd.DataFrame({"bars": counts, "pct": pct})
