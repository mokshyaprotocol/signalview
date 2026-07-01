"""Multi-timeframe utilities.

Two responsibilities:
  1. Dynamically pick the *primary* TF given a target stop-loss distance.
     Heuristic: pick the candidate TF whose typical ATR(14)/close ratio is
     closest to sl_pct * sl_atr_target. Idea — if the SL is 1× ATR on the
     chosen TF, random bar noise won't routinely take us out.
  2. Build a multi-TF feature stack aligned on the primary TF index, without
     look-ahead. Slower-TF data is shifted forward by its own bar duration
     before being forward-filled onto the primary index, so a slower bar
     only contributes after it has fully closed.
"""

from __future__ import annotations

import collections
import concurrent.futures
import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .config import CANDIDATE_TFS, TF_TIERS, FactorConfig
from .data import _INTERVAL_MS, build_dataset, fetch_klines
from .factors import compute_all_factors
from .regimes import classify

# In-process cache of built MTF feature stacks. The stack depends only on
# (symbol, primary_tf, window, factor cfg, include_oi) — NOT on a signal's
# weights/thresholds — so every signal on the same market+window rebuilds the
# exact same 5–10s feature stack. Caching it lets the warm cron re-evaluating a
# batch of BTC signals, and concurrent publishes on the same market, share one
# build on a warm container. Keyed by value; copies are returned so a consumer
# mutating its frame can never corrupt the cached entry. `end` is hourly-snapped
# by callers, so a new hour naturally yields a fresh key (no staleness within the
# cap). Small LRU cap keeps serverless memory bounded.
_MTF_CACHE: "collections.OrderedDict[str, tuple]" = collections.OrderedDict()
_MTF_CACHE_MAX = 8
# agent_tick fans build_mtf_features across up to 8 threads; the OrderedDict
# insert+move_to_end+evict sequence isn't atomic, so concurrent same-key builds
# could KeyError or over-evict. Guard only the cache touch points (the heavy 5-10s
# build stays OUTSIDE the lock, so builds still run concurrently).
_MTF_LOCK = threading.Lock()


def _mtf_cache_key(symbol, primary_tf, start, end, cfg, include_oi) -> str:
    return "|".join([
        str(symbol), str(primary_tf), start.isoformat(), end.isoformat(),
        "1" if include_oi else "0", repr(sorted(vars(cfg).items())),
    ])


def _copy_mtf(result: tuple) -> tuple:
    df, comp, reg = result
    return df.copy(), comp.copy(), reg.copy()


def clear_mtf_cache() -> None:
    """Drop all cached feature stacks (used by tests)."""
    with _MTF_LOCK:
        _MTF_CACHE.clear()

# 5-minute candidates allow finer TF picks if added; CANDIDATE_TFS is the active set.


# ---------------------------------------------------------------------------
def estimate_tf_volatility(symbol: str, tf: str, days: int = 30) -> float:
    """Return the median (ATR(14) / close) for a recent sample on `tf`."""
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    df = fetch_klines(symbol, tf, start, end)
    if df.empty:
        return float("nan")
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    ratio = (atr / df["close"]).dropna()
    return float(ratio.median()) if len(ratio) else float("nan")


def pick_primary_tf(symbol: str, sl_pct: float, sl_atr_target: float = 1.0,
                    candidates: tuple[str, ...] = CANDIDATE_TFS,
                    sample_days: int = 30) -> tuple[str, dict[str, float]]:
    """Pick TF whose typical ATR/close ratio is closest to sl_pct * sl_atr_target."""
    vols = {tf: estimate_tf_volatility(symbol, tf, days=sample_days) for tf in candidates}
    target = sl_pct * sl_atr_target
    valid = {tf: v for tf, v in vols.items() if v == v}  # drop NaN
    if not valid:
        raise RuntimeError(f"could not estimate volatility for any of {candidates}")
    best = min(valid.keys(), key=lambda tf: abs(valid[tf] - target))
    return best, vols


# ---------------------------------------------------------------------------
def _align_no_leak(primary_index: pd.DatetimeIndex, other: pd.DataFrame | pd.Series,
                   other_tf: str) -> pd.DataFrame | pd.Series:
    """Reindex other-TF data onto `primary_index` with no look-ahead.

    Shifts other's open_time-indexed values forward by other_tf's duration so
    a bar with open_time T becomes available only at T + duration_other (its
    close time). Then forward-fills onto primary_index.
    """
    if other is None or len(other) == 0:
        empty = pd.DataFrame(0.0, index=primary_index, columns=getattr(other, "columns", []))
        return empty if isinstance(other, pd.DataFrame) else pd.Series(0.0, index=primary_index)
    duration = pd.Timedelta(milliseconds=_INTERVAL_MS[other_tf])
    shifted = other.copy()
    shifted.index = shifted.index + duration
    aligned = shifted.reindex(primary_index, method="pad")
    return aligned


# ---------------------------------------------------------------------------
def build_mtf_features(symbol: str, primary_tf: str, start: datetime, end: datetime,
                       cfg: FactorConfig | None = None,
                       include_oi: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build the multi-TF factor stack.

    The three TF datasets (faster / primary / slower) are fetched **in
    parallel** — each is independent and pure I/O. Combined with the in-build
    parallelisation of klines/funding/OI inside `build_dataset`, a cache-miss
    request fires up to 9 simultaneous Binance round-trips and waits on the
    slowest one rather than serializing them.

    `include_oi=False` propagates to skip OI fetches entirely.

    Returns:
      df_primary  — primary-TF OHLCV/funding/OI (used by the backtester)
      factors     — composite factor frame at primary index, factor columns
                    sourced from the most appropriate TF
      regimes     — regime label at primary index, classified on the slower TF
    """
    cfg = cfg or FactorConfig()
    # Serve a copy of the cached stack if this exact (market, window, cfg) was
    # built recently on this container.
    key = _mtf_cache_key(symbol, primary_tf, start, end, cfg, include_oi)
    with _MTF_LOCK:
        cached = _MTF_CACHE.get(key)
        if cached is not None:
            _MTF_CACHE.move_to_end(key)
    if cached is not None:
        return _copy_mtf(cached)

    tiers = TF_TIERS[primary_tf]
    faster = tiers["faster"]
    slower = tiers["slower"]

    # Pad: enough warmup for the slowest TF's EMA200 (~200 bars × slowest duration)
    pad_days = 60 if slower in (None, "1d") else 30
    pad_start = start - timedelta(days=pad_days)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        f_primary = ex.submit(build_dataset, symbol, primary_tf, pad_start, end, True, include_oi)
        f_slower = ex.submit(build_dataset, symbol, slower, pad_start, end, True, include_oi) if slower else None
        f_faster = ex.submit(build_dataset, symbol, faster, pad_start, end, True, include_oi) if faster else None
        df_primary = f_primary.result()
        df_slower = f_slower.result() if f_slower is not None else df_primary
        df_faster = f_faster.result() if f_faster is not None else df_primary

    f_primary = compute_all_factors(df_primary, cfg)
    f_slower = compute_all_factors(df_slower, cfg)
    f_faster = compute_all_factors(df_faster, cfg)

    composite = pd.DataFrame(index=df_primary.index)
    # Primary TF: oscillator-style factors that need decision-frequency resolution
    composite["rsi"] = f_primary["rsi"]
    composite["ema_cross"] = f_primary["ema_cross"]
    composite["macd"] = f_primary["macd"]
    composite["funding"] = f_primary["funding"]
    # Slower TF: trend context — aligned with no leak
    if slower and slower != primary_tf:
        composite["trend"] = _align_no_leak(df_primary.index, f_slower["trend"], slower).fillna(0.0)
    else:
        composite["trend"] = f_primary["trend"]
    # Slower TF — oscillator variants. Same factor definitions as the
    # primary versions above (RSI / EMA spread / MACD hist), but
    # computed on the slower bars and aligned forward without
    # look-ahead. Lets multi-TF rules like "long when 4h RSI is deeply
    # oversold even though we execute on 1h" become first-class
    # without conflating with the primary-TF RSI used for entry timing.
    # When the slower tier is the same as primary (1d has no slower)
    # we fall back to the primary series so the column always exists.
    if slower and slower != primary_tf:
        composite["rsi_slow"] = _align_no_leak(df_primary.index, f_slower["rsi"], slower).fillna(0.0)
        composite["ema_cross_slow"] = _align_no_leak(df_primary.index, f_slower["ema_cross"], slower).fillna(0.0)
        composite["macd_slow"] = _align_no_leak(df_primary.index, f_slower["macd"], slower).fillna(0.0)
    else:
        composite["rsi_slow"] = f_primary["rsi"]
        composite["ema_cross_slow"] = f_primary["ema_cross"]
        composite["macd_slow"] = f_primary["macd"]
    # Faster TF: flow factors — average over the faster bars within each primary bar
    # so we get a confirmation signal from finer granularity without daily flicker.
    if faster and faster != primary_tf:
        composite["volume"] = _align_no_leak(df_primary.index, f_faster["volume"], faster).fillna(0.0)
        composite["oi"] = _align_no_leak(df_primary.index, f_faster["oi"], faster).fillna(0.0)
    else:
        composite["volume"] = f_primary["volume"]
        composite["oi"] = f_primary["oi"]
    # Phase 1 audit factors. All sourced from the primary TF because they
    # encode primary-TF behaviour the author wants to act on at execution
    # frequency (slope, vol regime, ATR stability). If an author wants a
    # slower-TF variant later we can mirror the *_slow pattern.
    for col in ("slope_regression", "adx_strength", "rsi_delta",
                "volume_zscore", "atr_stability", "bb_width", "vwap_distance"):
        composite[col] = f_primary[col]

    # Regime on slower TF for stability; align to primary
    regimes_src = classify(df_slower, cfg) if (slower and slower != primary_tf) else classify(df_primary, cfg)
    if slower and slower != primary_tf:
        # regimes is categorical Series; align by converting to int code
        code = regimes_src.map({"trend_up": 1, "trend_down": -1, "chop": 0}).astype(float)
        aligned = _align_no_leak(df_primary.index, code, slower)
        regimes = aligned.round().fillna(0.0).map({1.0: "trend_up", -1.0: "trend_down", 0.0: "chop"})
    else:
        regimes = regimes_src.reindex(df_primary.index, method="pad").fillna("chop")

    result = (df_primary, composite, regimes)
    with _MTF_LOCK:
        _MTF_CACHE[key] = result
        _MTF_CACHE.move_to_end(key)
        while len(_MTF_CACHE) > _MTF_CACHE_MAX:
            _MTF_CACHE.popitem(last=False)
    return _copy_mtf(result)
