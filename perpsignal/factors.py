"""Factor library. Each factor returns a per-bar series in [-1, +1].

Sign convention everywhere: **positive = long bias, negative = short bias.**
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FactorConfig


# ---------------------------------------------------------------------------
# Indicator primitives
# ---------------------------------------------------------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing ≈ ema with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's true range: max(high−low, |high−prevClose|, |low−prevClose|).
    Shared primitive so the DSL, ATR, and ADX all compute TR identically."""
    prev_close = close.shift(1)
    return pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average true range via Wilder smoothing (ema, alpha = 1/period)."""
    return true_range(high, low, close).ewm(alpha=1.0 / period, adjust=False).mean()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
         period: int) -> pd.Series:
    """Rolling VWAP core (typical-price × volume, summed / volume summed) with no
    NaN fill — callers apply their own ffill/fillna policy."""
    typical = (high + low + close) / 3.0
    pv = (typical * volume).rolling(period).sum()
    vsum = volume.rolling(period).sum().replace(0.0, np.nan)
    return pv / vsum


def bb_width(series: pd.Series, period: int, num_std: float) -> pd.Series:
    """Bollinger band width as a fraction of price: (2·num_std·stdev)/price."""
    sd = series.rolling(period).std()
    return (2.0 * num_std * sd) / series.replace(0.0, np.nan)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's ADX. Returns the ADX line (not +DI / -DI)."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    alpha = 1.0 / period
    atr_s = atr(high, low, close, period)
    plus_di = 100.0 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s.replace(0.0, np.nan))
    minus_di = 100.0 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s.replace(0.0, np.nan))
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False).mean().fillna(0.0)


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    r = np.log(close / close.shift(1))
    return r.rolling(window).std()


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window).mean()
    sd = s.rolling(window).std()
    z = (s - m) / sd.replace(0.0, np.nan)
    return z.fillna(0.0)


def clip_tanh(x: pd.Series, scale: float = 1.0) -> pd.Series:
    """Squash to [-1, +1] via tanh."""
    return np.tanh(x.astype(float) * scale)


# ---------------------------------------------------------------------------
# Factor functions — each returns a series in roughly [-1, +1]
# ---------------------------------------------------------------------------
def factor_rsi(close: pd.Series, period: int) -> pd.Series:
    """+1 at deep oversold (RSI=0), -1 at deep overbought (RSI=100), 0 at 50."""
    r = rsi(close, period)
    return ((50.0 - r) / 50.0).clip(-1.0, 1.0)


def factor_ema_cross(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """Normalized EMA spread, squashed to [-1, +1]."""
    diff = (ema(close, fast) - ema(close, slow)) / close
    # Typical magnitude of (ema20-ema50)/close on BTC 1h is small (~0.01).
    # Scale by ~50 so 1% spread → tanh(0.5) ≈ 0.46.
    return clip_tanh(diff, scale=50.0).fillna(0.0)


def factor_macd(close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    """MACD histogram, normalized by price then squashed."""
    _, _, hist = macd(close, fast, slow, signal)
    return clip_tanh(hist / close, scale=200.0).fillna(0.0)


def factor_trend(high: pd.Series, low: pd.Series, close: pd.Series,
                 adx_period: int, ema_trend_period: int) -> pd.Series:
    """ADX-gated trend direction.

    Direction = sign of (close - EMA200) / EMA200 (continuous, squashed).
    Magnitude scaled by ADX strength: weak ADX → near zero.
    """
    e = ema(close, ema_trend_period)
    direction = clip_tanh((close - e) / e, scale=30.0)
    a = adx(high, low, close, adx_period)
    # Map ADX: <15 → 0, >=40 → 1, linear between
    a_scaled = ((a - 15.0) / 25.0).clip(0.0, 1.0)
    return (direction * a_scaled).fillna(0.0)


def factor_volume(close: pd.Series, volume: pd.Series, window: int,
                  smooth: int = 4) -> pd.Series:
    """Volume z-score, signed by the bar's recent price direction.

    We sign by the EMA of returns over the last `smooth` bars rather than the
    single-bar sign, so the factor doesn't flip every bar in chop. Then we EMA-
    smooth the signed value itself.
    """
    z = rolling_zscore(volume, window).clip(-3.0, 3.0) / 3.0
    ret = close.pct_change().fillna(0.0)
    direction = np.sign(ret.ewm(span=smooth, adjust=False).mean()).fillna(0.0)
    signed = z * direction
    return signed.ewm(span=smooth, adjust=False).mean().fillna(0.0)


def factor_oi(close: pd.Series, open_interest: pd.Series, window: int,
              smooth: int = 4) -> pd.Series:
    """OI flow factor.

    Rising OI on up move → new longs entering (+).
    Rising OI on down move → new shorts entering (-).
    NaN where OI unavailable (returns 0). EMA-smoothed to reduce flicker.
    """
    oi = open_interest.astype(float)
    if oi.isna().all():
        return pd.Series(0.0, index=close.index)
    d_oi = oi.diff()
    d_oi_z = rolling_zscore(d_oi, window).clip(-3.0, 3.0) / 3.0
    ret = close.pct_change().fillna(0.0)
    direction = np.sign(ret.ewm(span=smooth, adjust=False).mean()).fillna(0.0)
    signed = d_oi_z * direction
    return signed.ewm(span=smooth, adjust=False).mean().fillna(0.0)


def factor_funding(funding_rate: pd.Series, window: int) -> pd.Series:
    """Funding z-score with **contrarian sign**: very high funding → -1 (longs crowded).

    The combiner gives this factor a *negative* weight in trending regimes (we want
    contrarian tilt to be applied as `weight * factor`); to keep all factor outputs
    on the "+ = long bias" convention, we return -z so weights stay positive in the
    chop regime and the meaning composes cleanly with the rest of the matrix.
    """
    if funding_rate.isna().all():
        return pd.Series(0.0, index=funding_rate.index)
    z = rolling_zscore(funding_rate, window).clip(-3.0, 3.0) / 3.0
    return (-z).fillna(0.0)


# ---------------------------------------------------------------------------
# Extended factor library — Phase 1 audit additions. Each new factor is
# designed so weight=0 leaves it inert; opt-in via REGIME_WEIGHTS or a
# user-published signal's `weights` matrix.
# ---------------------------------------------------------------------------

def factor_slope_regression(close: pd.Series, lookback: int = 20) -> pd.Series:
    """Linear-regression slope of close over `lookback` bars, normalized by
    stdev(close)/stdev(idx) and squashed to [-1, +1].

    Equivalent to corr(close, idx) × stdev(close)/stdev(idx) — the closed-form
    OLS slope — but we use rolling correlation × rolling-stdev so the whole
    series is O(n) instead of running a regression per bar. Sign follows close:
    rising slope → positive.
    """
    idx = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    corr = close.rolling(lookback).corr(idx)
    sd_close = close.rolling(lookback).std()
    sd_idx = idx.rolling(lookback).std().replace(0.0, np.nan)
    slope = corr * sd_close / sd_idx
    # Normalize by price so the factor scale doesn't change with the asset's
    # absolute price. Tanh-squash so extreme breakouts saturate near ±1.
    return clip_tanh((slope / close.replace(0.0, np.nan)).fillna(0.0), scale=2000.0)


def factor_adx_strength(high: pd.Series, low: pd.Series, close: pd.Series,
                        period: int = 14) -> pd.Series:
    """ADX as a direction-AGNOSTIC strength factor mapped to [-1, +1].

    ADX is conventionally a 0..100 strength reading. We map: ADX≤15 → -1
    (chop), ADX=25 → 0 (neutral), ADX≥40 → +1 (strong trend). Useful as a
    gate — multiply alongside another directional factor in the weights
    matrix, e.g. `slope_regression × adx_strength` to dampen slope signal
    during chop.
    """
    a = adx(high, low, close, period)
    # Linear ramp 15→40 mapped to [-1, +1].
    return ((a - 27.5) / 12.5).clip(-1.0, 1.0).fillna(0.0)


def factor_rsi_delta(close: pd.Series, period: int = 14) -> pd.Series:
    """1-bar change in RSI, scaled by /50 to land in roughly [-1, +1].

    Captures momentum *acceleration* rather than absolute RSI level. Positive
    delta = momentum building, negative = fading. Pair with a regime gate
    when you only want acceleration signals during trends.
    """
    r = rsi(close, period)
    return ((r - r.shift(1)) / 50.0).clip(-1.0, 1.0).fillna(0.0)


def factor_volume_zscore(volume: pd.Series, window: int = 48) -> pd.Series:
    """Raw volume z-score (unsigned). Differs from `factor_volume` which
    signs the z by recent price direction. Use this when you want pure
    volume-pop detection without imposing a directional bias — e.g.
    multiply with another factor in the combiner to gate "trade only when
    volume is elevated."
    """
    z = rolling_zscore(volume, window).clip(-3.0, 3.0) / 3.0
    return z.fillna(0.0)


def factor_atr_stability(high: pd.Series, low: pd.Series, close: pd.Series,
                         period: int = 14) -> pd.Series:
    """Volatility-stability factor: 1 - |ΔATR| / max(ATR, ε), then rescaled
    so 1.0 (stable vol) maps to +1 and erratic conditions map toward -1.

    Useful for "only trade in clean trends" gating — multiply with a
    directional factor and the signal mutes when ATR is whipsawing.
    """
    atr_s = atr(high, low, close, period)
    eps = atr_s.median() * 0.0001 if atr_s.median() > 0 else 1e-6
    d_atr = atr_s.diff().abs()
    stability = 1.0 - (d_atr / atr_s.clip(lower=eps))
    # Map [0, 1] stability → [-1, +1] so weight signs compose normally.
    return (stability.clip(0.0, 1.0) * 2.0 - 1.0).fillna(0.0)


def factor_bb_width(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Bollinger band width relative to its rolling baseline, squashed.

    Positive when bands are *wider* than recent average (expansion, often
    precedes trend continuation); negative when bands are tighter than
    average (squeeze, often precedes breakout). Range-agnostic — normalized
    by the rolling median of width so it works on any asset/TF.
    """
    width = bb_width(close, period, num_std)
    baseline = width.rolling(period * 4).median()
    relative = (width - baseline) / baseline.replace(0.0, np.nan)
    # ±50% relative width pings ±1 after squashing.
    return clip_tanh(relative.fillna(0.0), scale=2.0)


def factor_vwap_distance(high: pd.Series, low: pd.Series, close: pd.Series,
                          volume: pd.Series, period: int = 96) -> pd.Series:
    """Distance from rolling VWAP, normalized by price and squashed.

    Positive = price above VWAP (buyers in control on average), negative
    = below. Resets via a rolling window rather than a session anchor
    because the engine has no session concept across all TFs.
    """
    vwap_s = vwap(high, low, close, volume, period)
    dist = (close - vwap_s) / vwap_s.replace(0.0, np.nan)
    # ±2% from VWAP saturates the factor — most assets trade close enough
    # that further-out moves should clip rather than dominate.
    return clip_tanh(dist.fillna(0.0), scale=50.0)


# ---------------------------------------------------------------------------
# Compute all factors at once
# ---------------------------------------------------------------------------
def compute_all_factors(df: pd.DataFrame, cfg: FactorConfig | None = None) -> pd.DataFrame:
    cfg = cfg or FactorConfig()
    out = pd.DataFrame(index=df.index)
    out["rsi"] = factor_rsi(df["close"], cfg.rsi_period)
    out["ema_cross"] = factor_ema_cross(df["close"], cfg.ema_fast, cfg.ema_slow)
    out["macd"] = factor_macd(df["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    out["trend"] = factor_trend(df["high"], df["low"], df["close"], cfg.adx_period, cfg.ema_trend)
    out["volume"] = factor_volume(df["close"], df["volume"], cfg.vol_z_window)
    out["oi"] = factor_oi(df["close"], df.get("open_interest", pd.Series(index=df.index, dtype=float)), cfg.oi_z_window)
    out["funding"] = factor_funding(df.get("funding_rate", pd.Series(index=df.index, dtype=float)), cfg.funding_z_window)
    # Phase 1 audit additions. Each defaults to 0 weight in REGIME_WEIGHTS
    # so existing signals are unaffected; authors opt in by giving them a
    # non-zero weight in their published signal definition.
    out["slope_regression"] = factor_slope_regression(df["close"], cfg.slope_lookback)
    out["adx_strength"] = factor_adx_strength(df["high"], df["low"], df["close"], cfg.adx_period)
    out["rsi_delta"] = factor_rsi_delta(df["close"], cfg.rsi_period)
    out["volume_zscore"] = factor_volume_zscore(df["volume"], cfg.vol_z_window)
    out["atr_stability"] = factor_atr_stability(df["high"], df["low"], df["close"], cfg.atr_period)
    out["bb_width"] = factor_bb_width(df["close"], cfg.bb_period, cfg.bb_num_std)
    out["vwap_distance"] = factor_vwap_distance(df["high"], df["low"], df["close"], df["volume"], cfg.vwap_period)
    return out
