from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Allow override via env var for serverless environments (Vercel /tmp, etc.).
DATA_DIR = Path(os.environ.get("PERPSIGNAL_DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Costs:
    taker_fee_bps: float = 4.0      # 0.04% per side (Binance perp taker)
    slippage_bps: float = 2.0       # 0.02% per side
    # Funding is paid every 8h on Binance perp; cost is realized via funding_rate series.


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str = "BTCUSDT"
    interval: str = "4h"            # used only when auto-TF is off
    costs: Costs = field(default_factory=Costs)
    # Position sign in {-1, 0, +1}; flat-zones reduce churn.
    long_threshold: float = 0.20
    short_threshold: float = -0.20
    # Min bars to hold once entered (reduces flip-flop). 0 disables.
    # NOTE: min-hold only blocks *signal-driven* exits; TP/SL hits always fire.
    min_hold_bars: int = 8


@dataclass(frozen=True)
class RiskConfig:
    """Risk profile that drives dynamic TF selection and backtest exits."""
    leverage: float = 3.0
    take_profit_pct: float | None = 0.04    # 4% TP from entry; None disables
    stop_loss_pct: float | None = 0.015     # 1.5% SL from entry; None disables
    # When True, primary TF is picked so ATR(14)≈sl_pct*sl_atr_target.
    auto_tf: bool = True
    sl_atr_target: float = 1.0              # SL ≈ 1× ATR on chosen TF


# Candidate TFs in increasing order of duration
CANDIDATE_TFS = ("5m", "15m", "1h", "4h", "1d", "3d")

# For a chosen primary TF, which TFs serve as the faster (entry timing /
# confirmation) and slower (trend / regime) tiers. None = same as primary.
TF_TIERS: dict[str, dict[str, str | None]] = {
    "5m":  {"faster": None,  "primary": "5m",  "slower": "15m"},
    "15m": {"faster": "5m",  "primary": "15m", "slower": "1h"},
    "1h":  {"faster": "15m", "primary": "1h",  "slower": "4h"},
    "4h":  {"faster": "1h",  "primary": "4h",  "slower": "1d"},
    "1d":  {"faster": "4h",  "primary": "1d",  "slower": None},
    "3d":  {"faster": "1d",  "primary": "3d",  "slower": None},
}


# Lookback for indicators
@dataclass(frozen=True)
class FactorConfig:
    rsi_period: int = 14
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_period: int = 14
    vol_z_window: int = 48          # ~2 days on 1h
    oi_z_window: int = 48
    funding_z_window: int = 24 * 7  # ~1 week of 1h bars
    realized_vol_window: int = 24 * 7
    # Phase 1 audit additions — periods for the new factor library.
    # Tuned conservatively: each is a sensible default that the AI can
    # rely on without per-signal tuning, since the published weight
    # matrix is what users actually adjust.
    slope_lookback: int = 20        # bars used for linear-regression slope
    atr_period: int = 14            # ATR period for stability factor
    bb_period: int = 20             # Bollinger period
    bb_num_std: float = 2.0         # std multiplier for Bollinger bands
    vwap_period: int = 96           # rolling VWAP window (~4d on 1h, ~1d on 15m)


# Regime-conditional weight matrix. Rows = regimes, cols = factors.
# Factors are normalized to [-1, +1] and weights need not sum to 1.
#
# The "_slow" variants of RSI / MACD / EMA-cross are sourced from the
# slower TF tier (e.g. 4h on a 1h primary; see TF_TIERS) and aligned
# onto the primary index without look-ahead. They exist so an author
# can express genuinely multi-timeframe rules — "long when the 4h RSI
# is deeply oversold, even if we're trading 1h bars" — without
# changing the primary execution TF. New factors default to 0 weight
# so legacy signals and the canonical Perps DNA fit are unaffected.
FACTOR_ORDER = (
    "rsi",            # primary-TF RSI: mean-reverting in chop, trend-confirming at extremes
    "ema_cross",      # primary-TF EMA spread, sign = trend direction
    "macd",           # primary-TF MACD histogram
    "trend",          # ADX-gated EMA200 slope (already sourced from slower TF)
    "volume",         # volume z-score (confirmation, faster TF when available)
    "oi",             # OI delta (positioning flow, faster TF when available)
    "funding",        # funding z (contrarian when extreme)
    "rsi_slow",       # slower-TF RSI — same definition as `rsi`, sourced from the slower tier
    "ema_cross_slow", # slower-TF EMA spread
    "macd_slow",      # slower-TF MACD histogram
    # Phase 1 audit additions. Default weight 0 across all regimes so
    # the canonical Perps DNA fit + every legacy signal is unaffected;
    # authors opt in by giving them non-zero weight in their published
    # signal. parse_signal() reads each row with .get(f, 0.0) so old
    # records still load fine.
    "slope_regression",  # OLS slope of close, normalized by stdev — directional momentum
    "adx_strength",      # ADX mapped to ±1: chop ↔ strong trend (direction-agnostic, use as gate)
    "rsi_delta",         # 1-bar change in RSI / 50 — momentum acceleration
    "volume_zscore",     # raw unsigned volume z (gate factor)
    "atr_stability",     # 1 − |ΔATR|/ATR rescaled — clean vol vs erratic
    "bb_width",          # Bollinger width relative to baseline (expansion / squeeze)
    "vwap_distance",     # close vs rolling VWAP, normalized by price
)

# Hand-tuned starting weights; the README of the design doc.
# rsi sign convention: +1 = oversold (long bias), -1 = overbought (short bias).
# The *_slow factors default to 0 in the canonical matrix — authors opt in.
REGIME_WEIGHTS = {
    "trend_up": {
        "rsi":            0.05,  # mild dip-buy bias when oversold in uptrend
        "ema_cross":      0.40,
        "macd":           0.30,
        "trend":          0.40,
        "volume":         0.10,
        "oi":             0.15,
        "funding":        0.10,  # funding inverted: high funding → negative; we want short bias when overly crowded
        "rsi_slow":       0.00,
        "ema_cross_slow": 0.00,
        "macd_slow":      0.00,
        # Phase 1 additions default to 0 — keeps the canonical fit identical
        # to pre-extension behavior. Authors opt in via published signals.
        "slope_regression": 0.00,
        "adx_strength":     0.00,
        "rsi_delta":        0.00,
        "volume_zscore":    0.00,
        "atr_stability":    0.00,
        "bb_width":         0.00,
        "vwap_distance":    0.00,
    },
    "trend_down": {
        "rsi":            0.05,
        "ema_cross":      0.40,
        "macd":           0.30,
        "trend":          0.40,
        "volume":         0.10,
        "oi":             0.15,
        "funding":        0.10,
        "rsi_slow":       0.00,
        "ema_cross_slow": 0.00,
        "macd_slow":      0.00,
        "slope_regression": 0.00,
        "adx_strength":     0.00,
        "rsi_delta":        0.00,
        "volume_zscore":    0.00,
        "atr_stability":    0.00,
        "bb_width":         0.00,
        "vwap_distance":    0.00,
    },
    "chop": {
        "rsi":            0.50,  # mean revert RSI hardest in chop
        "ema_cross":      0.00,
        "macd":           0.05,
        "trend":          0.00,
        "volume":         0.05,
        "oi":             0.05,
        "funding":        0.30,  # funding extremes meaningful in chop
        "rsi_slow":       0.00,
        "ema_cross_slow": 0.00,
        "macd_slow":      0.00,
        "slope_regression": 0.00,
        "adx_strength":     0.00,
        "rsi_delta":        0.00,
        "volume_zscore":    0.00,
        "atr_stability":    0.00,
        "bb_width":         0.00,
        "vwap_distance":    0.00,
    },
}
