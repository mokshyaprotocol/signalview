"""Walk-forward weight optimization via random search.

Why knob parameterization (and not raw 21 weights)?
  The full regime × factor weight matrix is 21-dimensional, which is too large
  for cheap random search to cover. We reduce it to 14 knobs that map onto a
  21-element weight matrix via fixed within-group ratios:

    Per regime (3 regimes):
      trend_strength  ∈ [0, 1]    → split equally across ema_cross, macd, trend
      rsi_weight      ∈ [-0.5, 0.5]
      flow_weight     ∈ [0, 0.5]  → split equally across volume, oi
      funding_weight  ∈ [-0.5, 0.5]
    Global:
      long_threshold, short_threshold

  Total: 3 × 4 + 2 = 14 dims. Far more tractable, and the constraint that
  trend-family weights stay tied is structurally sensible.

Scoring:
  Sharpe on the in-sample window, with a minimum-trades floor (else -inf).
  We multiply by a `n_trades` factor at the low end to penalize configs that
  barely trade, which can otherwise post huge Sharpe by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from .backtest import run as run_backtest
from .config import BacktestConfig, RiskConfig
from .signal import combine, discretize


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WeightKnobs:
    """14-dimensional parameterization of regime × factor weights + thresholds."""
    # trend_up
    up_trend: float
    up_rsi: float
    up_flow: float
    up_funding: float
    # trend_down
    dn_trend: float
    dn_rsi: float
    dn_flow: float
    dn_funding: float
    # chop
    cp_trend: float
    cp_rsi: float
    cp_flow: float
    cp_funding: float
    # thresholds
    long_thr: float
    short_thr: float


def knobs_to_weights(k: WeightKnobs) -> dict[str, dict[str, float]]:
    def regime_block(trend_w: float, rsi_w: float, flow_w: float, funding_w: float) -> dict[str, float]:
        return {
            "rsi":        rsi_w,
            "ema_cross":  trend_w / 3.0,
            "macd":       trend_w / 3.0,
            "trend":      trend_w / 3.0,
            "volume":     flow_w / 2.0,
            "oi":         flow_w / 2.0,
            "funding":    funding_w,
        }
    return {
        "trend_up":   regime_block(k.up_trend, k.up_rsi, k.up_flow, k.up_funding),
        "trend_down": regime_block(k.dn_trend, k.dn_rsi, k.dn_flow, k.dn_funding),
        "chop":       regime_block(k.cp_trend, k.cp_rsi, k.cp_flow, k.cp_funding),
    }


# ---------------------------------------------------------------------------
# Priors (uniform within these bounds)
KNOB_BOUNDS = {
    "up_trend":   (0.0, 1.0),
    "up_rsi":     (-0.3, 0.3),
    "up_flow":    (0.0, 0.5),
    "up_funding": (-0.3, 0.3),
    "dn_trend":   (0.0, 1.0),
    "dn_rsi":     (-0.3, 0.3),
    "dn_flow":    (0.0, 0.5),
    "dn_funding": (-0.3, 0.3),
    "cp_trend":   (0.0, 0.3),
    "cp_rsi":     (0.0, 0.7),       # chop should rely on RSI
    "cp_flow":    (0.0, 0.3),
    "cp_funding": (-0.5, 0.5),
    "long_thr":   (0.10, 0.40),
    "short_thr":  (-0.40, -0.10),
}


def sample_knobs(rng: np.random.Generator) -> WeightKnobs:
    vals = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in KNOB_BOUNDS.items()}
    return WeightKnobs(**vals)


# ---------------------------------------------------------------------------
def _score(metrics: dict, min_trades: int = 5) -> float:
    n = metrics.get("trades", 0)
    if n < min_trades:
        return -np.inf
    sh = metrics.get("sharpe", float("nan"))
    if not np.isfinite(sh):
        return -np.inf
    # Light penalty for under-traded configs so we prefer ones with sample weight.
    floor = min(n, 30)
    return float(sh * np.sqrt(floor / 30.0))


def evaluate(df: pd.DataFrame, factors: pd.DataFrame, regimes: pd.Series,
             knobs: WeightKnobs, bt_cfg: BacktestConfig, risk: RiskConfig,
             bars_per_year: int, metrics_only: bool = False):
    weights = knobs_to_weights(knobs)
    score_series = combine(factors, regimes, weights)
    cfg = replace(bt_cfg, long_threshold=knobs.long_thr, short_threshold=knobs.short_thr)
    signal = discretize(score_series, cfg)
    # metrics_only (from the random_search trial loop): only the score (Sharpe +
    # trade count) is read, so skip the output frame + benchmark. Callers that
    # need the equity curve / benchmark (walk_forward's test fold) leave it False.
    res = run_backtest(df, signal, cfg, risk, bars_per_year=bars_per_year,
                       metrics_only=metrics_only)
    return _score(res.metrics), res


def random_search(df: pd.DataFrame, factors: pd.DataFrame, regimes: pd.Series,
                  bt_cfg: BacktestConfig, risk: RiskConfig, bars_per_year: int,
                  n_trials: int = 300, seed: int = 0) -> tuple[WeightKnobs, float]:
    rng = np.random.default_rng(seed)
    best_score = -np.inf
    best_knobs = None
    for _ in range(n_trials):
        knobs = sample_knobs(rng)
        score, _ = evaluate(df, factors, regimes, knobs, bt_cfg, risk, bars_per_year,
                            metrics_only=True)
        if score > best_score:
            best_score = score
            best_knobs = knobs
    return best_knobs, best_score


# ---------------------------------------------------------------------------
@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_knobs: WeightKnobs
    train_score: float
    test_metrics: dict
    test_bench: dict
    test_equity: pd.Series


def walk_forward(df: pd.DataFrame, factors: pd.DataFrame, regimes: pd.Series,
                 bt_cfg: BacktestConfig, risk: RiskConfig, bars_per_year: int,
                 train_bars: int, test_bars: int, step_bars: int,
                 n_trials: int = 300, seed: int = 0,
                 verbose: bool = True) -> list[Fold]:
    """Roll a train/test split across the dataset, fitting on train + scoring OOS on test."""
    n = len(df)
    folds: list[Fold] = []
    fold_i = 0
    cursor = 0
    while cursor + train_bars + test_bars <= n:
        train_slice = slice(cursor, cursor + train_bars)
        test_slice = slice(cursor + train_bars, cursor + train_bars + test_bars)

        df_tr = df.iloc[train_slice]
        f_tr = factors.iloc[train_slice]
        r_tr = regimes.iloc[train_slice]
        df_te = df.iloc[test_slice]
        f_te = factors.iloc[test_slice]
        r_te = regimes.iloc[test_slice]

        best_knobs, train_score = random_search(df_tr, f_tr, r_tr, bt_cfg, risk,
                                                bars_per_year, n_trials=n_trials,
                                                seed=seed + fold_i)
        if best_knobs is None:
            if verbose:
                print(f"  fold {fold_i}: no viable config in train — skipping")
            cursor += step_bars
            fold_i += 1
            continue

        # Apply best to test
        _, test_res = evaluate(df_te, f_te, r_te, best_knobs, bt_cfg, risk, bars_per_year)

        folds.append(Fold(
            train_start=df_tr.index[0], train_end=df_tr.index[-1],
            test_start=df_te.index[0], test_end=df_te.index[-1],
            best_knobs=best_knobs, train_score=train_score,
            test_metrics=test_res.metrics, test_bench=test_res.benchmark,
            test_equity=test_res.df["equity"],
        ))
        if verbose:
            print(f"  fold {fold_i}: train [{df_tr.index[0].date()} → {df_tr.index[-1].date()}] "
                  f"score={train_score:5.2f}  test [{df_te.index[0].date()} → {df_te.index[-1].date()}] "
                  f"ret={test_res.metrics['total_return'] * 100:+6.2f}%  "
                  f"sharpe={test_res.metrics['sharpe']:5.2f}  trades={test_res.metrics['trades']}")
        cursor += step_bars
        fold_i += 1
    return folds


# ---------------------------------------------------------------------------
def aggregate_oos(folds: list[Fold], bars_per_year: int) -> dict:
    """Stitch OOS test equities into a single series; report aggregate metrics."""
    if not folds:
        return {}
    # Compute per-fold per-bar returns from each fold's equity
    rets = []
    for f in folds:
        e = f.test_equity
        r = e.pct_change().fillna(e.iloc[0] - 1.0)
        # First bar return: equity[0] / 1.0 - 1
        r.iloc[0] = e.iloc[0] - 1.0
        rets.append(r)
    full = pd.concat(rets).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    eq = (1.0 + full).cumprod()
    years = len(full) / bars_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and eq.iloc[-1] > 0 else float("nan")
    mu = full.mean() * bars_per_year
    sd = full.std() * np.sqrt(bars_per_year)
    sharpe = mu / sd if sd > 0 else float("nan")
    peak = eq.cummax()
    dd = (eq / peak.replace(0, np.nan)) - 1.0
    return {
        "bars": len(full),
        "folds": len(folds),
        "total_return": float(eq.iloc[-1] - 1.0),
        "cagr": float(cagr) if not (isinstance(cagr, float) and np.isnan(cagr)) else float("nan"),
        "sharpe": float(sharpe) if not np.isnan(sharpe) else float("nan"),
        "max_drawdown": float(dd.min()) if not dd.empty else float("nan"),
        "equity": eq,
    }
