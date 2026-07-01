"""Regime-conditional signal combiner.

Inputs:
  - factors: DataFrame with columns from FACTOR_ORDER, each in [-1, +1]
  - regimes: Series of regime labels aligned with factors.index
  - weights: dict[regime][factor] = float (from config.REGIME_WEIGHTS)

Output:
  - score: continuous signal in roughly [-1, +1]
  - position: discretized to {-1, 0, +1} via long/short thresholds and optional min-hold
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FACTOR_ORDER, REGIME_WEIGHTS, BacktestConfig


def combine(factors: pd.DataFrame, regimes: pd.Series,
            weights: dict[str, dict[str, float]] | None = None,
            score_expr: str | None = None,
            frame: pd.DataFrame | None = None) -> pd.Series:
    """Two paths:

    1. Legacy weights matrix (default): `score = Σ weight[regime,factor]
       × factor`, clipped to [-1, +1]. The combine of `regimes × weights
       × factors` is the original Perps DNA design.

    2. DSL score expression (Phase 2): a string in the indicator-
       expression DSL is evaluated against the OHLCV `frame` and used as
       the score directly. Regime gating is the author's responsibility
       inside the expression (e.g. `slope(close, 20) * (adx(14)/50)`).

    When `score_expr` is non-empty AND `frame` is provided, route through
    perpsignal.dsl.evaluate. Otherwise fall back to weights — preserves
    every existing signal's behavior under a definition without `score`.
    """
    if score_expr and frame is not None:
        from . import dsl as _dsl
        return _dsl.evaluate(score_expr, frame, clip=True)

    weights = weights or REGIME_WEIGHTS
    # Vectorised regime-weighted sum. Equivalent to building a per-bar (n × 17)
    # weight DataFrame and dotting it with factors, but without the O(regimes ×
    # factors) Python loop + label-based .loc assignment + intermediate frame —
    # this is the hottest allocation in the optimiser (one call per trial × fold).
    order = list(FACTOR_ORDER)
    fac = factors[order].to_numpy(dtype=float)                 # (n, 17)
    # Weight table: one row per known regime + a trailing all-zero row that
    # unknown / missing regimes map to (matches the old fillna(0.0) for any bar
    # whose regime isn't in `weights`).
    regime_codes = {r: i for i, r in enumerate(weights.keys())}
    zero_row = len(regime_codes)
    w_table = np.zeros((zero_row + 1, len(order)), dtype=float)
    for r, i in regime_codes.items():
        fwts = weights[r]
        w_table[i] = [float(fwts.get(f, 0.0)) for f in order]
    codes = regimes.map(regime_codes).fillna(zero_row).to_numpy().astype(np.intp)
    # nansum (not plain sum) to match the old DataFrame .sum(axis=1) skipna=True:
    # a NaN factor contributes 0 rather than poisoning the whole bar's score.
    score = np.clip(np.nansum(fac * w_table[codes], axis=1), -1.0, 1.0)
    return pd.Series(score, index=factors.index)


def discretize(score: pd.Series, cfg: BacktestConfig,
               long_when_expr: str | None = None,
               short_when_expr: str | None = None,
               frame: pd.DataFrame | None = None) -> pd.Series:
    """Map a continuous score series to {-1, 0, +1} positions.

    Two paths:

    1. Threshold mode (default, backwards compat). `score >= long_thr →
       +1`; `score <= short_thr → -1`. Min-hold optional. This is what
       legacy weights-matrix signals use.

    2. Conditional mode (Phase 2C). When `long_when_expr` /
       `short_when_expr` are provided AND `frame` is available, the
       boolean DSL expressions are evaluated against the frame and any
       bar where the expression is true gets the corresponding position.
       Comes from the user's `long_when` / `short_when` fields on
       SignalDef. If only ONE side is provided, the other still uses
       the threshold rule — so `long_when=...; short_when="" ` means
       "use the entry condition for longs, the score threshold for
       shorts." That's the common shape — most signals are asymmetric.

    Conflict handling: if both expressions evaluate True on the same
    bar (illegal in a sane signal), short wins. This matches the
    threshold path's behaviour where a value that crosses both
    thresholds (impossible with thresholds of opposite sign, but
    possible with badly-formed conditions) lands on the second branch.
    """
    pos = pd.Series(0, index=score.index, dtype=int)
    long_set = False
    short_set = False
    if long_when_expr and frame is not None:
        from . import dsl as _dsl
        mask = _dsl.evaluate_bool(long_when_expr, frame).reindex(score.index).fillna(False).astype(bool)
        pos[mask.to_numpy()] = 1
        long_set = True
    if short_when_expr and frame is not None:
        from . import dsl as _dsl
        mask = _dsl.evaluate_bool(short_when_expr, frame).reindex(score.index).fillna(False).astype(bool)
        pos[mask.to_numpy()] = -1
        short_set = True
    if not long_set:
        pos[score >= cfg.long_threshold] = 1
    if not short_set:
        pos[score <= cfg.short_threshold] = -1
    if cfg.min_hold_bars > 0:
        pos = _apply_min_hold(pos, cfg.min_hold_bars)
    return pos


def _apply_min_hold(pos: pd.Series, min_hold: int) -> pd.Series:
    arr = pos.to_numpy().copy()
    held = 0
    cur = 0
    for i, v in enumerate(arr):
        if v != cur:
            if cur != 0 and held < min_hold:
                arr[i] = cur
                held += 1
            else:
                cur = v
                held = 1
        else:
            held += 1
    return pd.Series(arr, index=pos.index)
