"""Guard the shipped example signals: each must parse, evaluate, discretize, and
backtest to finite metrics with at least one trade on the synthetic dataset.
Keeps the examples (and the evaluate -> discretize -> run pipeline) honest in CI.
"""
import glob
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from perpsignal import evaluate, discretize, run, parse_signal, BacktestConfig, RiskConfig

SIGNALS = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "examples", "signals", "*.json")))


def _data(bars: int = 2000, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = np.zeros(bars)
    t = 0
    while t < bars:
        seg = int(rng.integers(80, 300))
        drift[t:t + seg] = rng.choice([0.0005, -0.0005, 0.0, 0.0])
        t += seg
    ret = drift[:bars] + rng.normal(0, 0.011, bars)
    close = 30_000 * np.exp(np.cumsum(ret))
    high = close * (1 + rng.uniform(0, 0.004, bars))
    low = close * (1 - rng.uniform(0, 0.004, bars))
    idx = pd.date_range("2025-01-01", periods=bars, freq="1h", tz="UTC")
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close,
                         "volume": rng.uniform(100, 1000, bars)}, index=idx)


@pytest.mark.parametrize("path", SIGNALS, ids=[os.path.basename(p) for p in SIGNALS])
def test_example_signal_backtests(path):
    sig = parse_signal(json.load(open(path)))
    assert sig.score, f"{path} has no score expression"
    df = _data()
    cfg = BacktestConfig(symbol=sig.asset, interval="1h",
                         long_threshold=sig.long_threshold,
                         short_threshold=sig.short_threshold)
    position = discretize(evaluate(sig.score, df), cfg)
    assert set(np.unique(position)).issubset({-1, 0, 1})
    m = run(df, position, cfg, RiskConfig(leverage=1.0), bars_per_year=24 * 365).metrics
    assert m["trades"] > 0, f"{os.path.basename(path)} never trades"
    assert math.isfinite(m["sharpe"]), f"{os.path.basename(path)} sharpe not finite"


def test_all_archetypes_present():
    assert len(SIGNALS) >= 5
