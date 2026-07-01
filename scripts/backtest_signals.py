"""Backtest every signal in examples/signals/ and print a ranked leaderboard.

Each signal file is a JSON SignalDef with a `score` DSL expression. The score is
evaluated into a Series, backtested, and ranked by Sharpe. Runs on deterministic
synthetic data by default so it needs no network — pass a Parquet/CSV of real
OHLCV as the first arg to score against real data.

    python scripts/backtest_signals.py                 # synthetic
    python scripts/backtest_signals.py BTCUSDT-1h.parquet

This doubles as the reward signal for automated strategy search: an agent writes
a `score` expression, drops it in examples/signals/, and reads its rank here.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from perpsignal import evaluate, discretize, run, parse_signal, BacktestConfig, RiskConfig

SIGNALS_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "signals")


def synthetic_ohlcv(bars: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Random walk with regime structure — alternating trending and choppy
    segments — so trend/breakout AND mean-reversion strategies all have something
    to catch. Deterministic (seeded). Not real data; illustrative only."""
    rng = np.random.default_rng(seed)
    drift = np.zeros(bars)
    t = 0
    while t < bars:
        seg = int(rng.integers(80, 300))
        # More chop than trend so neither trend- nor reversion-style signals
        # dominate — a fairer illustration.
        drift[t:t + seg] = rng.choice([0.0005, -0.0005, 0.0, 0.0])
        t += seg
    ret = drift[:bars] + rng.normal(0, 0.011, bars)
    close = 30_000 * np.exp(np.cumsum(ret))
    high = close * (1 + rng.uniform(0, 0.004, bars))
    low = close * (1 - rng.uniform(0, 0.004, bars))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.uniform(100, 1000, bars)
    idx = pd.date_range("2025-01-01", periods=bars, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def load_data(path: str | None) -> pd.DataFrame:
    if not path:
        return synthetic_ohlcv()
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def main() -> None:
    df = load_data(sys.argv[1] if len(sys.argv) > 1 else None)
    rows = []
    for f in sorted(glob.glob(os.path.join(SIGNALS_DIR, "*.json"))):
        name = os.path.basename(f)
        try:
            sig = parse_signal(json.load(open(f)))
            if not sig.score:
                print(f"  skip {name}: no `score` expression"); continue
            cfg = BacktestConfig(symbol=sig.asset, interval="1h",
                                 long_threshold=sig.long_threshold,
                                 short_threshold=sig.short_threshold)
            score = evaluate(sig.score, df)          # continuous score
            position = discretize(score, cfg)        # -> {-1, 0, +1} using thresholds
            risk = RiskConfig(leverage=1.0)          # 1x for a gentle illustration
            m = run(df, position, cfg, risk, bars_per_year=24 * 365).metrics
            rows.append((sig.name, m["sharpe"], m["total_return"], m["max_drawdown"],
                         m["win_rate"], int(m["trades"])))
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {name}: {e}")

    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"\nLeaderboard ({'synthetic' if len(sys.argv) < 2 else sys.argv[1]} data — illustrative):\n")
    print(f"{'strategy':32s} {'sharpe':>7} {'return':>8} {'maxDD':>7} {'win':>6} {'trades':>6}")
    print("-" * 72)
    for name, sharpe, ret, dd, win, trades in rows:
        print(f"{name[:32]:32s} {sharpe:7.2f} {ret:8.2%} {dd:7.2%} {win:6.1%} {trades:6d}")


if __name__ == "__main__":
    main()
