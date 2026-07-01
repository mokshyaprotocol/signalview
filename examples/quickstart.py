"""Minimal end-to-end example: build a signal from a DSL expression and backtest
it. Uses synthetic OHLCV so it runs with no network or API keys.

    python examples/quickstart.py
"""
import numpy as np
import pandas as pd

from perpsignal import evaluate, discretize, run, BacktestConfig, RiskConfig


def synthetic_ohlcv(bars: int = 2000, seed: int = 7) -> pd.DataFrame:
    """A deterministic random-walk price series with OHLCV columns."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.01, bars)
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


def main() -> None:
    df = synthetic_ohlcv()

    cfg = BacktestConfig(symbol="BTCUSDT", interval="1h")  # holds the entry thresholds

    # Mean-reversion fade: score is high when price is stretched below its 48-bar
    # mean, low when stretched above. Any expression that returns a Series works.
    score = evaluate("zscore(close, 48) * -1", df)

    # Map the continuous score to a {-1, 0, +1} position using the config's
    # long/short thresholds. run() expects a discrete position, not a raw score.
    position = discretize(score, cfg)

    result = run(
        df,
        position,
        cfg,
        RiskConfig(),          # leverage, take-profit, stop-loss defaults
        bars_per_year=24 * 365,  # hourly bars
    )

    print("Backtest metrics (synthetic data — illustrative only):")
    for k, v in result.metrics.items():
        print(f"  {k:16s} {v}")
    print(f"\nTrades: {len(result.trades)}")


if __name__ == "__main__":
    main()
