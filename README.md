# perpsignal

A backtestable **signal engine** for perpetual-futures strategies. Write a
strategy as a compact expression (or a JSON signal definition), evaluate it
against OHLCV data, and backtest it into honest metrics — Sharpe, return,
drawdown, win rate — with **fees, funding, stops/targets, and leverage** modelled.

This is the open-source core extracted from [Signalview](https://www.signalview.xyz),
a non-custodial platform where backtested perps strategies are scored and traded
by AI agents on Hyperliquid. The engine here has **no wallet, key, custody, or
live-trading code** — it's a pure research/backtest library, safe to run anywhere.

> ⚠️ Research and backtesting only. Nothing here is financial advice, and a
> backtest is not a promise of future results. Perpetual-futures trading is
> high-risk.

## Install

```bash
pip install perpsignal          # once published
# or, from source:
pip install git+https://github.com/mokshyaprotocol/signalview
```

Requires Python 3.10+, `pandas`, `numpy`, `requests`.

## Quickstart

```python
import pandas as pd
from perpsignal import evaluate, discretize, run, BacktestConfig, RiskConfig

# df needs columns: open, high, low, close, volume (a DatetimeIndex is ideal)
df = pd.read_parquet("BTCUSDT-1h.parquet")

cfg = BacktestConfig(symbol="BTCUSDT", interval="1h")  # holds the entry thresholds

# A score is any expression that evaluates to a Series (positive = long).
# This one is a mean-reversion fade: high when price is stretched below its
# 48-bar mean, low when stretched above.
score = evaluate("zscore(close, 48) * -1", df)

# Map the continuous score to a {-1, 0, +1} position via the config thresholds.
# run() expects a discrete position, not a raw score.
position = discretize(score, cfg)

result = run(
    df, position, cfg,
    RiskConfig(),            # leverage / take-profit / stop-loss (sane defaults)
    bars_per_year=24 * 365,  # hourly bars
)
print(result.metrics)   # {'sharpe': ..., 'total_return': ..., 'max_drawdown': ..., 'win_rate': ..., 'trades': ...}
```

## The signal DSL

Expressions are built from market **variables** and **functions**.

**Variables:** `close`, `high`, `low`, `open`, `volume`, `funding`, `oi`
(open interest), `bar_index`.

**Functions:**
| | |
|---|---|
| Trend / momentum | `rsi(close, n)`, `ema(close, n)`, `sma(close, n)`, `slope(close, n)`, `adx(n)`, `macd`-style via `ema` diffs |
| Volatility | `atr(n)`, `stdev(x, n)`, `bb_width(close, n, k)`, `bb_upper/bb_lower/bb_mid(close, n, k)` |
| Normalization | `zscore(x, n)`, `clip(x, lo, hi)`, `sign(x)`, `abs(x)`, `log(x)`, `sqrt(x)`, `tanh(x)` |
| Volume / flow | `vwap(n)`, `session_vwap()`, `corr(a, b, n)` |
| Windowing | `highest(x, n)`, `lowest(x, n)`, `prev(x, k)` |
| Logic | `if(cond, a, b)`, `min`, `max`, comparisons (`>`, `<`, `>=`, ...), `and`/`or` |

Expressions are parsed by a real tokenizer→parser→evaluator (`perpsignal.dsl`)
and include an **auto-repair** pass (`parse_with_repair`) that fixes common
mistakes and reports what it changed — handy when the author is an LLM.

## Signals as data

A strategy can also be a JSON `SignalDef` (portable, diffable, PR-able):

```python
from perpsignal import parse_signal
sig = parse_signal({
    "asset": "BTCUSDT", "timeframe": "1h",
    "expression": "rsi(close, 14) - 50",
    # ... weights, regime config, risk bounds
})
```

See [`examples/`](examples/) for runnable scripts and a sample signal file.

## Fetching data

`perpsignal.data` pulls public Binance / Hyperliquid OHLCV and caches to disk.
An optional Upstash cache layer self-disables when its env vars are unset — you
never need it to run locally.

## Contributing

New indicators, factors, and strategies are very welcome — this engine exists to
be extended. A new built-in is usually **one function plus one test**. See
[CONTRIBUTING.md](CONTRIBUTING.md). Every PR is backtested in CI so improvements
are judged on objective metrics, not opinion — which also makes this a clean
target for autonomous/AI-agent strategy search.

## License

[Apache License 2.0](LICENSE). © 2026 Signalview.
