# Example signals

Each `.json` here is a `SignalDef` — a trading strategy expressed as **data**.
They're intentionally simple, one idea each, and cover the main archetypes:

| File | Idea |
|---|---|
| `rsi-mean-reversion.json` | fade RSI away from 50 |
| `bollinger-reversion.json` | fade extension from the Bollinger midline |
| `trend-follow-ema.json` | 20/50 EMA cross, scaled by ADX trend strength |
| `donchian-breakout.json` | break of the prior 20-bar high/low |
| `vol-scaled-momentum.json` | 24-bar move measured in ATR units |

## Run them

```bash
pip install -e ".[dev]"
python scripts/backtest_signals.py            # backtests every signal, prints a leaderboard
python scripts/backtest_signals.py data.parquet   # ...against your own OHLCV
```

The leaderboard runs on deterministic **synthetic** data by default, so results
are illustrative — a smoke test that each signal is valid and trades, not a claim
about live performance.

## The pipeline

A `score` expression is continuous; the backtester wants a discrete position:

```python
from perpsignal import evaluate, discretize, run, parse_signal, BacktestConfig, RiskConfig

sig = parse_signal(json.load(open("examples/signals/rsi-mean-reversion.json")))
cfg = BacktestConfig(symbol=sig.asset, interval="1h",
                     long_threshold=sig.long_threshold,
                     short_threshold=sig.short_threshold)
score    = evaluate(sig.score, df)      # continuous
position = discretize(score, cfg)       # -> {-1, 0, +1} via the thresholds
result   = run(df, position, cfg, RiskConfig(), bars_per_year=24 * 365)
```

## Add your own

1. Copy any file above and change `name`, `description`, and `score`.
2. `score` is a DSL expression over `open/high/low/close/volume/funding/oi` and
   the built-in functions (see the top-level README). Keep it **causal** — no
   look-ahead. Use `prev(x, k)` when you need the previous bar's value.
3. Set `long_threshold` / `short_threshold` so the score actually crosses them.
4. Run `scripts/backtest_signals.py` and open a PR — CI backtests it for you.

Note: `funding` and `oi` are only present with real data; the synthetic
smoke-test data is OHLCV only.
