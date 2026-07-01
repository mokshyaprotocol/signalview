# Contributing to perpsignal

This engine exists to be extended. New indicators, factors, and strategies are
the whole point — from human quants and autonomous agents alike. Contributions
are judged on **objective backtest metrics**, so a good PR is small, tested, and
measurable.

## Setup

```bash
git clone https://github.com/mokshyaprotocol/signalview
cd signalview
pip install -e ".[dev]"
pytest -q
```

## Adding a DSL indicator (the most common PR)

Built-in functions live in `perpsignal/dsl.py`. Adding one is usually three small
steps:

1. Write the function `def _bi_myindicator(ctx, args): ...` returning a
   `pd.Series` aligned to `ctx.index`. Use `ctx.frame["close"]` etc. for market
   data and the helpers in `perpsignal/factors.py` where you can.
2. Register it in the `BUILTINS` dict (and add natural-language aliases in
   `_BUILTIN_ALIASES` if useful — the auto-repair pass uses them).
3. Add a test in `tests/` that evaluates an expression using it and asserts the
   output shape/values.

Keep indicators **causal** (no look-ahead: never use future bars) and
NaN-safe at the series head.

## Adding / improving a strategy

Strategies are expressions or JSON `SignalDef`s. If you're proposing a strategy,
include its backtest metrics on a public dataset and the exact config
(`BacktestConfig` + `RiskConfig`) you used, so results are reproducible.

## Rules of the house

- **No look-ahead bias.** This is the fastest way to a rejected PR.
- **No performance promises** in docs or comments. A backtest is not a forecast.
- **Deterministic tests.** Seed any randomness; don't depend on live network in
  unit tests (the CI backtest uses a committed dataset).
- Match the surrounding style; keep functions pure where the existing ones are.

## CI

Every PR runs the test suite and a backtest smoke-run. Green + a clear metric
delta is what gets a strategy or indicator merged.

By contributing you agree your contributions are licensed under Apache-2.0.
