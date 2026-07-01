"""perpsignal — a backtestable signal engine for perpetual-futures strategies.

Open-source core extracted from Signalview (https://www.signalview.xyz). Write a
strategy as a signal expression or a SignalDef, evaluate it against OHLCV data,
and backtest it into honest metrics (Sharpe, return, drawdown, win rate) — with
fees, funding, stops/targets and leverage modelled.

Quickstart:

    import pandas as pd
    from perpsignal import evaluate, run, BacktestConfig

    df = pd.read_parquet("BTCUSDT-1h.parquet")       # open/high/low/close/volume
    signal = evaluate("zscore(close, 48) * -1", df)  # any DSL expression -> Series
    result = run(df, signal, BacktestConfig(symbol="BTCUSDT", interval="1h"))
    print(result.summary)

See the DSL reference and examples/ for the full built-in indicator set.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .backtest import run, BacktestResult
from .config import BacktestConfig, Costs, RiskConfig
from .dsl import evaluate, evaluate_bool, parse, parse_with_repair, EvalError, ParseError
from .signal_def import SignalDef, parse_signal, set_market_meta
from . import factors

__all__ = [
    "run",
    "BacktestResult",
    "BacktestConfig",
    "Costs",
    "RiskConfig",
    "evaluate",
    "evaluate_bool",
    "parse",
    "parse_with_repair",
    "EvalError",
    "ParseError",
    "SignalDef",
    "parse_signal",
    "set_market_meta",
    "factors",
    "__version__",
]
