"""Pretty-print backtest results."""

from __future__ import annotations

import pandas as pd

from .backtest import BacktestResult


def _fmt_pct(x: float) -> str:
    if x != x:  # NaN
        return "  n/a"
    return f"{x * 100:7.2f}%"


def _fmt(x: float, w: int = 7, p: int = 2) -> str:
    if x != x:
        return "n/a".rjust(w)
    return f"{x:{w}.{p}f}"


def print_window_summary(label: str, res: BacktestResult, regime_counts: pd.Series | None = None) -> None:
    m = res.metrics
    b = res.benchmark
    print(f"\n=== {label} ({m['bars']} bars) ===")
    print(f"  strategy total: {_fmt_pct(m['total_return'])}    CAGR: {_fmt_pct(m['cagr'])}    "
          f"Sharpe: {_fmt(m['sharpe'])}    MaxDD: {_fmt_pct(m['max_drawdown'])}")
    print(f"  buy-and-hold  : {_fmt_pct(b['total_return'])}    CAGR: {_fmt_pct(b['cagr'])}    "
          f"Sharpe: {_fmt(b['sharpe'])}    MaxDD: {_fmt_pct(b['max_drawdown'])}")
    print(f"  trades: {m['trades']:4d}   win-rate: {_fmt_pct(m['win_rate'])}   "
          f"avg trade: {_fmt_pct(m['avg_trade_return'])}   exposure: {_fmt_pct(m['exposure'])}")
    print(f"  exits: tp={m.get('exits_tp', 0)} sl={m.get('exits_sl', 0)} "
          f"flip={m.get('exits_flip', 0)} flat={m.get('exits_flat', 0)}")
    print(f"  turnover: {_fmt(m['turnover'], w=6, p=1)}   total cost: {_fmt_pct(m['total_cost'])}")
    if regime_counts is not None:
        rc = " ".join(f"{r}={n}" for r, n in regime_counts.items())
        print(f"  regimes: {rc}")


def print_overall_table(rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    cols = ["window", "bars", "total_return", "cagr", "sharpe", "max_drawdown",
            "trades", "win_rate", "exposure", "bh_total_return"]
    df = df[cols]
    df["total_return"] = df["total_return"].map(_fmt_pct)
    df["cagr"] = df["cagr"].map(_fmt_pct)
    df["max_drawdown"] = df["max_drawdown"].map(_fmt_pct)
    df["win_rate"] = df["win_rate"].map(_fmt_pct)
    df["exposure"] = df["exposure"].map(_fmt_pct)
    df["bh_total_return"] = df["bh_total_return"].map(_fmt_pct)
    df["sharpe"] = df["sharpe"].map(lambda x: _fmt(x))
    print("\n=== Summary across windows ===")
    print(df.to_string(index=False))
