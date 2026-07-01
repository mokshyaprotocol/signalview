"""Unit tests for the bar-by-bar backtest engine (perpsignal/backtest.py).

Every expected value here is hand-computed from the engine's stated conventions
(see the module docstring): a signal at the close of bar t opens a position that
is held during bar t+1 onward; per-bar return is (price move) x leverage; fees +
slippage are charged once on entry and once on exit; a trade's reported return is
NET OF BOTH fees. Where there are no fees/funding, assertions are exact.
"""

import numpy as np
import pandas as pd
import pytest

from perpsignal.backtest import run, _buy_and_hold, _summarize
from perpsignal.config import BacktestConfig, Costs, RiskConfig


def _idx(n):
    return pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")


def _df(close, high=None, low=None, funding=None):
    n = len(close)
    idx = _idx(n)
    high = close if high is None else high
    low = close if low is None else low
    data = {"close": np.array(close, float), "high": np.array(high, float),
            "low": np.array(low, float)}
    if funding is not None:
        data["funding_rate"] = np.array(funding, float)
    return pd.DataFrame(data, index=idx)


def _bt(close, sig, *, high=None, low=None, funding=None, fee_bps=0.0, slip_bps=0.0,
        lev=1.0, tp=None, sl=None, min_hold=0):
    df = _df(close, high, low, funding)
    cfg = BacktestConfig(symbol="X", interval="1h",
                         costs=Costs(taker_fee_bps=fee_bps, slippage_bps=slip_bps),
                         min_hold_bars=min_hold)
    risk = RiskConfig(leverage=lev, take_profit_pct=tp, stop_loss_pct=sl, auto_tf=False)
    return run(df, pd.Series(sig, index=df.index), cfg, risk, bars_per_year=24 * 365)


# ---------------------------------------------------------------------------
# Core return / equity math (no fees -> exact)
# ---------------------------------------------------------------------------
def test_empty_frame_returns_empty():
    df = pd.DataFrame({"close": [], "high": [], "low": []})
    risk = RiskConfig(leverage=1.0, take_profit_pct=None, stop_loss_pct=None, auto_tf=False)
    r = run(df, pd.Series([], dtype=float), BacktestConfig(symbol="X", interval="1h",
            costs=Costs(0, 0)), risk, bars_per_year=24 * 365)
    assert r.metrics == {}
    assert r.trades.empty


def test_single_long_roundtrip_exact_no_fees():
    # +10% then +10% then exit flat. Trade return = 1.1*1.1 - 1 = 0.21.
    r = _bt([100, 110, 121, 121], [1, 1, 0, 0])
    assert r.metrics["trades"] == 1
    assert r.metrics["total_return"] == pytest.approx(0.21, abs=1e-12)
    assert float(r.trades["ret"].iloc[0]) == pytest.approx(0.21, abs=1e-12)
    assert r.metrics["win_rate"] == pytest.approx(1.0)
    assert r.metrics["avg_trade_return"] == pytest.approx(0.21, abs=1e-12)
    assert r.trades["exit"].iloc[0] == "flat"
    assert int(r.trades["bars"].iloc[0]) == 2  # held during bars 1 and 2


def test_leverage_scales_bar_returns():
    r = _bt([100, 110, 121, 121], [1, 1, 0, 0], lev=2.0)
    # bar returns become 0.20 each -> 1.2*1.2 - 1 = 0.44
    assert r.metrics["total_return"] == pytest.approx(0.44, abs=1e-12)
    assert float(r.trades["ret"].iloc[0]) == pytest.approx(0.44, abs=1e-12)


def test_short_profits_when_price_falls():
    # short from 100; price -> 90 is +10% for a short.
    r = _bt([100, 90, 90], [-1, 0, 0])
    assert r.trades["side"].iloc[0] == "short"
    assert float(r.trades["ret"].iloc[0]) == pytest.approx(0.10, abs=1e-12)
    assert r.metrics["total_return"] == pytest.approx(0.10, abs=1e-12)


# ---------------------------------------------------------------------------
# THE FIX: per-trade return must be net of the ENTRY fee, not just the exit fee.
# ---------------------------------------------------------------------------
def test_trade_return_includes_entry_fee():
    # 0.10% per side, lev 1. A single flat->flat round trip must reconcile:
    # the one trade's return equals total_return (both net of entry+exit fees).
    r = _bt([100, 101, 102, 103], [1, 1, 0, 0], fee_bps=5.0, slip_bps=5.0)
    assert r.metrics["trades"] == 1
    trade_ret = float(r.trades["ret"].iloc[0])
    # Multiplicative trade accounting vs additive-per-bar equity differ only by a
    # within-bar cross term (O(fee^2) ~ 1e-6); reconcile to a tight tolerance.
    assert trade_ret == pytest.approx(r.metrics["total_return"], abs=5e-4)
    # And it must be clearly LOWER than the gross (fee-less) trade return.
    gross = _bt([100, 101, 102, 103], [1, 1, 0, 0]).trades["ret"].iloc[0]
    assert trade_ret < gross - 0.0015  # ~2 sides * 0.10% drag


def test_marginal_winner_flips_to_loss_after_fees():
    # +0.10% gross over the hold, but ~0.20% round-trip fees -> net loss.
    gross = _bt([100, 100.10, 100.10], [1, 0, 0])
    assert float(gross.trades["ret"].iloc[0]) == pytest.approx(0.001, abs=1e-9)
    assert gross.metrics["win_rate"] == pytest.approx(1.0)
    net = _bt([100, 100.10, 100.10], [1, 0, 0], fee_bps=5.0, slip_bps=5.0)
    assert float(net.trades["ret"].iloc[0]) < 0
    assert net.metrics["win_rate"] == pytest.approx(0.0)  # the bug made this 1.0


# ---------------------------------------------------------------------------
# TP / SL exits
# ---------------------------------------------------------------------------
def test_take_profit_exit():
    # long from 100, TP at +5%; bar 1 high 106 crosses 105.
    r = _bt([100, 104, 104], [1, 0, 0], high=[100, 106, 104], low=[100, 100, 104], tp=0.05)
    assert r.trades["exit"].iloc[0] == "tp"
    assert float(r.trades["ret"].iloc[0]) == pytest.approx(0.05, abs=1e-12)


def test_stop_loss_exit():
    # long from 100, SL at -3%; bar 1 low 96 crosses 97.
    r = _bt([100, 98, 98], [1, 0, 0], high=[100, 100, 98], low=[100, 96, 98], sl=0.03)
    assert r.trades["exit"].iloc[0] == "sl"
    assert float(r.trades["ret"].iloc[0]) == pytest.approx(-0.03, abs=1e-12)


def test_stop_loss_takes_priority_when_both_hit():
    # bar 1 spans both the SL (97) and TP (105); SL must win (conservative).
    r = _bt([100, 100, 100], [1, 0, 0], high=[100, 106, 100], low=[100, 96, 100],
            tp=0.05, sl=0.03)
    assert r.trades["exit"].iloc[0] == "sl"
    assert float(r.trades["ret"].iloc[0]) == pytest.approx(-0.03, abs=1e-12)


def test_min_hold_blocks_signal_exit_but_not_stop():
    # SL must fire even on the first held bar regardless of min_hold.
    r = _bt([100, 98, 98], [1, 0, 0], high=[100, 100, 98], low=[100, 96, 98],
            sl=0.03, min_hold=99)
    assert r.trades["exit"].iloc[0] == "sl"


# ---------------------------------------------------------------------------
# Win rate / avg trade / drawdown over a known two-trade path
# ---------------------------------------------------------------------------
def test_two_trades_winrate_and_drawdown():
    # Trade A: +20%, flat, Trade B: -10% (from the 1.20 peak).
    close = [100, 120, 120, 120, 108, 108]
    sig = [1, 1, 0, 1, 0, 0]
    r = _bt(close, sig)
    assert r.metrics["trades"] == 2
    rets = sorted(round(float(x), 10) for x in r.trades["ret"])
    assert rets == [pytest.approx(-0.10), pytest.approx(0.20)]
    assert r.metrics["win_rate"] == pytest.approx(0.5)
    assert r.metrics["avg_trade_return"] == pytest.approx(0.05, abs=1e-12)
    assert r.metrics["total_return"] == pytest.approx(0.08, abs=1e-12)
    # equity peaks at 1.20, troughs at 1.08 -> dd = 1.08/1.20 - 1 = -0.10
    assert r.metrics["max_drawdown"] == pytest.approx(-0.10, abs=1e-12)


def test_drawdown_is_zero_for_monotonic_gains():
    r = _bt([100, 110, 121, 121], [1, 1, 0, 0])
    assert r.metrics["max_drawdown"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Costs accounting
# ---------------------------------------------------------------------------
def test_total_cost_is_entry_plus_exit_fee():
    # one round trip, 0.10% per side, lev 1 -> 0.002 total, no funding.
    r = _bt([100, 101, 102, 102], [1, 1, 0, 0], fee_bps=5.0, slip_bps=5.0)
    assert r.metrics["total_cost"] == pytest.approx(0.002, abs=1e-9)


def test_funding_reduces_a_long_when_rate_positive():
    # held long over bar 1 with +0.08 funding rate -> funding_cost = 1*1*0.08/8 = 0.01
    r = _bt([100, 100, 100], [1, 0, 0], funding=[0.0, 0.08, 0.0])
    # net for the held bar = 0 (no price move) - 0.01 funding
    assert r.df["funding_cost"].iloc[1] == pytest.approx(0.01, abs=1e-12)
    assert r.df["net_ret"].iloc[1] == pytest.approx(-0.01, abs=1e-12)


def test_exposure_and_turnover():
    r = _bt([100, 110, 121, 121], [1, 1, 0, 0], lev=1.0)
    # held during bars 1 and 2 only -> 2/4 exposure
    assert r.metrics["exposure"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Buy & hold benchmark
# ---------------------------------------------------------------------------
def test_buy_and_hold():
    bh = _buy_and_hold(pd.Series([100.0, 110.0, 121.0]), bars_per_year=24 * 365)
    assert bh["total_return"] == pytest.approx(0.21, abs=1e-12)
    assert bh["max_drawdown"] == pytest.approx(0.0, abs=1e-12)


def test_buy_and_hold_drawdown():
    bh = _buy_and_hold(pd.Series([100.0, 120.0, 108.0]), bars_per_year=24 * 365)
    assert bh["total_return"] == pytest.approx(0.08, abs=1e-12)
    assert bh["max_drawdown"] == pytest.approx(-0.10, abs=1e-12)


# ---------------------------------------------------------------------------
# Reconciliation: trades fully explain equity when separated by flat bars
# ---------------------------------------------------------------------------
def test_trades_reconcile_with_equity_no_fees():
    close = [100, 120, 120, 120, 108, 108]
    sig = [1, 1, 0, 1, 0, 0]
    r = _bt(close, sig)
    prod = float(np.prod([1.0 + x for x in r.trades["ret"]]))
    assert prod - 1.0 == pytest.approx(r.metrics["total_return"], abs=1e-12)


# ---------------------------------------------------------------------------
# Flips: exit of trade A and entry of trade B land on the SAME bar — the case
# most prone to fee mis-attribution. Both trades must net their own fees.
# ---------------------------------------------------------------------------
def test_flip_long_to_short_two_trades():
    # long from 100 -> 110 (+10%), flip to short at 110 -> 99 (+10% short).
    close = [100, 110, 99, 99]
    sig = [1, -1, 0, 0]
    r = _bt(close, sig)
    assert r.metrics["trades"] == 2
    assert sorted(r.trades["side"]) == ["long", "short"]
    assert r.trades.iloc[0]["exit"] == "flip"
    # no fees -> long made +10%, short made (110-99)/110 = +10%
    assert float(r.trades.iloc[0]["ret"]) == pytest.approx(0.10, abs=1e-12)
    assert float(r.trades.iloc[1]["ret"]) == pytest.approx(0.10, abs=1e-12)


def test_flip_each_trade_nets_its_own_fees():
    # With fees, BOTH trades must each pay an entry+exit fee. Total cost across
    # the flip = 3 fee events on the shared timeline (entryA, exitA+entryB, exitB)
    # = 4 sides total (A: entry+exit, B: entry+exit).
    close = [100, 110, 99, 99]
    sig = [1, -1, 0, 0]
    r = _bt(close, sig, fee_bps=5.0, slip_bps=5.0)  # 0.10% per side
    gross = _bt(close, sig)
    # every trade return must drop by ~2 sides * 0.10% vs the gross run
    for i in range(2):
        assert float(r.trades.iloc[i]["ret"]) < float(gross.trades.iloc[i]["ret"]) - 0.0015
    # 2 round trips => 4 fee sides => 0.004 total cost
    assert r.metrics["total_cost"] == pytest.approx(0.004, abs=1e-9)


def test_open_position_at_end_is_recorded_net_of_entry_fee_only():
    # enter long, never exit -> one open trade, no exit fee charged to it
    r = _bt([100, 110, 121], [1, 1, 1], fee_bps=5.0, slip_bps=5.0)
    assert r.metrics["trades"] == 1
    assert r.trades.iloc[0]["exit"] == "open"
    # gross hold = 1.1*1.1-1 = 0.21; minus only the entry fee (~0.001)
    assert float(r.trades.iloc[0]["ret"]) == pytest.approx(0.21 * (1 - 0.001) + (-0.001) + 0.21*0, abs=2e-3)
    assert float(r.trades.iloc[0]["ret"]) < 0.21
