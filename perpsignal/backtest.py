"""Bar-by-bar backtest engine with TP/SL, leverage, fees, slippage, funding.

Conventions:
  - Signal at bar t uses data through bar t's close. Entry/exit decided at the
    close of bar t fires at that close price; the resulting position is active
    during bar t+1 onward.
  - Intra-bar TP/SL: while a position is open during bar k, we check whether
    the bar's [low, high] range crosses the entry-relative TP or SL price.
    If both could fire in the same bar, we assume SL fires first (conservative).
  - Leverage scales bar PnL linearly; we cap cumulative equity at 0 to model
    full margin wipeout (no negative equity).
  - Costs: (taker_fee + slippage) bps per unit of leveraged turnover, charged
    on every entry and every exit.
  - Funding: charged each bar in proportion to position_size * funding/8.
  - min_hold_bars only blocks **signal-driven** exits/flips; TP/SL hits always
    fire (you want your stop to work even on bar 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import BacktestConfig, RiskConfig


@dataclass
class BacktestResult:
    df: pd.DataFrame
    metrics: dict
    trades: pd.DataFrame
    benchmark: dict


@dataclass
class PolicyConfig:
    """Optional per-bar policy filters layered on top of the signal.

    - skip_hours_utc: hours of the UTC day (0..23) where new entries are
      blocked. Open positions keep their TP/SL (a 03:55-entered trade
      doesn't get force-closed at 04:00; only NEW entries are gated).
    - news_timestamps_ms / news_skip_minutes: each timestamp is the centre
      of a ±N-minute blackout window during which new entries are blocked.
    - max_trades_per_24h: rolling 24h cap on entries. 0 disables the cap.
    - max_consecutive_losses: stop trading for the rest of the UTC day
      after N consecutive losing trades. Counter resets at 00:00 UTC.
      0 disables the circuit breaker.

    All defaults are no-op so a caller that doesn't construct a
    PolicyConfig (or passes the bare default) backtests identically to
    pre-Phase-3 behaviour.
    """
    skip_hours_utc: tuple[int, ...] = field(default_factory=tuple)
    news_timestamps_ms: tuple[int, ...] = field(default_factory=tuple)
    news_skip_minutes: int = 15
    max_trades_per_24h: int = 0
    max_consecutive_losses: int = 0

    def is_active(self) -> bool:
        """Cheap fast-path check — callers can skip the per-bar mask
        construction entirely when no policy is set."""
        return bool(
            self.skip_hours_utc
            or self.news_timestamps_ms
            or self.max_trades_per_24h
            or self.max_consecutive_losses
        )


def run(df: pd.DataFrame, signal: pd.Series, cfg: BacktestConfig,
        risk: RiskConfig, bars_per_year: int,
        policy: PolicyConfig | None = None,
        metrics_only: bool = False) -> BacktestResult:
    """df must contain columns: close, high, low, funding_rate (optional).

    `policy` (optional) layers per-bar entry filters and circuit breakers
    on top of the signal-driven flow. Defaults to a no-op PolicyConfig so
    existing callers (every test + legacy code path) are unaffected.

    `metrics_only` (optional): skip assembling the per-bar output DataFrame and
    the buy-and-hold benchmark, returning only `metrics` (+ the small trades
    frame). The optimiser/threshold-sweep run hundreds of backtests just to read
    Sharpe + trade count, so this avoids building output they immediately discard.
    Metrics are computed by the same code path, so they're identical either way.
    """
    required = {"close", "high", "low"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in df: {missing}")

    policy = policy or PolicyConfig()

    close = df["close"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    funding = df.get("funding_rate", pd.Series(0.0, index=df.index)).fillna(0.0).astype(float).to_numpy()
    sig = signal.reindex(df.index).fillna(0).astype(int).to_numpy()

    n = len(df)
    if n == 0:
        return BacktestResult(pd.DataFrame(), {}, pd.DataFrame(), {})

    fee_slip = (cfg.costs.taker_fee_bps + cfg.costs.slippage_bps) / 1e4
    lev = float(risk.leverage)
    tp = risk.take_profit_pct
    sl = risk.stop_loss_pct
    min_hold = cfg.min_hold_bars

    # Precompute static entry-block mask + per-bar UTC day key. The mask
    # covers skip-hours and news-blackout windows — both decided once
    # from the bar's timestamp, no inner-loop state. The day key (UTC
    # date as int) drives the consecutive-loss circuit breaker's daily
    # reset and the 24h-trade-cap window.
    entry_blocked, bar_ms, day_key = _build_policy_mask(df.index, policy)
    max_trades = max(0, int(policy.max_trades_per_24h))
    max_losses = max(0, int(policy.max_consecutive_losses))
    # Rolling 24h trade-entry timestamps. Trim from the head each tick
    # so the check stays O(1) amortized regardless of bar count.
    recent_entries_ms: list[float] = []
    ROLL_WINDOW_MS = 24 * 60 * 60 * 1000
    # Circuit-breaker counters. consec_losses tracks runs; day_blocked
    # latches True for the rest of the day once the cap is hit and is
    # cleared on the first bar of the next UTC day.
    consec_losses = 0
    day_blocked = False
    current_day = day_key[0] if len(day_key) else 0
    # Track per-position PnL so we can update the loss counter on exit
    # without re-walking the trades DataFrame.
    position_pnl = 0.0

    # State
    side = 0           # -1, 0, +1
    entry_price = 0.0
    held_bars = 0

    # Per-bar arrays
    pos_track = np.zeros(n)
    bar_ret = np.zeros(n)
    trade_cost = np.zeros(n)
    funding_cost = np.zeros(n)
    exit_kind = np.full(n, "", dtype=object)  # 'tp', 'sl', 'flip', 'flat'

    # Per-trade ledger, recorded in-loop so each trade's return is NET OF BOTH its
    # own entry and exit fees. (Reconstructing trades post-hoc by grouping equal
    # `position` values drops the entry fee — it's charged on the decision bar,
    # which still has position==0, so it lands in the preceding flat run and is
    # never attributed to the trade. That made win_rate / avg_trade_return
    # systematically optimistic.) `cost` is one side's leveraged fee+slippage; a
    # round trip pays it on entry and again on exit.
    cost = fee_slip * lev
    trades_rows: list[dict] = []
    cur: dict | None = None  # open trade: {side, factor, first_t, last_t, bars}

    def _row(c: dict, exk: str) -> dict:
        return {
            "start": df.index[c["first_t"]],
            "end": df.index[c["last_t"]],
            "bars": c["bars"],
            "side": "long" if c["side"] > 0 else "short",
            "ret": c["factor"] - 1.0,
            "exit": exk,
        }

    def _hold(c: dict, t_: int) -> None:
        """Fold bar t_'s held P&L (price move net of funding, leverage already in
        bar_ret) into the open trade's compounding return factor."""
        if c["first_t"] is None:
            c["first_t"] = t_
        c["last_t"] = t_
        c["bars"] += 1
        c["factor"] *= (1.0 + bar_ret[t_] - funding_cost[t_])

    for t in range(n):
        ht, lt, ct = high[t], low[t], close[t]
        prev_close = close[t - 1] if t > 0 else ct
        held_side = side  # position held DURING bar t (decided at end of bar t-1)

        # Daily reset for the circuit breaker (00:00 UTC). day_key is the
        # int UTC date — when it changes, both `day_blocked` and the
        # consecutive-loss counter clear so a new trading day starts fresh.
        if max_losses and day_key[t] != current_day:
            current_day = day_key[t]
            day_blocked = False
            consec_losses = 0
        elif not max_losses:
            current_day = day_key[t]

        sl_or_tp_hit = False
        bar_pnl_for_position = 0.0  # accumulator: tracks the held position's PnL this bar

        # ---- Intra-bar TP/SL check (if holding) ----
        if held_side != 0:
            if held_side == 1:
                sl_price = entry_price * (1 - sl) if sl is not None else -np.inf
                tp_price = entry_price * (1 + tp) if tp is not None else np.inf
                hit_sl = lt <= sl_price
                hit_tp = ht >= tp_price
            else:
                sl_price = entry_price * (1 + sl) if sl is not None else np.inf
                tp_price = entry_price * (1 - tp) if tp is not None else -np.inf
                hit_sl = ht >= sl_price
                hit_tp = lt <= tp_price

            if hit_sl or hit_tp:
                # SL takes priority if both could fire in the same bar (conservative)
                exit_price = sl_price if hit_sl else tp_price
                pnl = (exit_price - prev_close) / prev_close * held_side
                bar_ret[t] = pnl * lev
                trade_cost[t] += fee_slip * lev          # exit cost
                funding_cost[t] = held_side * lev * funding[t] / 8.0 * 0.5  # half-bar funding
                exit_kind[t] = "sl" if hit_sl else "tp"
                bar_pnl_for_position = pnl * lev
                position_pnl += bar_pnl_for_position
                # Close the open trade: count this (partial) bar's hold, then the
                # exit fee. Recorded net of entry+exit fees.
                if cur is not None:
                    _hold(cur, t)
                    cur["factor"] *= (1.0 - cost)
                    if cur["bars"] > 0:
                        trades_rows.append(_row(cur, exit_kind[t]))
                    cur = None
                # Circuit breaker — update loss streak based on the
                # full position PnL (entry → exit), then maybe latch
                # `day_blocked`.
                if max_losses:
                    if position_pnl < 0:
                        consec_losses += 1
                        if consec_losses >= max_losses:
                            day_blocked = True
                    else:
                        consec_losses = 0
                side = 0
                entry_price = 0.0
                held_bars = 0
                position_pnl = 0.0
                sl_or_tp_hit = True
            else:
                # Normal hold for the full bar
                pnl = (ct - prev_close) / prev_close * held_side
                bar_ret[t] = pnl * lev
                funding_cost[t] = held_side * lev * funding[t] / 8.0
                held_bars += 1
                position_pnl += pnl * lev
                if cur is not None:
                    _hold(cur, t)

        # Position held DURING this bar (independent of any end-of-bar updates)
        pos_track[t] = held_side * lev

        # ---- End-of-bar signal action (decides the side held during bar t+1) ----
        if not sl_or_tp_hit:
            next_desired = int(sig[t])
            # Honor signal-driven exits / flips first (independent of entry
            # filters — closing out an existing position is always allowed,
            # even during a news blackout or after the trade cap is hit).
            if side != 0 and next_desired != side:
                if held_bars >= min_hold:
                    trade_cost[t] += fee_slip * lev      # exit cost
                    exit_kind[t] = "flip" if next_desired != 0 else "flat"
                    if max_losses:
                        if position_pnl < 0:
                            consec_losses += 1
                            if consec_losses >= max_losses:
                                day_blocked = True
                        else:
                            consec_losses = 0
                    # Close the open trade net of its exit fee. Bar t's hold was
                    # already folded in by the normal-hold branch above this iter.
                    if cur is not None:
                        cur["factor"] *= (1.0 - cost)
                        if cur["bars"] > 0:
                            trades_rows.append(_row(cur, exit_kind[t]))
                        cur = None
                    side = 0
                    entry_price = 0.0
                    held_bars = 0
                    position_pnl = 0.0
            if side == 0 and next_desired != 0:
                # Apply entry-side filters in order: static blackout mask,
                # daily-loss circuit breaker, then 24h trade-cap budget.
                blocked = False
                if entry_blocked is not None and entry_blocked[t]:
                    blocked = True
                if max_losses and day_blocked:
                    blocked = True
                if not blocked and max_trades:
                    cutoff = bar_ms[t] - ROLL_WINDOW_MS
                    while recent_entries_ms and recent_entries_ms[0] < cutoff:
                        recent_entries_ms.pop(0)
                    if len(recent_entries_ms) >= max_trades:
                        blocked = True
                if not blocked:
                    trade_cost[t] += fee_slip * lev          # entry cost
                    side = next_desired
                    entry_price = ct
                    held_bars = 0
                    position_pnl = 0.0
                    # Open a new trade; the entry fee is pre-loaded into its factor
                    # so the trade's return is net of it. First held bar is t+1.
                    cur = {"side": next_desired, "factor": (1.0 - cost),
                           "first_t": None, "last_t": None, "bars": 0}
                    if max_trades:
                        recent_entries_ms.append(bar_ms[t])
        # If TP/SL fired this bar, do not re-enter the same bar; the next bar will
        # check the signal and may re-open if it's still firing.

    # Net per-bar return, equity capped at 0 (full liquidation)
    net = bar_ret - trade_cost - funding_cost
    # Clamp downside per bar to -1 to avoid impossible (1+x)<=0; then compound
    net_clamped = np.clip(net, -1.0, None)
    equity = np.cumprod(1.0 + net_clamped)
    # If equity ever hits 0, it stays at 0
    equity = np.where(equity > 0, equity, 0.0)

    # A position still open at the last bar — record it net of its entry fee only
    # (no exit fee, it hasn't closed), so the trade table still reflects it.
    if cur is not None and cur["bars"] > 0:
        trades_rows.append(_row(cur, "open"))
    trades = pd.DataFrame(trades_rows, columns=["start", "end", "bars", "side", "ret", "exit"])

    if metrics_only:
        # Skip the per-bar output frame + benchmark — the optimiser/sweep only
        # reads `metrics`. Same summarizer, so the numbers are identical.
        metrics = _summarize_arrays(equity, net_clamped, pos_track,
                                    trade_cost, funding_cost, trades, bars_per_year)
        return BacktestResult(df=pd.DataFrame(), metrics=metrics, trades=trades, benchmark={})

    out = pd.DataFrame({
        "close": close,
        "position": pos_track,
        "bar_ret": bar_ret,
        "trade_cost": trade_cost,
        "funding_cost": funding_cost,
        "net_ret": net_clamped,
        "equity": equity,
        "exit": exit_kind,
    }, index=df.index)

    metrics = _summarize(out, trades, bars_per_year)
    benchmark = _buy_and_hold(df["close"], bars_per_year)
    return BacktestResult(df=out, metrics=metrics, trades=trades, benchmark=benchmark)


# ---------------------------------------------------------------------------
def _build_policy_mask(index: pd.Index, policy: PolicyConfig) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Precompute the per-bar policy state used by `run`:

    - entry_blocked: bool[n] of bars where NEW entries are disallowed
      (skip-hours + news-blackout windows). None when no static filter
      applies, so the caller can skip the check.
    - bar_ms: int[n] of each bar's UTC ms timestamp — used to age out
      the 24h rolling trade-cap window.
    - day_key: int[n] of each bar's UTC date as an integer — drives the
      consecutive-loss circuit breaker's daily reset.

    All three are O(n) and constructed once; the inner loop is O(1) per
    bar against them.
    """
    n = len(index)
    if not isinstance(index, pd.DatetimeIndex):
        # Synthetic / non-temporal index — no policy filter can apply
        # meaningfully. Return all-zero day keys + zero timestamps; the
        # loop will treat the entire series as a single "day."
        return None, np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)

    utc_idx = index.tz_convert("UTC") if index.tz is not None else index
    bar_ms = (utc_idx.asi8 // 1_000_000).astype(np.int64)
    day_key = (bar_ms // (24 * 60 * 60 * 1000)).astype(np.int64)

    blocked = np.zeros(n, dtype=bool)
    any_block = False

    if policy.skip_hours_utc:
        hours = utc_idx.hour.astype(np.int64).to_numpy()
        skip_set = {int(h) for h in policy.skip_hours_utc}
        blocked |= np.isin(hours, list(skip_set))
        any_block = True

    if policy.news_timestamps_ms:
        window_ms = max(1, int(policy.news_skip_minutes)) * 60 * 1000
        for ts_ms in policy.news_timestamps_ms:
            ts = int(ts_ms)
            blocked |= (bar_ms >= ts - window_ms) & (bar_ms <= ts + window_ms)
        any_block = True

    return (blocked if any_block else None), bar_ms, day_key


def _summarize(df: pd.DataFrame, trades: pd.DataFrame, bars_per_year: int) -> dict:
    """DataFrame-facing summary — delegates to the array summarizer so the metrics
    are computed by exactly one code path (the metrics-only backtest reuses it)."""
    return _summarize_arrays(
        df["equity"].to_numpy(),
        df["net_ret"].to_numpy(),
        df["position"].to_numpy(),
        df["trade_cost"].to_numpy(),
        df["funding_cost"].to_numpy(),
        trades, bars_per_year,
    )


def _summarize_arrays(equity: np.ndarray, net: np.ndarray, position: np.ndarray,
                      trade_cost: np.ndarray, funding_cost: np.ndarray,
                      trades: pd.DataFrame, bars_per_year: int) -> dict:
    """Compute backtest metrics straight from the per-bar arrays — no intermediate
    output DataFrame. Reductions mirror pandas exactly (NaN-skipping; std uses
    ddof=1) so results are identical to summarizing the assembled frame."""
    eq = equity
    n = len(eq)
    # Keep last_eq as a NumPy scalar (not a Python float): the cagr power below can
    # overflow, and np.errstate only suppresses that for NumPy ops — a Python float
    # ** would raise OverflowError instead of yielding inf (then folded to nan).
    last_eq = eq[-1] if n else 0.0
    total = (last_eq - 1.0) if n else 0.0
    years = n / bars_per_year if bars_per_year else 0
    # Annualised return. Short windows raised to a large 1/years power can
    # overflow to +inf (which would then be nulled downstream, silently dropping
    # the metric); compute it overflow-safe and fold any non-finite result to nan.
    with np.errstate(over="ignore", invalid="ignore"):
        cagr = (last_eq ** (1.0 / years) - 1.0) if (years > 0 and last_eq > 0) else float("nan")
    if not np.isfinite(cagr):
        cagr = float("nan")

    mu = (np.nanmean(net) if n else float("nan")) * bars_per_year
    # pandas Series.std() defaults to ddof=1 and skips NaN — match both.
    sd = (np.nanstd(net, ddof=1) if n else float("nan")) * np.sqrt(bars_per_year)
    sharpe = mu / sd if sd > 0 else float("nan")

    peak = np.maximum.accumulate(eq) if n else eq
    with np.errstate(divide="ignore", invalid="ignore"):
        peak_safe = np.where(peak == 0, np.nan, peak)
        dd = (eq / peak_safe) - 1.0
    max_dd = float(np.nanmin(dd)) if (n and not np.all(np.isnan(dd))) else float("nan")

    n_trades = len(trades)
    win_rate = (trades["ret"] > 0).mean() if n_trades else float("nan")
    avg_trade = trades["ret"].mean() if n_trades else float("nan")

    # Count exits by kind
    sl_n = int((trades["exit"] == "sl").sum()) if n_trades else 0
    tp_n = int((trades["exit"] == "tp").sum()) if n_trades else 0
    flip_n = int((trades["exit"] == "flip").sum()) if n_trades else 0
    flat_n = int((trades["exit"] == "flat").sum()) if n_trades else 0

    # Turnover on a per-bar basis (sum of |change|) — np.diff drops the leading
    # bar, matching pandas .diff().abs().sum() (which skips the leading NaN).
    turnover = float(np.nansum(np.abs(np.diff(position)))) if n > 1 else 0.0
    exposure = float((np.abs(position) > 0).mean()) if n else 0.0
    total_cost = float(np.nansum(trade_cost) + np.nansum(funding_cost))

    return {
        "bars": n,
        "total_return": float(total),
        "cagr": float(cagr) if not (isinstance(cagr, float) and np.isnan(cagr)) else float("nan"),
        "sharpe": float(sharpe) if not np.isnan(sharpe) else float("nan"),
        "max_drawdown": float(max_dd) if not np.isnan(max_dd) else float("nan"),
        "trades": n_trades,
        "win_rate": float(win_rate) if not np.isnan(win_rate) else float("nan"),
        "avg_trade_return": float(avg_trade) if not np.isnan(avg_trade) else float("nan"),
        "turnover": float(turnover),
        "exposure": float(exposure),
        "total_cost": float(total_cost),
        "exits_sl": sl_n, "exits_tp": tp_n, "exits_flip": flip_n, "exits_flat": flat_n,
    }


def _buy_and_hold(close: pd.Series, bars_per_year: int) -> dict:
    if len(close) < 2:
        return {"total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "sharpe": float("nan")}
    ret = close.pct_change().fillna(0.0)
    eq = (1.0 + ret).cumprod()
    years = len(close) / bars_per_year
    with np.errstate(over="ignore", invalid="ignore"):
        cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and eq.iloc[-1] > 0 else float("nan")
    if not np.isfinite(cagr):
        cagr = float("nan")
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    sd = ret.std() * np.sqrt(bars_per_year)
    mu = ret.mean() * bars_per_year
    sharpe = mu / sd if sd > 0 else float("nan")
    return {
        "total_return": float(eq.iloc[-1] - 1.0),
        "cagr": float(cagr) if not np.isnan(cagr) else float("nan"),
        "max_drawdown": float(dd.min()),
        "sharpe": float(sharpe),
    }
