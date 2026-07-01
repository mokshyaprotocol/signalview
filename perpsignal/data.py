"""Binance Futures public-data fetchers with on-disk parquet cache.

We only use Binance for the backtest data layer:
  - klines (price + volume) — full multi-year history
  - fundingRate — full history
  - openInterestHist — only ~30 days (Binance limitation); we fill NaN for older bars

All endpoints are public (no API key required).
"""

from __future__ import annotations

import concurrent.futures
import io
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .config import DATA_DIR
from . import upstash_cache as _uc

FAPI = "https://fapi.binance.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "perpsignal/0.1"})

# How many milliseconds in one bar of a given interval
_INTERVAL_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
}


def _utc_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _get(url: str, params: dict, retries: int | None = None) -> list | dict:
    # Default attempts favour ROBUST real-time data — this is the path that feeds
    # live agent ticks (agent_tick builds in-process) and backtests, where we'd
    # rather ride out a multi-second Binance wobble (5 → 1+2+4+8+16 = 31s) than
    # skip a tick or drop bars. Env-tunable: the *dashboard* function lowers it to
    # fail fast (api/recommend.py sets PERPSIGNAL_BINANCE_RETRIES=3) because that
    # endpoint has the circuit breaker + last-good stale cache as its safety net
    # and must not spin into the 60s function cap; recommend and agent_tick are
    # SEPARATE Vercel functions (separate processes), so the env default each
    # process picks is independent. Bulk offline jobs can bump it higher still.
    if retries is None:
        retries = int(os.environ.get("PERPSIGNAL_BINANCE_RETRIES") or 5)
    backoff = 1.0
    last_err = "?"
    for attempt in range(retries):
        # A transport fault (connection reset, DNS hiccup, read timeout) is a
        # transient upstream wobble just like a 429/5xx — retry it on the same
        # backoff instead of letting it propagate immediately. Without this a
        # single dropped connection to Binance turns into a hard 5xx upstream
        # (recommend → 503) even though the next attempt would succeed.
        try:
            r = SESSION.get(url, params=params, timeout=30)
        except requests.exceptions.RequestException as e:                   # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(backoff)
            backoff *= 2
            continue
        if r.status_code == 200:
            return r.json()
        # 429 rate-limited or 5xx → backoff
        if r.status_code in (418, 429) or r.status_code >= 500:
            last_err = f"{r.status_code} {r.text[:200]}"
            time.sleep(backoff)
            backoff *= 2
            continue
        r.raise_for_status()
    raise RuntimeError(f"GET {url} failed after {retries} retries: {last_err}")


# ---------------------------------------------------------------------------
# Hyperliquid public-info fallback
#
# 44 Hyperliquid perps aren't listed on Binance USD-M futures, so Binance
# returns no klines/funding for them. Hyperliquid's own /info endpoint serves
# OHLCV (candleSnapshot) + funding (fundingHistory) for every perp it lists, so
# we transparently fall back to it when Binance has no data. Results flow through
# the SAME disk/Upstash cache as the Binance path (the fallback runs inside
# fetch_klines/fetch_funding, before _write_cache), so a given (symbol, interval,
# bar) is fetched from HL at most once — no rate-limit pressure.
# ---------------------------------------------------------------------------
HL_INFO = "https://api.hyperliquid.xyz/info"

# Polite pacing for HL /info: a process-wide minimum spacing between calls.
# HL's REST budget is weight-based per IP (~1200/min; candleSnapshot and
# friends weigh a few units each) and is SHARED with live agent ticks, so a
# burst here (multi-TF dataset builds, warm slices) must never starve a
# trade. ~7 req/s per warm container stays far inside the budget while
# adding at most ~100ms × calls of latency to a cold multi-TF build; the
# 429-backoff below remains the hard backstop. (0.15 made a fully-cold
# xyz recommend graze the function cap — observed FUNCTION_INVOCATION_TIMEOUT
# on the first uncached call.)
# Both knobs are env-tunable for BULK offline jobs (the DNA seeder walks
# hundreds of pairs and prefers slower+stubborner over serverless-snappy):
#   PERPSIGNAL_HL_MIN_INTERVAL — seconds between calls (default 0.10)
#   PERPSIGNAL_HL_RETRIES      — attempts per call      (default 4)
_HL_MIN_INTERVAL_S = float(os.environ.get("PERPSIGNAL_HL_MIN_INTERVAL") or 0.10)
_hl_pace_lock = threading.Lock()
_hl_last_call = 0.0


def _hl_pace() -> None:
    global _hl_last_call
    with _hl_pace_lock:
        wait = _hl_last_call + _HL_MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _hl_last_call = time.monotonic()


def _hl_post(body: dict, retries: int | None = None) -> list | dict:
    if retries is None:
        retries = int(os.environ.get("PERPSIGNAL_HL_RETRIES") or 4)
    backoff = 1.0
    last_err = "?"
    for _ in range(retries):
        _hl_pace()
        # Transport faults (reset/timeout/DNS) are transient — retry on the same
        # backoff as a 429/5xx rather than raising on the first dropped socket.
        try:
            r = SESSION.post(HL_INFO, json=body, timeout=30)
        except requests.exceptions.RequestException as e:                   # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(backoff)
            backoff *= 2
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429 or r.status_code >= 500:
            last_err = f"{r.status_code} {r.text[:200]}"
            time.sleep(backoff)
            backoff *= 2
            continue
        r.raise_for_status()
    raise RuntimeError(f"HL POST {body.get('type')} failed: {last_err}")


def _hl_coin(symbol: str) -> str:
    """Map a Binance-style symbol (FTTUSDT) to a Hyperliquid coin (FTT). Preserves
    case so HL's k-prefixed perps (kPEPEUSDT → kPEPE) resolve to the exact coin
    name HL expects — uppercasing would break them (KPEPE is not a market).
    Namespaced HIP-3 coins ("xyz:TSLA") pass through whole."""
    return symbol[:-4] if symbol.upper().endswith("USDT") else symbol


def _is_builder_dex(symbol: str) -> bool:
    """True for namespaced HIP-3 builder-dex coins ("xyz:TSLA" — trade.xyz
    TradFi markets). Those never exist on Binance, so the fetchers skip the
    doomed Binance attempt and go straight to Hyperliquid."""
    return ":" in (symbol or "")


def _hl_dex(symbol: str) -> str:
    return symbol.split(":", 1)[0].lower() if _is_builder_dex(symbol) else ""


def _hl_klines_rows(coin: str, interval: str, start_ms: int, end_ms: int) -> list[list]:
    """Fetch HL candleSnapshot and shape rows like Binance klines. HL omits
    quote/taker volume — quote_volume is approximated as base_volume × close and
    taker fields are 0 (the factor set doesn't read taker volume)."""
    bar_ms = _INTERVAL_MS.get(interval, _INTERVAL_MS["1h"])
    rows: list[list] = []
    cur = start_ms
    while cur < end_ms:
        batch = _hl_post({"type": "candleSnapshot", "req": {
            "coin": coin, "interval": interval, "startTime": cur, "endTime": end_ms,
        }})
        if not isinstance(batch, list) or not batch:
            break
        for c in batch:
            o, h, l, cl = float(c["o"]), float(c["h"]), float(c["l"]), float(c["c"])
            v = float(c["v"])
            rows.append([int(c["t"]), o, h, l, cl, v, int(c["T"]),
                         v * cl, int(c.get("n", 0) or 0), 0.0, 0.0, 0])
        last_open = int(batch[-1]["t"])
        nxt = last_open + bar_ms
        if nxt <= cur:
            break
        cur = nxt
        if len(batch) < 5000:
            break
    return rows


def _hl_funding_rows(coin: str, start_ms: int, end_ms: int) -> list[dict]:
    rows: list[dict] = []
    cur = start_ms
    while cur < end_ms:
        batch = _hl_post({"type": "fundingHistory", "coin": coin, "startTime": cur, "endTime": end_ms})
        if not isinstance(batch, list) or not batch:
            break
        for f in batch:
            rows.append({"fundingTime": int(f["time"]), "fundingRate": float(f["fundingRate"])})
        last = int(batch[-1]["time"])
        if last + 1 <= cur:
            break
        cur = last + 1
        if len(batch) < 500:
            break
    return rows


def _cache_keys(name: str, symbol: str, interval: str, start: datetime) -> tuple[Path, str, str]:
    """Returns (disk_path, upstash_key, bar_suffix) for a given fetch.

    Suffix encodes the current bar boundary for `interval`. As soon as a
    new bar opens (i.e., the previous one fully closed), the suffix
    changes → cache miss on both disk and Upstash → we refetch. Suffix
    is shared between paths so a single change ages every layer at once.

    The Upstash key omits `start` since Upstash is keyed on the rolling
    snapshot rather than the calendar slice — a 30-day vs 60-day
    request lands on the same snapshot bucket, the longer one just
    pre-fetches from Binance once and the cache covers both. Disk path
    keeps `start` for back-compat with the existing parquet directory.
    """
    bar_ms = _INTERVAL_MS.get(interval, _INTERVAL_MS["1h"])
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    snapped_ms = (now_ms // bar_ms) * bar_ms
    snap_dt = datetime.fromtimestamp(snapped_ms / 1000, tz=timezone.utc)
    suffix = snap_dt.strftime("%Y%m%dT%H%MZ")
    # Builder-dex coins carry a ':' (xyz:TSLA) — fine in a Redis key, not in
    # every filesystem. Sanitise the DISK name only, keeping the prefix so
    # a same-named coin on two dexes can never share a cache file.
    fs_symbol = symbol.replace(":", "-")
    # gzip-pickle (not parquet): pandas-native, so the serverless bundle doesn't
    # ship pyarrow (~126MB). Staying under Vercel's bundle limit avoids the
    # cold-start runtime dep-install that was timing out heavy recommend calls.
    disk_key = f"{name}_{fs_symbol}_{interval}_{start.date()}_to_{suffix}.pkl.gz"
    return DATA_DIR / disk_key, _uc.upstash_key(name, symbol, interval, suffix), suffix


def _cache_path(name: str, symbol: str, interval: str, start: datetime, end: datetime) -> Path:
    """Back-compat shim. Existing callers (older code outside this file)
    expect the single-Path return; keep them working while migrating."""
    path, _u, _s = _cache_keys(name, symbol, interval, start)
    return path


def _read_cached(disk_path: Path, upstash_key: str) -> Optional[pd.DataFrame]:
    """Try Upstash first (works in prod across cold containers), then
    disk (works in dev / on a long-running worker). Returns None on a
    full miss so the caller falls through to the network."""
    # Upstash hit?
    blob = _uc.cache_get(upstash_key)
    if blob:
        try:
            return pd.read_pickle(io.BytesIO(blob), compression="gzip")
        except Exception:                                              # noqa: BLE001
            # Corrupt or legacy-parquet blob — fall through and re-fetch
            # (the next write overwrites it with the gzip-pickle format).
            pass
    if disk_path.exists():
        try:
            df = pd.read_pickle(disk_path)  # gzip inferred from .pkl.gz
            # Warm Upstash from disk so the next cold container in
            # prod inherits the cache without another Binance trip.
            try:
                _uc.cache_put(upstash_key, disk_path.read_bytes())
            except Exception:                                          # noqa: BLE001
                pass
            return df
        except Exception:                                              # noqa: BLE001
            pass
    return None


def _write_cache(df: pd.DataFrame, disk_path: Path, upstash_key: str) -> None:
    """Persist to both disk and Upstash. Disk first (the durable store),
    Upstash second (best-effort)."""
    disk_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(disk_path)  # gzip-pickle (compression inferred from .pkl.gz)
    try:
        _uc.cache_put(upstash_key, disk_path.read_bytes())
    except Exception:                                                  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Klines (price + volume)
# ---------------------------------------------------------------------------
def fetch_klines(symbol: str, interval: str, start: datetime, end: datetime,
                 use_cache: bool = True) -> pd.DataFrame:
    disk_path, ukey, _suffix = _cache_keys("klines", symbol, interval, start)
    if use_cache:
        hit = _read_cached(disk_path, ukey)
        if hit is not None:
            return hit

    bar_ms = _INTERVAL_MS[interval]
    start_ms = _utc_ms(start)
    end_ms = _utc_ms(end)
    rows: list[list] = []
    # Builder-dex coins (xyz:TSLA) are HL-native — skip the doomed Binance call.
    if not _is_builder_dex(symbol):
        try:
            cur = start_ms
            while cur < end_ms:
                batch = _get(f"{FAPI}/fapi/v1/klines", {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cur,
                    "endTime": end_ms,
                    "limit": 1500,
                })
                if not batch:
                    break
                rows.extend(batch)
                last_open = batch[-1][0]
                next_cur = last_open + bar_ms
                if next_cur <= cur:
                    break
                cur = next_cur
                if len(batch) < 1500:
                    break
        except Exception:                                               # noqa: BLE001
            rows = []  # symbol not on Binance — try the HL fallback below
    # Hyperliquid-native perp (builder-dex or not listed on Binance USD-M)
    # → HL candleSnapshot.
    if not rows:
        rows = _hl_klines_rows(_hl_coin(symbol), interval, start_ms, end_ms)

    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ("open", "high", "low", "close", "volume",
              "quote_volume", "taker_buy_base", "taker_buy_quote"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.drop(columns=["ignore"]).drop_duplicates(subset=["open_time"]).sort_values("open_time")
    df = df.set_index("open_time")

    # Drop any bar that hasn't fully closed yet. Binance returns the in-progress
    # bar with its `close` set to the current price — using it would constitute
    # look-ahead bias (the signal would see "the future" of an unfinished bar).
    # A bar with open_time t is fully closed when t + bar_duration <= now.
    bar_ms = _INTERVAL_MS[interval]
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(milliseconds=bar_ms)
    df = df[df.index <= cutoff]

    _write_cache(df, disk_path, ukey)
    return df


# ---------------------------------------------------------------------------
# Funding rate (per-funding-event series, ~every 8h)
# ---------------------------------------------------------------------------
def fetch_funding(symbol: str, start: datetime, end: datetime,
                  use_cache: bool = True) -> pd.DataFrame:
    disk_path, ukey, _suffix = _cache_keys("funding", symbol, "8h", start)
    if use_cache:
        hit = _read_cached(disk_path, ukey)
        if hit is not None:
            return hit

    rows: list[dict] = []
    start_ms = _utc_ms(start)
    end_ms = _utc_ms(end)
    # Builder-dex coins (xyz:TSLA) are HL-native — skip the doomed Binance call.
    if not _is_builder_dex(symbol):
        try:
            cur = start_ms
            while cur < end_ms:
                batch = _get(f"{FAPI}/fapi/v1/fundingRate", {
                    "symbol": symbol,
                    "startTime": cur,
                    "endTime": end_ms,
                    "limit": 1000,
                })
                if not batch:
                    break
                rows.extend(batch)
                last = batch[-1]["fundingTime"]
                if last + 1 <= cur:
                    break
                cur = last + 1
                if len(batch) < 1000:
                    break
        except Exception:                                               # noqa: BLE001
            rows = []  # symbol not on Binance — try the HL fallback below
    # Hyperliquid-native perp → HL fundingHistory (already shaped as fundingTime/Rate).
    if not rows:
        rows = _hl_funding_rows(_hl_coin(symbol), start_ms, end_ms)

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["fundingTime", "fundingRate"])
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df[["fundingTime", "fundingRate"]].drop_duplicates(subset=["fundingTime"]).sort_values("fundingTime")
    df = df.set_index("fundingTime")

    _write_cache(df, disk_path, ukey)
    return df


# ---------------------------------------------------------------------------
# Open Interest history (~last 30 days only on this endpoint)
# ---------------------------------------------------------------------------
def fetch_open_interest(symbol: str, interval: str, start: datetime, end: datetime,
                        use_cache: bool = True) -> pd.DataFrame:
    # Builder-dex coins have NO historical OI source (HL only serves a live
    # snapshot; Binance doesn't list them). Return the empty frame the
    # factor layer already treats as "OI factor off" — zero network calls.
    if _is_builder_dex(symbol):
        return pd.DataFrame(
            columns=["timestamp", "sumOpenInterest", "sumOpenInterestValue"],
        ).set_index("timestamp")
    disk_path, ukey, _suffix = _cache_keys("oi", symbol, interval, start)
    if use_cache:
        hit = _read_cached(disk_path, ukey)
        if hit is not None:
            return hit

    # Endpoint supports 5m,15m,30m,1h,2h,4h,6h,12h,1d
    bar_ms = _INTERVAL_MS.get(interval, _INTERVAL_MS["1h"])
    rows: list[dict] = []
    # Binance only returns last ~30 days; clamp start.
    min_start = datetime.now(timezone.utc) - timedelta(days=29)
    eff_start = max(start, min_start)
    cur = _utc_ms(eff_start)
    end_ms = _utc_ms(end)
    while cur < end_ms:
        batch = _get(f"{FAPI}/futures/data/openInterestHist", {
            "symbol": symbol,
            "period": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": 500,
        })
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1]["timestamp"]
        next_cur = last_ts + bar_ms
        if next_cur <= cur:
            break
        cur = next_cur
        if len(batch) < 500:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["timestamp", "sumOpenInterest", "sumOpenInterestValue"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for c in ("sumOpenInterest", "sumOpenInterestValue"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    _write_cache(df, disk_path, ukey)
    return df


# ---------------------------------------------------------------------------
# Combined dataset aligned to klines bar index
# ---------------------------------------------------------------------------
def build_dataset(symbol: str, interval: str, start: datetime, end: datetime,
                  use_cache: bool = True, include_oi: bool = True) -> pd.DataFrame:
    """Return a DataFrame indexed at `interval` bars containing OHLCV + funding + OI.

    The three endpoints (klines, funding, OI) are fetched IN PARALLEL via a
    thread pool — this is pure I/O wait, so threading is a clean win.

    `include_oi=False` skips the open-interest fetch entirely. Useful on the
    live API path: Binance only serves ~30d of OI history, so the OI factor
    falls back to 0 most of the time anyway. Saves one round-trip per TF.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        kl_f = ex.submit(fetch_klines, symbol, interval, start, end, use_cache)
        fr_f = ex.submit(fetch_funding, symbol, start, end, use_cache)
        oi_f = ex.submit(fetch_open_interest, symbol, interval, start, end, use_cache) if include_oi else None
        kl = kl_f.result()
        fr = fr_f.result()
        oi = oi_f.result() if oi_f is not None else None

    df = kl[["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote"]].copy()

    # Funding: realized at the funding timestamp; align by forward-fill so a
    # bar only knows past funding rates (no look-ahead leak).
    if fr is not None and not fr.empty:
        df["funding_rate"] = fr["fundingRate"].reindex(df.index, method="ffill")
    else:
        df["funding_rate"] = float("nan")

    if oi is not None and not oi.empty:
        df["open_interest"] = oi["sumOpenInterest"].reindex(df.index, method="nearest", tolerance=pd.Timedelta(hours=1))
    else:
        df["open_interest"] = float("nan")

    return df


# ---------------------------------------------------------------------------
# Live ticker (real-time last trade price). Never cached — always fresh.
# ---------------------------------------------------------------------------
def fetch_ticker_price(symbol: str) -> tuple[float, int]:
    """Return (last_trade_price, server_time_ms) for `symbol`.

    Hits Binance's lightweight ticker/price endpoint. Typical round-trip ~80ms.
    """
    if _is_builder_dex(symbol):
        # HL-native: allMids takes the dex name and keys by the namespaced
        # coin ("xyz:TSLA"). No Binance equivalent exists.
        mids = _hl_post({"type": "allMids", "dex": _hl_dex(symbol)})
        px = float(mids[symbol]) if isinstance(mids, dict) and mids.get(symbol) else 0.0
        return px, int(time.time() * 1000)
    try:
        data = _get(f"{FAPI}/fapi/v1/ticker/price", {"symbol": symbol})
        return float(data["price"]), int(data.get("time", 0))
    except Exception:                                                   # noqa: BLE001
        # Hyperliquid-only perp → HL mid price.
        mids = _hl_post({"type": "allMids"})
        coin = _hl_coin(symbol)
        px = float(mids[coin]) if isinstance(mids, dict) and mids.get(coin) else 0.0
        return px, int(time.time() * 1000)


def funding_cost_per_bar(funding: pd.Series, bars_per_funding: int = 8) -> pd.Series:
    """Spread the 8h funding payment across the bars in that 8h window.

    funding is the raw funding rate at each bar (forward-filled from the last event).
    Returns a per-bar cost (positive = long pays, negative = long receives).
    """
    return funding.fillna(0.0) / bars_per_funding
