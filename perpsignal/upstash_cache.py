"""Upstash REST shim for caching market-data (klines/funding) parquet blobs
across serverless invocations.

On Vercel each function invocation gets a fresh `/tmp`, so without a shared
cache the first publish/recommend in a cold container re-fetches all the bars
from Binance/HL. This module stores each (symbol, interval, window) parquet
frame under a deterministic key so the next cold container reads it in ~50ms
instead of ~3000ms.

IMPORTANT — dedicated cache DB:
These frames are large (~100-600KB each) and there are #tokens × #TFs of them,
so they must NOT share the persistent database (signals/agents/earnings), or
they fill its quota and break writes. The cache therefore lives on its OWN
Upstash database, configured via UPSTASH_CACHE_REST_URL / UPSTASH_CACHE_REST_TOKEN.
Provision that DB with eviction enabled (maxmemory-policy = allkeys-lru): when it
fills it drops the oldest frames instead of erroring, so it self-bounds forever.

When the cache env vars aren't set the cache is simply disabled (the on-disk
parquet path still runs). It never falls back to the persistent DB — that's the
whole point: the persistent store stays tiny.
"""
from __future__ import annotations

import base64
import os
from typing import Optional

import requests


_DISABLED_LOGGED = False


def _cache_cfg() -> tuple[str, str]:
    url = os.environ.get("UPSTASH_CACHE_REST_URL", "").strip().rstrip("/")
    tok = os.environ.get("UPSTASH_CACHE_REST_TOKEN", "").strip()
    return url, tok


def _enabled() -> bool:
    """True only when the DEDICATED cache DB is configured. Logged once per
    process so a misconfigured deploy is visible in the function logs."""
    global _DISABLED_LOGGED
    url, tok = _cache_cfg()
    ok = bool(url and tok)
    if not ok and not _DISABLED_LOGGED:
        _DISABLED_LOGGED = True
        print("[upstash_cache] disabled — UPSTASH_CACHE_REST_{URL,TOKEN} not set; "
              "using on-disk parquet cache only. The market-data cache must run on "
              "a DEDICATED Upstash DB (eviction = allkeys-lru) so it never fills the "
              "persistent quota.", flush=True)
    return ok


def _cache_cmd(*args) -> object:
    """One Redis command against the dedicated cache DB via Upstash REST.
    Self-contained (does NOT use agentkit.store, which targets the persistent
    DB) so the two stores stay fully isolated."""
    url, tok = _cache_cfg()
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {tok}"},
        json=[str(a) for a in args],
        timeout=10,
    )
    if r.status_code != 200:
        raise RuntimeError(f"cache HTTP {r.status_code}: {r.text[:160]}")
    body = r.json()
    if "error" in body:
        raise RuntimeError(f"cache error: {body['error']}")
    return body.get("result")


def cache_get(key: str) -> Optional[bytes]:
    """Return the raw bytes stored under `key`, or None on miss / error.
    Errors are swallowed — the data.py call site falls through to a re-fetch,
    so a transient cache blip never breaks a backtest."""
    if not _enabled():
        return None
    try:
        raw = _cache_cmd("GET", key)
        if not raw:
            return None
        return base64.b64decode(raw)  # stored base64 so the JSON value stays a string
    except Exception:                                                   # noqa: BLE001
        return None


def cache_put(key: str, data: bytes, ttl_seconds: int = 6 * 3600) -> None:
    """Write `data` under `key` with `ttl_seconds` expiry. Default 6h — the
    bar-snapshot suffix flips on each closed bar so older keys go stale; the TTL
    is the cleanup safety net. Oversized frames (>900KB) are dropped silently
    (the on-disk parquet cache is the durable copy). With eviction enabled on
    the cache DB, the total size self-bounds regardless of TTL."""
    if not _enabled():
        return
    if len(data) > 900_000:
        return
    try:
        encoded = base64.b64encode(data).decode("ascii")
        _cache_cmd("SET", key, encoded, "EX", str(ttl_seconds))
    except Exception:                                                   # noqa: BLE001
        pass


def upstash_key(name: str, symbol: str, interval: str, suffix: str) -> str:
    """Single source of truth for the cache key shape."""
    return f"bars:v1:{name}:{symbol}:{interval}:{suffix}"
