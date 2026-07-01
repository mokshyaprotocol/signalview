"""The in-process MTF feature cache must (a) avoid rebuilding the same
(market, window, cfg) stack, (b) return equal data, and (c) be mutation-safe
so a consumer editing its frame can't corrupt the cached entry.
"""

from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

import perpsignal.mtf as mtf
from perpsignal.config import FactorConfig


def _synth_dataset(symbol, interval, start, end, with_funding=True, include_oi=True):
    """Stand-in for data.build_dataset: a deterministic OHLCV frame long enough
    for the EMA200 / ADX windows the factor stack needs."""
    n = 400
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    base = 100 + np.sin(np.linspace(0, 12, n)) * 5 + np.linspace(0, 8, n)
    df = pd.DataFrame({
        "open": base, "high": base * 1.004, "low": base * 0.996,
        "close": base, "volume": np.linspace(1000, 2000, n),
        "funding_rate": np.full(n, 0.0001), "open_interest": np.linspace(1e6, 2e6, n),
    }, index=idx)
    return df


@pytest.fixture(autouse=True)
def _patch_build_dataset(monkeypatch):
    mtf.clear_mtf_cache()
    calls = {"n": 0}

    def counting(symbol, interval, start, end, with_funding=True, include_oi=True):
        calls["n"] += 1
        return _synth_dataset(symbol, interval, start, end, with_funding, include_oi)

    monkeypatch.setattr(mtf, "build_dataset", counting)
    return calls


def _args():
    end = datetime(2024, 6, 1, tzinfo=timezone.utc)
    start = end - timedelta(days=10)
    return ("BTCUSDT", "1h", start, end)


def test_second_identical_build_is_served_from_cache(_patch_build_dataset):
    sym, tf, start, end = _args()
    mtf.build_mtf_features(sym, tf, start, end, FactorConfig(), include_oi=True)
    first = _patch_build_dataset["n"]
    assert first >= 1  # at least the primary tier was fetched
    mtf.build_mtf_features(sym, tf, start, end, FactorConfig(), include_oi=True)
    assert _patch_build_dataset["n"] == first  # no extra build_dataset calls


def test_cache_returns_equal_data(_patch_build_dataset):
    sym, tf, start, end = _args()
    d1, f1, r1 = mtf.build_mtf_features(sym, tf, start, end, FactorConfig())
    d2, f2, r2 = mtf.build_mtf_features(sym, tf, start, end, FactorConfig())
    pd.testing.assert_frame_equal(d1, d2)
    pd.testing.assert_frame_equal(f1, f2)
    pd.testing.assert_series_equal(r1, r2)


def test_cache_is_mutation_safe(_patch_build_dataset):
    sym, tf, start, end = _args()
    d1, f1, r1 = mtf.build_mtf_features(sym, tf, start, end, FactorConfig())
    # Corrupt the returned frames; the cached copy must be untouched.
    d1.loc[:, "close"] = -999.0
    f1.loc[:, "rsi"] = -999.0
    d3, f3, r3 = mtf.build_mtf_features(sym, tf, start, end, FactorConfig())
    assert (d3["close"] != -999.0).all()
    assert (f3["rsi"] != -999.0).all()


def test_distinct_window_is_a_cache_miss(_patch_build_dataset):
    sym, tf, start, end = _args()
    mtf.build_mtf_features(sym, tf, start, end, FactorConfig())
    after_first = _patch_build_dataset["n"]
    mtf.build_mtf_features(sym, tf, start - timedelta(days=1), end, FactorConfig())
    assert _patch_build_dataset["n"] > after_first  # different start -> rebuilt
