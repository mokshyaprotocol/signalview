"""perpsignal.signal_def — signal definition parsing & validation.

The SignalDef dataclass, parse_signal(), and all the validation bounds.
Dependency-free (stdlib only): describes a trading signal as data so it can be
serialised, validated, and handed to the backtester. Asset validation uses the
static VALID_ASSETS list by default; call set_market_meta() to validate against a
live exchange universe instead.
"""

from __future__ import annotations

import re
import json
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

# has no dependency on the pandas-heavy signal package (which keeps the
# Redis-only callers light — agent_create, agent_status).
#
# _slow factors are the slower-TF variants of RSI / EMA-cross / MACD —
# sourced from the slower tier (e.g. 4h on a 1h primary). Authors opt
# in by giving them non-zero weight; legacy signals without these
# keys keep working because parse_signal() reads with .get(f, 0.0).
FACTOR_ORDER = (
    "rsi", "ema_cross", "macd", "trend", "volume", "oi", "funding",
    "rsi_slow", "ema_cross_slow", "macd_slow",
    # Phase 1 audit additions — keep in sync with perpsignal.config.FACTOR_ORDER.
    # All new keys default to 0 weight in REGIME_WEIGHTS so legacy signals
    # (which never sent these keys) keep working — parse_signal reads each
    # row with .get(f, 0.0).
    "slope_regression", "adx_strength", "rsi_delta",
    "volume_zscore", "atr_stability", "bb_width", "vwap_distance",
)
VALID_REGIMES = ("trend_up", "trend_down", "chop")
VALID_ASSETS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT")
VALID_TFS = ("5m", "15m", "1h", "4h", "1d")

# Optional live-market-metadata hook. Standalone, the package validates assets
# against the static VALID_ASSETS list above. A host app can inject a provider
# exposing `canonical_coin(str) -> str | None` and `is_listed(str) -> bool` to
# accept the full live exchange universe (e.g. every Hyperliquid perp) instead of
# just the static four. See set_market_meta().
_MARKET_META = None


def set_market_meta(provider) -> None:
    """Inject a market-metadata provider to validate assets against a live
    exchange universe. `provider` must expose canonical_coin() and is_listed().
    Pass None to revert to static VALID_ASSETS validation."""
    global _MARKET_META
    _MARKET_META = provider


def _normalize_asset(raw: str) -> str:
    """Canonicalise an asset symbol to <COIN>USDT, upper-casing the coin EXCEPT
    HL's 1000x meme perps, which HL lists with a lowercase-k prefix (kPEPE, kBONK,
    kSHIB). Preserving that exact case matters: the backtest's HL candleSnapshot is
    case-sensitive and 500s on "KPEPE". A bare "kXXXX" (lowercase k + upper rest) is
    kept as-is; everything else is upper-cased.

    HIP-3 builder-dex coins ("xyz:TSLA" — trade.xyz TradFi markets) are
    namespaced and case-sensitive as a whole: lowercase dex prefix, exact
    ticker case. They never carry a USDT suffix; canonicalise through HL's
    own listing (tolerates "XYZ:tsla" input) and keep unknowns as
    dex-lowered/ticker-uppered so the downstream is_listed check rejects
    them with a clear error instead of a garbled name."""
    s = (str(raw or "BTCUSDT").strip() or "BTCUSDT")
    if ":" in s:
        try:
            canonical = _MARKET_META.canonical_coin(s) if _MARKET_META else None
        except Exception:  # noqa: BLE001
            canonical = None
        if canonical:
            return canonical
        dex, _, ticker = s.partition(":")
        return f"{dex.lower()}:{ticker.upper()}"
    up = s.upper()
    coin = s[:-4] if up.endswith("USDT") else s
    coin = coin if (coin[:1] == "k" and coin[1:].isupper()) else coin.upper()
    return coin + ("USDT" if up.endswith("USDT") else "")

# Bounds borrowed from RegimeConfig / RiskConfig in perpsignal.config —
# kept loose so users can experiment, but tight enough to catch
# nonsense before a backtest churn.
MAX_ABS_WEIGHT = 5.0
MAX_THRESHOLD = 1.0
MAX_LEVERAGE = 50.0
MAX_TP_PCT = 0.50
MAX_SL_PCT = 0.20
MAX_MIN_HOLD = 100
NAME_MAX = 80
DESC_MAX = 400
# Policy filters / circuit-breakers — bounded so a runaway prompt can't ship
# a definition with 10k skip hours or 100 news timestamps and balloon the
# Redis record. Numbers tuned for the highest-frequency timeframe (5m =
# 288 bars/day): max 24 skip-hour entries, max 200 news timestamps.
MAX_SKIP_HOURS = 24
MAX_NEWS_TIMESTAMPS = 200
NEWS_SKIP_MAX_MINUTES = 240   # ±4h is the widest the engine will honour
MAX_TRADES_CAP = 100          # absolute ceiling on max_trades_per_24h
MAX_CONSEC_LOSSES = 20        # absolute ceiling on circuit breaker


# ---------------------------------------------------------------------------


@dataclass
class SignalDef:
    """The user-facing parameterization. Serializable to JSON and stable
    under signal_id hashing."""

    weights: dict[str, dict[str, float]]
    long_threshold: float
    short_threshold: float
    take_profit_pct: float
    stop_loss_pct: float
    min_hold_bars: int
    asset: str = "BTCUSDT"
    primary_tf: str = "4h"
    name: str = ""
    description: str = ""
    # Phase 2 — optional DSL scoring expression. When non-empty, the
    # weights matrix is ignored at backtest time and the score is
    # computed via perpsignal.dsl.evaluate. Kept on SignalDef so the
    # canonical hash + parse path are one place.
    score: str = ""
    # Phase 2C — optional boolean DSL expressions that override the
    # threshold-based discretization. When `long_when` is set, the bar
    # goes long whenever the expression evaluates True. Same for
    # `short_when`. Either side can stay empty to keep using the
    # corresponding threshold rule from BacktestConfig.
    long_when: str = ""
    short_when: str = ""
    # Phase 3 — risk + circuit-breaker policy. These were previously
    # hard-coded at backtest time (leverage=3.0 in compute_metrics) or
    # absent (no news-skip, no per-day trade cap, no consecutive-loss
    # circuit breaker). Threading them onto SignalDef lets a prompt like
    # "BTC scalp at 40x, max 4 trades/24h, stop after 2 losses, skip
    # 00:00-04:00 UTC, skip ±15min around CPI/FOMC/NFP timestamps"
    # round-trip through parse → canonical → backtest → leaderboard.
    #
    # Defaults are chosen so a legacy signal omitting them backtest-
    # identically to before: leverage=3.0 matches the old hard-coded
    # value; empty lists / 0 caps disable the new filters entirely.
    leverage: float = 3.0
    skip_hours_utc: tuple[int, ...] = field(default_factory=tuple)
    news_timestamps_ms: tuple[int, ...] = field(default_factory=tuple)
    news_skip_minutes: int = 15
    max_trades_per_24h: int = 0     # 0 = no cap
    max_consecutive_losses: int = 0 # 0 = circuit breaker disabled
    # Diagnostic-only — not part of canonical() and not stored on disk.
    # Carries any DSL auto-repair notes parse_signal applied so the API
    # response can surface them ("Auto-fixed 2 missing parens"). The UI
    # uses these to show a small chip under the DSL textareas; the user
    # can then verify the repaired expression matches their intent.
    auto_repair_notes: tuple[str, ...] = field(default_factory=tuple)

    def canonical(self) -> dict[str, Any]:
        """Stable representation used to compute signal_id. Weights are
        rounded + sorted so cosmetically-different inputs that mean the
        same thing share an id; meta fields (name, description) are
        excluded so renames don't create a new signal. When `score` is
        set the weights matrix is irrelevant to behavior but is still
        canonicalised — keeps the hash stable across schema migrations."""
        w: dict[str, dict[str, float]] = {}
        for regime in sorted(self.weights):
            w[regime] = {
                f: round(float(self.weights[regime].get(f, 0.0)), 4)
                for f in FACTOR_ORDER
            }
        out: dict[str, Any] = {
            "weights": w,
            "long_threshold": round(float(self.long_threshold), 4),
            "short_threshold": round(float(self.short_threshold), 4),
            "take_profit_pct": round(float(self.take_profit_pct), 4),
            "stop_loss_pct": round(float(self.stop_loss_pct), 4),
            "min_hold_bars": int(self.min_hold_bars),
            "asset": self.asset,
            "primary_tf": self.primary_tf,
        }
        # Include score only when non-empty — preserves legacy ids for
        # weights-only signals (they hash identically to pre-Phase-2 runs).
        if self.score:
            # Normalize whitespace so cosmetic edits don't collide on the hash.
            out["score"] = " ".join(self.score.split())
        if self.long_when:
            out["long_when"] = " ".join(self.long_when.split())
        if self.short_when:
            out["short_when"] = " ".join(self.short_when.split())
        # Phase 3 policy fields — emit only when non-default so existing
        # signals authored before this surface keep their original signal_id.
        # The legacy leverage was hard-coded to 3.0 in compute_metrics, so
        # that value is treated as the canonical default here.
        if abs(self.leverage - 3.0) > 1e-6:
            out["leverage"] = round(float(self.leverage), 4)
        if self.skip_hours_utc:
            out["skip_hours_utc"] = sorted({int(h) for h in self.skip_hours_utc})
        if self.news_timestamps_ms:
            out["news_timestamps_ms"] = sorted({int(t) for t in self.news_timestamps_ms})
        if self.news_timestamps_ms and self.news_skip_minutes != 15:
            out["news_skip_minutes"] = int(self.news_skip_minutes)
        if self.max_trades_per_24h:
            out["max_trades_per_24h"] = int(self.max_trades_per_24h)
        if self.max_consecutive_losses:
            out["max_consecutive_losses"] = int(self.max_consecutive_losses)
        return out

    def signal_id(self) -> str:
        """Content hash of the canonical form — 12 hex chars is enough
        entropy at the volumes we expect (~10⁹ before collision risk
        becomes meaningful) and fits in a URL slug.

        Note: extending FACTOR_ORDER (e.g. adding `rsi_slow`) shifts
        this hash for definitions that would otherwise be identical,
        because canonical() includes every factor key with its rounded
        value. Existing records remain reachable by their stored id;
        only future republishes will dedupe under the new hash space.
        """
        payload = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return "sig_" + hashlib.sha256(payload.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Validation


def _err(msg: str) -> ValueError:
    return ValueError(msg)


# Names are display strings, so the only hard constraints are length
# and "no control characters" (React escapes everything else on render).
# The previous strict whitelist `[A-Za-z0-9 _-.',()]` kept rejecting
# perfectly reasonable AI-generated names that include "+", "&", ":",
# em-dash, etc. Anything ord < 0x20 or 0x7F is rejected via the
# explicit check in parse_signal below.
_NAME_CONTROL_CHAR = re.compile(r"[\x00-\x1f\x7f]")


def parse_signal(body: dict[str, Any]) -> SignalDef:
    """Validate + sanitize a JSON request body into a SignalDef. Raises
    ValueError with a user-presentable message on any rejection. Keep
    every limit close to the corresponding perpsignal.config bound so
    a signal that survives parse_signal is guaranteed to backtest."""
    if not isinstance(body, dict):
        raise _err("request body must be a JSON object")

    # Phase 2 — optional DSL scoring expression. When provided, it
    # replaces the weights matrix at backtest time; we still accept a
    # weights matrix in the same payload (defaults to all-zero) so the
    # canonical hash is stable. Validate the expression upfront so a
    # bad one returns a clear 400 rather than crashing inside the
    # backtest worker — perpsignal.dsl.parse() raises LexError or
    # ParseError on any syntactic violation.
    #
    # Phase 3.1 — also run dsl.parse_with_repair which attempts common
    # syntactic fixes (paren imbalance, trailing operators / boolean
    # keywords) before giving up. The repaired form replaces the user's
    # submitted expression so the canonical signal_id is computed from
    # the version that actually parses at backtest time. Repair notes
    # are collected on the SignalDef for the UI to surface ("auto-fixed
    # 2 missing parens").
    auto_repair_notes: list[str] = []
    raw_score = body.get("score")
    score = ""
    if raw_score is not None and str(raw_score).strip():
        score = str(raw_score).strip()
        if len(score) > 4000:
            raise _err("score expression must be ≤ 4000 chars")
        try:
            from perpsignal import dsl as _dsl
            _, score, fixes = _dsl.parse_with_repair(score)
            for f in fixes:
                auto_repair_notes.append(f"score: {f}")
        except Exception as e:  # noqa: BLE001 — surface lex/parse errors verbatim
            raise _err(f"score expression: {e}")

    # Phase 2C — entry-condition expressions. Same validation as score:
    # parse upfront via dsl.parse so a syntactic error returns 400, not
    # 500 from inside the backtest worker.
    long_when = ""
    short_when = ""
    raw_lw = body.get("long_when")
    if raw_lw is not None and str(raw_lw).strip():
        long_when = str(raw_lw).strip()
        if len(long_when) > 4000:
            raise _err("long_when expression must be ≤ 4000 chars")
        try:
            from perpsignal import dsl as _dsl
            _, long_when, fixes = _dsl.parse_with_repair(long_when)
            for f in fixes:
                auto_repair_notes.append(f"long_when: {f}")
        except Exception as e:  # noqa: BLE001
            raise _err(f"long_when expression: {e}")
    raw_sw = body.get("short_when")
    if raw_sw is not None and str(raw_sw).strip():
        short_when = str(raw_sw).strip()
        if len(short_when) > 4000:
            raise _err("short_when expression must be ≤ 4000 chars")
        try:
            from perpsignal import dsl as _dsl
            _, short_when, fixes = _dsl.parse_with_repair(short_when)
            for f in fixes:
                auto_repair_notes.append(f"short_when: {f}")
        except Exception as e:  # noqa: BLE001
            raise _err(f"short_when expression: {e}")

    # Weights — required only when no DSL surface is given. With a DSL
    # score or DSL entry conditions (long_when / short_when), an empty
    # weights object is accepted (we fill it with zeros) so the canonical
    # form stays consistent across both code paths. A pure entry-condition
    # signal needs neither a score nor weights — the per-bar position is
    # decided entirely by long_when / short_when.
    # See docs/SIGNAL_SPEC.md for the full contract.
    has_dsl = bool(score or long_when or short_when)
    raw_w = body.get("weights")
    if not has_dsl:
        if not isinstance(raw_w, dict) or not raw_w:
            raise _err(
                "weights must be a non-empty object (or provide a `score` / "
                "`long_when` / `short_when` DSL expression)"
            )
    elif raw_w is None:
        raw_w = {}
    elif not isinstance(raw_w, dict):
        raise _err("weights, when provided, must be an object")
    # Strict-mode catches typos before they silently default to 0 and
    # bake an unintended signal_id. parse_signal previously read each
    # row with .get(factor, 0) which dropped unknown keys without
    # warning — a published "rai" weight would never apply, the
    # signal would still hash + backtest, and the author would only
    # notice their weights were lost when the live score didn't move.
    if isinstance(raw_w, dict):
        unknown_regimes = sorted(set(raw_w) - set(VALID_REGIMES))
        if unknown_regimes:
            raise _err(
                f"weights contains unknown regime(s) {unknown_regimes} — "
                f"must be one of {list(VALID_REGIMES)}"
            )
    weights: dict[str, dict[str, float]] = {}
    for regime in VALID_REGIMES:
        regime_w = raw_w.get(regime) if isinstance(raw_w, dict) else None
        if regime_w is None and has_dsl:
            # DSL path — synthesize an all-zero row so canonical() stays consistent.
            weights[regime] = {f: 0.0 for f in FACTOR_ORDER}
            continue
        if not isinstance(regime_w, dict):
            raise _err(f"weights.{regime} is required and must be an object")
        unknown_factors = sorted(set(regime_w) - set(FACTOR_ORDER))
        if unknown_factors:
            raise _err(
                f"weights.{regime} contains unknown factor(s) {unknown_factors} — "
                f"see docs/SIGNAL_SPEC.md for the canonical factor list"
            )
        out_row: dict[str, float] = {}
        for factor in FACTOR_ORDER:
            try:
                v = float(regime_w.get(factor, 0.0))
            except (TypeError, ValueError):
                raise _err(f"weights.{regime}.{factor} must be a number")
            if abs(v) > MAX_ABS_WEIGHT:
                raise _err(f"weights.{regime}.{factor} must be within ±{MAX_ABS_WEIGHT}")
            out_row[factor] = v
        weights[regime] = out_row

    # All-zero weights with no DSL surface would deterministically yield
    # score=0 on every bar — 0 trades, no signal. Reject at parse time
    # so the user sees a clear error instead of waiting ~10s for an
    # empty backtest and then trying to figure out why their signal
    # never fires. DSL signals skip this check (the score / long_when /
    # short_when expression is what does the work).
    if not has_dsl:
        if all(v == 0.0 for row in weights.values() for v in row.values()):
            raise _err(
                "weights matrix is all zeros — the signal would never fire. "
                "Set at least one factor weight to non-zero, or use a `score` / "
                "`long_when` / `short_when` DSL expression."
            )

    # Thresholds
    try:
        lt = float(body.get("long_threshold", 0.20))
        st = float(body.get("short_threshold", -0.20))
    except (TypeError, ValueError):
        raise _err("long_threshold / short_threshold must be numbers")
    if not (0 < lt <= MAX_THRESHOLD):
        raise _err(f"long_threshold must be > 0 and ≤ {MAX_THRESHOLD}")
    if not (-MAX_THRESHOLD <= st < 0):
        raise _err(f"short_threshold must be < 0 and ≥ -{MAX_THRESHOLD}")

    # Risk
    try:
        tp = float(body.get("take_profit_pct", 0.04))
        sl = float(body.get("stop_loss_pct", 0.015))
        mh = int(body.get("min_hold_bars", 0))
    except (TypeError, ValueError):
        raise _err("risk fields must be numbers")
    if not (0 < tp <= MAX_TP_PCT):
        raise _err(f"take_profit_pct must be > 0 and ≤ {MAX_TP_PCT}")
    if not (0 < sl <= MAX_SL_PCT):
        raise _err(f"stop_loss_pct must be > 0 and ≤ {MAX_SL_PCT}")
    if not (0 <= mh <= MAX_MIN_HOLD):
        raise _err(f"min_hold_bars must be 0..{MAX_MIN_HOLD}")

    # Phase 3 — leverage + filters + circuit breakers. Each is optional and
    # falls back to a default that matches pre-Phase-3 backtest behavior, so
    # a body missing these fields backtests identically to before.
    try:
        lev = float(body.get("leverage", 3.0))
    except (TypeError, ValueError):
        raise _err("leverage must be a number")
    if not (1.0 <= lev <= MAX_LEVERAGE):
        raise _err(f"leverage must be between 1 and {int(MAX_LEVERAGE)}")

    raw_hours = body.get("skip_hours_utc") or []
    if not isinstance(raw_hours, (list, tuple)):
        raise _err("skip_hours_utc must be a list of integers in [0, 23]")
    skip_hours_set: set[int] = set()
    for h in raw_hours:
        try:
            hi = int(h)
        except (TypeError, ValueError):
            raise _err(f"skip_hours_utc entry {h!r} is not an integer")
        if not (0 <= hi <= 23):
            raise _err(f"skip_hours_utc entry {hi} out of range [0, 23]")
        skip_hours_set.add(hi)
    if len(skip_hours_set) > MAX_SKIP_HOURS:
        raise _err(f"skip_hours_utc must contain at most {MAX_SKIP_HOURS} entries")
    skip_hours_tuple = tuple(sorted(skip_hours_set))

    raw_news = body.get("news_timestamps_ms") or []
    if not isinstance(raw_news, (list, tuple)):
        raise _err("news_timestamps_ms must be a list of unix-ms timestamps")
    if len(raw_news) > MAX_NEWS_TIMESTAMPS:
        raise _err(f"news_timestamps_ms must contain at most {MAX_NEWS_TIMESTAMPS} entries")
    news_ts_set: set[int] = set()
    for t in raw_news:
        try:
            ti = int(t)
        except (TypeError, ValueError):
            raise _err(f"news_timestamps_ms entry {t!r} is not an integer")
        # Reject obvious mis-specified units (seconds instead of ms or vice
        # versa). A 10-digit value is unix seconds; we want 13-digit ms.
        if ti < 10_000_000_000:
            raise _err(
                f"news_timestamps_ms entry {ti} looks like seconds — expected "
                "milliseconds since epoch"
            )
        news_ts_set.add(ti)
    news_ts_tuple = tuple(sorted(news_ts_set))

    try:
        news_minutes = int(body.get("news_skip_minutes", 15))
    except (TypeError, ValueError):
        raise _err("news_skip_minutes must be an integer")
    if not (1 <= news_minutes <= NEWS_SKIP_MAX_MINUTES):
        raise _err(
            f"news_skip_minutes must be between 1 and {NEWS_SKIP_MAX_MINUTES}"
        )

    try:
        max_trades = int(body.get("max_trades_per_24h", 0))
    except (TypeError, ValueError):
        raise _err("max_trades_per_24h must be an integer")
    if not (0 <= max_trades <= MAX_TRADES_CAP):
        raise _err(f"max_trades_per_24h must be between 0 and {MAX_TRADES_CAP}")

    try:
        max_losses = int(body.get("max_consecutive_losses", 0))
    except (TypeError, ValueError):
        raise _err("max_consecutive_losses must be an integer")
    if not (0 <= max_losses <= MAX_CONSEC_LOSSES):
        raise _err(f"max_consecutive_losses must be between 0 and {MAX_CONSEC_LOSSES}")

    # Meta
    asset = _normalize_asset(body.get("asset", "BTCUSDT"))
    # Accept any Hyperliquid-listed perp (not just the core 4) so signals can
    # span the whole HL universe. Falls back to the static list if the meta
    # fetch is unavailable.
    if asset not in VALID_ASSETS:
        try:
            listed = _MARKET_META.is_listed(asset) if _MARKET_META else False
        except Exception:  # noqa: BLE001
            listed = False
        if not listed:
            raise _err("asset must be a listed Hyperliquid perp")
    primary_tf = str(body.get("primary_tf", "4h")).lower()
    if primary_tf not in VALID_TFS:
        raise _err(f"primary_tf must be one of {sorted(VALID_TFS)}")
    name = str(body.get("name", "")).strip()
    if name and len(name) > NAME_MAX:
        raise _err(f"name must be ≤{NAME_MAX} chars")
    if name and _NAME_CONTROL_CHAR.search(name):
        raise _err("name must not contain control characters (newlines, tabs, etc.)")
    description = str(body.get("description", "")).strip()
    if len(description) > DESC_MAX:
        raise _err(f"description must be ≤{DESC_MAX} chars")

    return SignalDef(
        weights=weights,
        long_threshold=lt,
        short_threshold=st,
        take_profit_pct=tp,
        stop_loss_pct=sl,
        min_hold_bars=mh,
        asset=asset,
        primary_tf=primary_tf,
        name=name,
        description=description,
        score=score,
        long_when=long_when,
        short_when=short_when,
        leverage=lev,
        skip_hours_utc=skip_hours_tuple,
        news_timestamps_ms=news_ts_tuple,
        news_skip_minutes=news_minutes,
        max_trades_per_24h=max_trades,
        max_consecutive_losses=max_losses,
        auto_repair_notes=tuple(auto_repair_notes),
    )
