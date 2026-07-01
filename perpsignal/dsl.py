"""Indicator-expression DSL for signal scoring.

Phase 2A: a tiny, safe expression language that lets a signal's score
be authored as algebra over OHLCV-rooted indicators, rather than as a
linear combination of pre-baked factors with regime gating. This is the
structural fix for prompts like:

    "0.35 × slope(close, 20) × (ADX(14) / 50) + 0.25 × ΔRSI(14)/50
   + 0.20 × volume_zscore(12) + 0.20 × (1 − |ΔATR|/ATR)"

— which the old weights-matrix engine could only approximate as a sum
of pre-baked factor outputs (no genuine `a × b` product, no division by
ADX, no per-bar conditional shape).

NOT a general programming language. Hard scope by design:

  Allowed
  -------
  Numbers           42, 1.5, -0.2
  Series accessors  close, high, low, open, volume, funding, oi,
                    hour_utc, minute_utc, dayofweek_utc (time-of-day
                    filters, sourced from the bar's UTC timestamp)
  Indicator calls   rsi(close, 14), ema(close, 20), slope(close, 20),
                    adx(14), atr(14), vwap(96), session_vwap(),
                    zscore(volume, 48), bb_width(close, 20, 2),
                    bb_lower / bb_upper / bb_mid (same args as bb_width),
                    sma, stdev, corr, prev
  Math helpers      abs, min, max, sign, tanh, log, sqrt, clip
  Operators         + - * /, parentheses, unary minus

  Forbidden — by lexer or parser, not at eval time
  ------------------------------------------------
  - Attribute access, indexing, comprehensions, lambdas, imports
  - Function names not in the BUILTINS whitelist
  - Identifiers that aren't a recognised series accessor or builtin
  - Strings, lists, dicts, the words `exec`/`eval`/`__`/`import`

  Phase 2C also adds comparison (< <= > >= == !=), boolean
  (and / or / not), and the `if(cond, a, b)` ternary as a builtin.
  These return boolean Series that compose with the arithmetic surface
  via the implicit bool→float coercion pandas does element-wise, so
  `0.5 * (close > ema(close, 20))` is a legal momentum-tilt expression.

Output of `evaluate(expr, frame)` is a pandas.Series of floats indexed
by the bars in `frame`, ready to discretize via long/short thresholds.

Security note
-------------
There is no `eval()` / `exec()` / `compile()` of user input anywhere
in this module. The lexer rejects token shapes it doesn't recognise;
the parser only constructs AST nodes from a fixed grammar; the
evaluator only calls functions present in a hard-coded dict. A
published signal CANNOT execute arbitrary Python via this surface.

(A user CAN write `1/0` and get a ZeroDivisionError → caught at the
api/signals boundary and surfaced as a "bad expression" 400.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import factors as F

# ----------------------------------------------------------------------
# Token types
# ----------------------------------------------------------------------
NUMBER, IDENT, OP, LP, RP, COMMA, EOF = "NUMBER", "IDENT", "OP", "LP", "RP", "COMMA", "EOF"


def _humanize_token(kind: str, value: Any = None) -> str:
    """Human-readable label for a parser-internal token kind. The raw
    kinds (LP, RP, COMMA, IDENT, NUM, EOF) are unintelligible to users
    debugging an expression — translate them so error messages read
    like prose instead of compiler internals."""
    if kind == EOF:
        return "end of expression"
    if kind == LP:
        return "`(`"
    if kind == RP:
        return "closing `)`"
    if kind == COMMA:
        return "`,`"
    if kind == NUMBER:
        return f"number {value!r}" if value is not None else "a number"
    if kind == IDENT:
        return f"identifier {value!r}" if value is not None else "an identifier"
    if kind == OP:
        return f"operator {value!r}" if value is not None else "an operator"
    return f"{kind} {value!r}" if value is not None else kind


@dataclass(frozen=True)
class Token:
    kind: str
    value: Any
    pos: int          # column in source — used in error messages


# ----------------------------------------------------------------------
# Lexer
# ----------------------------------------------------------------------
class LexError(ValueError):
    pass


_IDENT_CHARS = set("abcdefghijklmnopqrstuvwxyz_0123456789")


def tokenize(src: str) -> list[Token]:
    """Linear scan. Numbers, identifiers, operators, parens, commas.

    Lower-cases identifiers so authors can write `RSI(close, 14)` or
    `rsi(...)` interchangeably. Doesn't handle Unicode minus/× signs
    — the upstream form normalizes those before submission.
    """
    if len(src) > 4000:
        raise LexError("Expression too long (max 4000 chars).")
    out: list[Token] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
            continue
        if c == "(":
            out.append(Token(LP, "(", i)); i += 1; continue
        if c == ")":
            out.append(Token(RP, ")", i)); i += 1; continue
        if c == ",":
            out.append(Token(COMMA, ",", i)); i += 1; continue
        if c in "+-*/":
            out.append(Token(OP, c, i)); i += 1; continue
        # Two-char comparison operators must be tried first so we don't
        # eat the first char as a one-char op (`<=` vs `<` followed by `=`).
        if c in "<>=!" and i + 1 < n and src[i + 1] == "=":
            out.append(Token(OP, c + "=", i)); i += 2; continue
        if c in "<>":
            out.append(Token(OP, c, i)); i += 1; continue
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            j = i
            while j < n and (src[j].isdigit() or src[j] == "."):
                j += 1
            try:
                num = float(src[i:j])
            except ValueError:
                raise LexError(f"Bad number at column {i}: {src[i:j]!r}") from None
            out.append(Token(NUMBER, num, i))
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (src[j].lower() in _IDENT_CHARS):
                j += 1
            ident = src[i:j].lower()
            # Disallow Python's dunder + control-flow names from ever entering
            # the token stream. Belt-and-braces — the evaluator also gates by
            # the BUILTINS whitelist, but rejecting these at lex time means
            # a misleading error message can't even refer to them.
            if "__" in ident or ident in {"exec", "eval", "import", "lambda", "return", "yield", "compile"}:
                raise LexError(f"Reserved identifier {ident!r} not allowed.")
            out.append(Token(IDENT, ident, i))
            i = j
            continue
        raise LexError(f"Unexpected character {c!r} at column {i}.")
    out.append(Token(EOF, None, n))
    return out


# ----------------------------------------------------------------------
# AST
# ----------------------------------------------------------------------
@dataclass
class Num:
    value: float


@dataclass
class Var:
    name: str         # series accessor: close, high, low, volume, funding, ...


@dataclass
class Call:
    fn: str           # function name (validated against BUILTINS at eval time)
    args: list = field(default_factory=list)


@dataclass
class BinOp:
    op: str           # + - * /
    lhs: Any
    rhs: Any


@dataclass
class UnaryNeg:
    operand: Any


@dataclass
class Cmp:
    """Comparison: `lhs OP rhs` where OP is one of < <= > >= == !=. Returns
    a Series of bool when either operand is a Series, else a Python bool."""
    op: str
    lhs: Any
    rhs: Any


@dataclass
class BoolOp:
    """Boolean combiner — `and` / `or` / `not`. Stored as a flat n-ary
    operand list for `and`/`or` so the evaluator can fold them efficiently;
    `not` uses operands=[x] and op='not'."""
    op: str  # "and" | "or" | "not"
    operands: list


AstNode = Num | Var | Call | BinOp | UnaryNeg | Cmp | BoolOp


# ----------------------------------------------------------------------
# Parser — Pratt (top-down operator precedence)
# ----------------------------------------------------------------------
class ParseError(ValueError):
    pass


# Bigger number = tighter binding. Standard math precedence. Comparison
# and boolean ops are handled in dedicated parse methods (parse_or →
# parse_and → parse_not → parse_cmp → parse_expr) since their right-hand
# operand types differ from arithmetic (bool vs number).
_PRECEDENCE = {"+": 10, "-": 10, "*": 20, "/": 20}
_CMP_OPS = {"<", "<=", ">", ">=", "==", "!="}
_BOOL_KEYWORDS = {"and", "or", "not"}


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.i = 0

    def peek(self) -> Token:
        return self.tokens[self.i]

    def advance(self) -> Token:
        t = self.tokens[self.i]
        self.i += 1
        return t

    def expect(self, kind: str, value: Any = None) -> Token:
        t = self.advance()
        if t.kind != kind or (value is not None and t.value != value):
            want = _humanize_token(kind, value)
            got = _humanize_token(t.kind, t.value if t.kind != EOF else None)
            raise ParseError(
                f"Expected {want} at column {t.pos}, got {got}. "
                "Check that every `(` has a matching `)`, and every function "
                "call has the right number of comma-separated arguments."
            )
        return t

    def parse(self) -> AstNode:
        # Entry point — full precedence climb starts at boolean `or` (lowest).
        node = self.parse_or()
        if self.peek().kind != EOF:
            t = self.peek()
            raise ParseError(
                f"Unexpected {_humanize_token(t.kind, t.value)} at column {t.pos}. "
                "Likely a stray token or a missing operator before it."
            )
        return node

    def _is_keyword(self, t: Token, kw: str) -> bool:
        return t.kind == IDENT and t.value == kw

    def parse_or(self) -> AstNode:
        # Boolean OR — lowest precedence, left-associative, flattened into
        # an n-ary BoolOp so `a or b or c` evaluates in one pass.
        lhs = self.parse_and()
        if not self._is_keyword(self.peek(), "or"):
            return lhs
        operands = [lhs]
        while self._is_keyword(self.peek(), "or"):
            self.advance()
            operands.append(self.parse_and())
        return BoolOp("or", operands)

    def parse_and(self) -> AstNode:
        lhs = self.parse_not()
        if not self._is_keyword(self.peek(), "and"):
            return lhs
        operands = [lhs]
        while self._is_keyword(self.peek(), "and"):
            self.advance()
            operands.append(self.parse_not())
        return BoolOp("and", operands)

    def parse_not(self) -> AstNode:
        if self._is_keyword(self.peek(), "not"):
            self.advance()
            return BoolOp("not", [self.parse_not()])
        return self.parse_cmp()

    def parse_cmp(self) -> AstNode:
        # Comparison — a single non-chained comparison (Python-style
        # `a < b < c` is rejected to keep semantics unambiguous). Returns
        # the additive expression as-is when no comparison follows.
        lhs = self.parse_expr(0)
        t = self.peek()
        if t.kind == OP and t.value in _CMP_OPS:
            op = self.advance().value
            rhs = self.parse_expr(0)
            # Reject chained comparisons.
            t2 = self.peek()
            if t2.kind == OP and t2.value in _CMP_OPS:
                raise ParseError(
                    f"Chained comparisons (`a {op} b {t2.value} c`) not allowed — "
                    f"use `(a {op} b) and (b {t2.value} c)` explicitly at column {t2.pos}."
                )
            return Cmp(op, lhs, rhs)
        return lhs

    def parse_expr(self, min_prec: int) -> AstNode:
        # Pratt loop — parse a leading primary / unary, then keep folding
        # right while the next operator binds tighter than min_prec.
        lhs = self.parse_unary()
        while True:
            t = self.peek()
            if t.kind != OP or t.value not in _PRECEDENCE:
                break
            prec = _PRECEDENCE[t.value]
            if prec < min_prec:
                break
            op = self.advance().value
            # +/-/*//// are all left-associative, so we recurse with prec+1
            # to ensure the same-precedence operator on the right binds
            # the immediate right operand only.
            rhs = self.parse_expr(prec + 1)
            lhs = BinOp(op, lhs, rhs)
        return lhs

    def parse_unary(self) -> AstNode:
        t = self.peek()
        if t.kind == OP and t.value == "-":
            self.advance()
            return UnaryNeg(self.parse_unary())
        if t.kind == OP and t.value == "+":
            self.advance()
            return self.parse_unary()
        return self.parse_primary()

    def parse_primary(self) -> AstNode:
        t = self.advance()
        if t.kind == NUMBER:
            return Num(float(t.value))
        if t.kind == LP:
            # Parens reset to the full precedence climb so users can write
            # `not (a < b)` and `if(a > b, x, y)` — args/parens are the
            # only places comparison and boolean expressions are legal as
            # sub-expressions of arithmetic context.
            inner = self.parse_or()
            self.expect(RP)
            return inner
        if t.kind == IDENT:
            # Reject bare keyword as primary — they're operators, not values.
            if t.value in _BOOL_KEYWORDS:
                raise ParseError(f"Unexpected keyword {t.value!r} at column {t.pos}.")
            # identifier followed by ( → function call. Bare identifier → series accessor.
            if self.peek().kind == LP:
                self.advance()
                args: list[AstNode] = []
                if self.peek().kind != RP:
                    # Same logic as parens: args parse at full precedence
                    # so `if(adx(14) > 25, slope, 0)` works.
                    args.append(self.parse_or())
                    while self.peek().kind == COMMA:
                        self.advance()
                        args.append(self.parse_or())
                self.expect(RP)
                if len(args) > 6:
                    raise ParseError(f"Too many arguments to {t.value!r} (max 6).")
                return Call(t.value, args)
            return Var(t.value)
        raise ParseError(f"Unexpected {t.kind} {t.value!r} at column {t.pos}.")


def parse(src: str) -> AstNode:
    return Parser(tokenize(src)).parse()


# ----------------------------------------------------------------------
# Auto-repair — heal the canonical AI-generated malformations
# ----------------------------------------------------------------------
# The LLM signal designer occasionally produces expressions that don't
# parse: unbalanced parens (most common — the model loses count in a
# 1000-char expression with nested function calls), trailing operators
# from a dropped clause, or a dangling boolean keyword. Rather than
# 400-ing the publish flow, we try a few cheap syntactic fixes before
# surfacing the parse error. The repaired form is what gets stored on
# the SignalDef so the canonical signal_id is stable.
#
# Repair is best-effort, not exhaustive: it handles the specific failure
# modes we've actually seen, and surfaces a human-readable note for
# every fix so the user can decide whether the auto-corrected meaning
# matches their intent.

_TRAILING_OPS = set("+-*/,")
_TRAILING_KEYWORDS = ("and", "or", "not")


import re as _re

# Pine-Script-style lookback: `close[20]` means "close 20 bars ago".
# The DSL has no `[`/`]` tokens (lexer rejects them), so we translate
# to the equivalent `prev(close, 20)` before the parse retry. Match is
# deliberately tight — bare identifier or factor accessor followed by
# `[<integer>]` — to avoid mangling unrelated brackets that might have
# slipped in (e.g. a stray list literal we'd rather let the parser
# reject loudly).
_PINE_BRACKET = _re.compile(r"\b([a-z_][a-z0-9_]*)\s*\[\s*(\d+)\s*\]")
# Pine namespacing: `ta.highest`, `math.abs`, `request.security` etc.
# We strip the prefix and let the identifier resolve through the
# normal BUILTINS / _BUILTIN_ALIASES path. `ta.highest` → `highest`
# (which we now provide), `ta.atr` → `atr`. Unknown identifiers after
# stripping still fail the eval-time whitelist with a clean error.
_PINE_NAMESPACE = _re.compile(r"\b(ta|math|request|input|str|color|array|matrix)\.")


def repair_expression(src: str) -> tuple[str, list[str]]:
    """Try common syntactic fixes for malformed DSL expressions.

    Returns (repaired_src, fixes). `fixes` is empty when no repair was
    needed (or possible); each entry is a one-line note suitable for
    surfacing to the user ("appended 2 missing closing parens").

    Repair order:
      1. Pine-style namespace stripping (`ta.highest` → `highest`) and
         bracket-lookback translation (`close[20]` → `prev(close, 20)`).
         These reshape the source before the parse-tree-level fixes so
         the paren counter sees the corrected form.
      2. Trim trailing junk (operators, commas, dangling and/or/not).
      3. Balance parens — append missing closers or drop stray ones.

    Steps 2+3 loop until the expression stabilises so multiple issues
    chain cleanly (a trailing `or` plus an unmatched `(` plus a
    trailing `+` all heal in one pass). Iteration is capped defensively
    against pathological inputs.
    """
    fixes: list[str] = []
    s = src.strip()
    if not s:
        return s, fixes

    # 1a. Namespace prefix strip (run once — purely textual).
    if _PINE_NAMESPACE.search(s):
        stripped = _PINE_NAMESPACE.sub("", s)
        if stripped != s:
            fixes.append("stripped Pine namespace prefix (e.g. `ta.`, `math.`)")
            s = stripped

    # 1b. Bracket lookback → prev(). Run iteratively in case a single
    # expression has several: `close[1] > close[2]` rewrites in one pass
    # because the regex is global. Tally the count for the user note.
    bracket_count = 0
    def _bracket_sub(m: "_re.Match[str]") -> str:
        nonlocal bracket_count
        bracket_count += 1
        return f"prev({m.group(1)}, {m.group(2)})"
    s_after_brackets = _PINE_BRACKET.sub(_bracket_sub, s)
    if bracket_count:
        fixes.append(
            f"translated {bracket_count} Pine-style `name[n]` lookback"
            + ("s" if bracket_count > 1 else "")
            + " to `prev(name, n)`"
        )
        s = s_after_brackets

    for _ in range(8):                       # cap iterations defensively
        before = s

        # 1. Trailing operator (`a + b *` → `a + b`).
        trimmed = 0
        while s and s[-1] in _TRAILING_OPS:
            s = s[:-1].rstrip()
            trimmed += 1
        if trimmed:
            fixes.append(
                f"trimmed {trimmed} trailing operator/comma"
                + ("s" if trimmed > 1 else "")
            )

        # 2. Trailing boolean keyword (`adx(14) > 25 and` → `adx(14) > 25`).
        # Word-boundary check stops us trimming the tail of an identifier
        # like `ema_cross_and` (not a real name but cheap to be defensive).
        for kw in _TRAILING_KEYWORDS:
            if s.lower().endswith(kw):
                if len(s) == len(kw) or not (s[-len(kw) - 1].isalnum() or s[-len(kw) - 1] == "_"):
                    s = s[:-len(kw)].rstrip()
                    fixes.append(f"trimmed trailing `{kw}`")
                    break

        # 3. Paren balance.
        open_n = s.count("(")
        close_n = s.count(")")
        if open_n > close_n:
            missing = open_n - close_n
            s = s + ")" * missing
            fixes.append(
                f"appended {missing} missing closing paren"
                + ("s" if missing > 1 else "")
            )
        elif close_n > open_n:
            excess = close_n - open_n
            # Walk from the right and drop the unmatched close parens.
            # We prefer trimming the right side because the AI almost
            # always overshoots at the end, not the start.
            chars = list(s)
            removed = 0
            i = len(chars) - 1
            while i >= 0 and removed < excess:
                if chars[i] == ")":
                    chars.pop(i)
                    removed += 1
                i -= 1
            s = "".join(chars).rstrip()
            fixes.append(
                f"removed {removed} stray closing paren"
                + ("s" if removed > 1 else "")
            )

        if s == before:
            break

    return s, fixes


def parse_with_repair(src: str) -> tuple[AstNode, str, list[str]]:
    """Strict-then-repair parse.

    1. Try `parse(src)` as-is. If it succeeds, return the AST with no
       fixes (the caller's expression was already valid).
    2. If it raises, try `repair_expression(src)`; if any fix was
       applied AND the repaired form parses, return the AST + the
       repaired source + the list of fixes.
    3. If neither path parses, raise the ORIGINAL exception — the user
       sees the actual problem with their input, not a misleading
       error from a half-repaired version.

    The repaired source is what the caller should store / hash /
    re-serialize, so the canonical signal_id matches the expression
    that actually parses at every future backtest.
    """
    try:
        return parse(src), src, []
    except (LexError, ParseError) as original_err:
        repaired, fixes = repair_expression(src)
        if not fixes or repaired == src.strip():
            raise
        try:
            return parse(repaired), repaired, fixes
        except (LexError, ParseError):
            raise original_err


# ----------------------------------------------------------------------
# Evaluator + builtins
# ----------------------------------------------------------------------
class EvalError(ValueError):
    pass


# Series accessors. Each maps a bare identifier to a column on the input
# frame. The evaluator falls back to a zero-series when a column is
# missing (e.g. `funding` on a venue without funding data) — that keeps
# expressions like `score + 0.1 * funding` from blowing up when the
# component just isn't available.
# Factor-column accessors. Each is one of the 17 Perps DNA factors
# computed by build_mtf_features and merged into the frame by
# agentkit.signal_backtest before dsl.evaluate is called. Listed here
# so _resolve_series doesn't reject them as typos. Names that already
# exist in _SERIES_ACCESSORS or BUILTINS (rsi, volume, oi, funding,
# bb_width) intentionally resolve to the raw / function form; authors
# wanting the factor-normalised version can use the explicit indicator
# (e.g. zscore(volume, 48)). Keep in sync with perpsignal.config.FACTOR_ORDER.
_FACTOR_ACCESSORS = (
    "ema_cross", "macd", "trend", "rsi_slow", "ema_cross_slow", "macd_slow",
    "slope_regression", "adx_strength", "rsi_delta",
    "volume_zscore", "atr_stability", "vwap_distance",
)
# Time-of-day accessors. `hour_utc` / `minute_utc` / `dayofweek_utc` resolve
# to a per-bar integer series derived from the frame's DatetimeIndex (UTC),
# so authors can write `hour_utc >= 0 and hour_utc < 4` to mute Asia hours.
# `dayofweek_utc` follows pandas convention: Mon=0..Sun=6.
_TIME_ACCESSORS = ("hour_utc", "minute_utc", "dayofweek_utc")
_SERIES_ACCESSORS = {
    "close", "high", "low", "open", "volume", "funding", "oi",
    "open_interest", "bar_index", *_FACTOR_ACCESSORS, *_TIME_ACCESSORS,
}


def _as_series(v: Any, index: pd.Index) -> pd.Series:
    if isinstance(v, pd.Series):
        return v
    if isinstance(v, (int, float, np.floating, np.integer)):
        return pd.Series(float(v), index=index)
    raise EvalError(f"Cannot coerce {type(v).__name__} into a Series.")


def _as_int(v: Any, name: str, lo: int = 1, hi: int = 1000) -> int:
    """Indicator periods MUST be plain numeric literals — `rsi(close, n)`
    where n is itself a Series doesn't make sense. Caught here and at the
    boundary so a bad expression returns a 400 not a runtime explosion."""
    if isinstance(v, pd.Series):
        if v.nunique(dropna=True) > 1:
            raise EvalError(f"{name}: period must be a constant, got a varying series.")
        try:
            v = float(v.iloc[0]) if len(v) else 0.0
        except Exception:
            raise EvalError(f"{name}: could not coerce series to scalar.") from None
    try:
        iv = int(v)
    except Exception:
        raise EvalError(f"{name}: period must be an integer.") from None
    if iv < lo or iv > hi:
        raise EvalError(f"{name}: period {iv} out of range [{lo}, {hi}].")
    return iv


# Indicator builtins — wrap perpsignal.factors so the DSL stays in sync
# with the production primitives. Adding a new factor to factors.py +
# registering it here is the entire extension point.

def _bi_rsi(ctx, args):
    if len(args) not in (1, 2):
        raise EvalError("rsi(series, period=14)")
    series = _as_series(args[0], ctx.index)
    period = _as_int(args[1], "rsi") if len(args) == 2 else 14
    return F.rsi(series, period)


def _bi_ema(ctx, args):
    if len(args) != 2:
        raise EvalError("ema(series, period)")
    return F.ema(_as_series(args[0], ctx.index), _as_int(args[1], "ema"))


def _bi_sma(ctx, args):
    if len(args) != 2:
        raise EvalError("sma(series, period)")
    series = _as_series(args[0], ctx.index)
    return series.rolling(_as_int(args[1], "sma")).mean()


def _bi_stdev(ctx, args):
    if len(args) != 2:
        raise EvalError("stdev(series, window)")
    series = _as_series(args[0], ctx.index)
    return series.rolling(_as_int(args[1], "stdev")).std()


def _bi_zscore(ctx, args):
    if len(args) != 2:
        raise EvalError("zscore(series, window)")
    return F.rolling_zscore(_as_series(args[0], ctx.index), _as_int(args[1], "zscore"))


def _bi_slope(ctx, args):
    if len(args) not in (1, 2):
        raise EvalError("slope(series, lookback=20)")
    series = _as_series(args[0], ctx.index)
    lookback = _as_int(args[1], "slope") if len(args) == 2 else 20
    # factor_slope_regression already returns a tanh-squashed [-1,+1]
    # version. For the DSL we want the raw slope-per-bar so users can
    # compose it (multiply by ADX/50, add to other terms). Build it
    # locally without the squash.
    idx = pd.Series(np.arange(len(series), dtype=float), index=series.index)
    corr = series.rolling(lookback).corr(idx)
    sd_s = series.rolling(lookback).std()
    sd_i = idx.rolling(lookback).std().replace(0.0, np.nan)
    slope = (corr * sd_s / sd_i).fillna(0.0)
    # Normalize by price so the scale doesn't drift with absolute price.
    return slope / series.replace(0.0, np.nan)


def _bi_adx(ctx, args):
    if len(args) not in (0, 1):
        raise EvalError("adx(period=14)")
    period = _as_int(args[0], "adx") if args else 14
    return F.adx(ctx.frame["high"], ctx.frame["low"], ctx.frame["close"], period)


def _bi_atr(ctx, args):
    if len(args) not in (0, 1):
        raise EvalError("atr(period=14)")
    period = _as_int(args[0], "atr") if args else 14
    # Shared TR+Wilder-smoothing primitive so the DSL stays in sync with factors.
    return F.atr(ctx.frame["high"], ctx.frame["low"], ctx.frame["close"], period)


def _bi_vwap(ctx, args):
    if len(args) not in (0, 1):
        raise EvalError("vwap(period=96)")
    period = _as_int(args[0], "vwap") if args else 96
    c = ctx.frame["close"]
    # Same rolling-VWAP core as factors; the DSL fills gaps with last/close so a
    # bare vwap() reference is always a usable number.
    return F.vwap(ctx.frame["high"], ctx.frame["low"], c, ctx.frame["volume"], period).ffill().fillna(c)


def _bi_bb_width(ctx, args):
    if len(args) not in (1, 2, 3):
        raise EvalError("bb_width(series, period=20, num_std=2)")
    series = _as_series(args[0], ctx.index)
    period = _as_int(args[1], "bb_width") if len(args) >= 2 else 20
    nstd = float(args[2]) if (len(args) == 3 and isinstance(args[2], (int, float))) else 2.0
    return F.bb_width(series, period, nstd)


def _bb_bands(series: pd.Series, period: int, nstd: float):
    """Shared helper for bb_lower / bb_upper / bb_mid. Returns
    (mid, lower, upper) — all NaN-padded for the first `period - 1` bars
    like every other rolling primitive in this module."""
    mid = series.rolling(period).mean()
    sd = series.rolling(period).std()
    band = nstd * sd
    return mid, mid - band, mid + band


def _bb_args(args, name: str):
    """Validate and unpack (series, period, nstd) for the bb_* family.
    Accepts the same 1/2/3-arg signature as bb_width so the four
    Bollinger primitives share a calling convention."""
    if len(args) not in (1, 2, 3):
        raise EvalError(f"{name}(series, period=20, num_std=2)")
    period = _as_int(args[1], name) if len(args) >= 2 else 20
    nstd = float(args[2]) if (len(args) == 3 and isinstance(args[2], (int, float))) else 2.0
    return period, nstd


def _bi_bb_lower(ctx, args):
    series = _as_series(args[0], ctx.index)
    period, nstd = _bb_args(args, "bb_lower")
    _, lo, _ = _bb_bands(series, period, nstd)
    return lo


def _bi_bb_upper(ctx, args):
    series = _as_series(args[0], ctx.index)
    period, nstd = _bb_args(args, "bb_upper")
    _, _, up = _bb_bands(series, period, nstd)
    return up


def _bi_bb_mid(ctx, args):
    series = _as_series(args[0], ctx.index)
    period, _ = _bb_args(args, "bb_mid")
    return series.rolling(period).mean()


def _bi_session_vwap(ctx, args):
    """Session-anchored VWAP. Resets at 00:00 UTC every day, in contrast
    to vwap(period) which is a fixed-window rolling VWAP. Matches the
    "session VWAP" most TradingView/Pine scripts mean by default —
    cumulative ΣPV / ΣV from the day's open through the current bar.

    No args. Requires the frame index to be a DatetimeIndex; otherwise
    falls back to vwap() with a generous default window so a bad input
    doesn't crash the backtest.
    """
    if len(args) != 0:
        raise EvalError("session_vwap() — no arguments")
    df = ctx.frame
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        # No tz-aware index → no session concept. Fall back to a 96-bar
        # rolling VWAP so the expression still evaluates instead of 500.
        return _bi_vwap(ctx, [])
    h = df["high"]; lo = df["low"]; c = df["close"]; v = df["volume"]
    typical = (h + lo + c) / 3.0
    # Group key = UTC date. tz_convert to UTC if the index is tz-aware,
    # else assume the timestamps are already UTC (matches build_mtf_features).
    if idx.tz is not None:
        day_key = idx.tz_convert("UTC").normalize()
    else:
        day_key = idx.normalize()
    pv = (typical * v).groupby(day_key).cumsum()
    vsum = v.groupby(day_key).cumsum().replace(0.0, np.nan)
    return (pv / vsum).ffill().fillna(c)


def _bi_corr(ctx, args):
    if len(args) != 3:
        raise EvalError("corr(s1, s2, window)")
    s1 = _as_series(args[0], ctx.index); s2 = _as_series(args[1], ctx.index)
    return s1.rolling(_as_int(args[2], "corr")).corr(s2)


def _bi_highest(ctx, args):
    """Rolling maximum over `lookback` bars. Equivalent to Pine's
    `ta.highest(series, n)` — the canonical primitive for breakout
    rules ("close above the highest high of the prior 20 candles")."""
    if len(args) != 2:
        raise EvalError("highest(series, lookback)")
    series = _as_series(args[0], ctx.index)
    n = _as_int(args[1], "highest", lo=1, hi=500)
    return series.rolling(n).max()


def _bi_lowest(ctx, args):
    """Rolling minimum over `lookback` bars. Pine's `ta.lowest`."""
    if len(args) != 2:
        raise EvalError("lowest(series, lookback)")
    series = _as_series(args[0], ctx.index)
    n = _as_int(args[1], "lowest", lo=1, hi=500)
    return series.rolling(n).min()


def _bi_prev(ctx, args):
    if len(args) not in (1, 2):
        raise EvalError("prev(series, n=1) — lag operator")
    series = _as_series(args[0], ctx.index)
    n = _as_int(args[1], "prev", lo=1, hi=500) if len(args) == 2 else 1
    return series.shift(n)

def _bi_roc(ctx, args):
    """
    Rate of Change (ROC) indicator.
    Formula: (close / prev(close, n)) - 1
    Usage in DSL: roc(close, 14)
    """
    if len(args) != 2:
        raise EvalError("roc(series, period)")
    series = _as_series(args[0], ctx.index)
    period = _as_int(args[1], "roc")
    
    prev_series = series.shift(period)
    return (series / prev_series.replace(0.0, np.nan)).fillna(0.0) - 1

# Math helpers. Each operates element-wise over Series via the underlying
# numpy/pandas op, so they compose with arithmetic operators naturally.
def _bi_abs(ctx, args):  return _as_series(args[0], ctx.index).abs()
def _bi_sign(ctx, args): return np.sign(_as_series(args[0], ctx.index))
def _bi_log(ctx, args):  return np.log(_as_series(args[0], ctx.index).replace(0.0, np.nan)).fillna(0.0)
def _bi_sqrt(ctx, args): return np.sqrt(_as_series(args[0], ctx.index).clip(lower=0.0))
def _bi_tanh(ctx, args):
    if len(args) not in (1, 2):
        raise EvalError("tanh(x, scale=1)")
    x = _as_series(args[0], ctx.index)
    scale = float(args[1]) if (len(args) == 2 and isinstance(args[1], (int, float))) else 1.0
    return np.tanh(x * scale)


def _bi_min(ctx, args):
    if len(args) < 2:
        raise EvalError("min(a, b, ...)")
    return pd.concat([_as_series(a, ctx.index) for a in args], axis=1).min(axis=1)


def _bi_max(ctx, args):
    if len(args) < 2:
        raise EvalError("max(a, b, ...)")
    return pd.concat([_as_series(a, ctx.index) for a in args], axis=1).max(axis=1)


def _bi_clip(ctx, args):
    if len(args) != 3:
        raise EvalError("clip(x, lo, hi)")
    return _as_series(args[0], ctx.index).clip(lower=float(args[1]), upper=float(args[2]))


def _bi_if(ctx, args):
    """if(cond, true_value, false_value) — element-wise ternary.

    cond can be a bool Series or a scalar bool. true_value / false_value
    are coerced to Series so the result aligns with the frame index.
    """
    if len(args) != 3:
        raise EvalError("if(cond, true_value, false_value)")
    cond = args[0]
    a = _as_series(args[1], ctx.index)
    b = _as_series(args[2], ctx.index)
    if isinstance(cond, pd.Series):
        return a.where(cond.astype(bool), b)
    return a if bool(cond) else b


# Master function table — single source of truth for what a published
# signal's expression can call. NEVER add `eval`, `exec`, or anything
# that takes a callable argument.
BUILTINS: dict[str, Callable] = {
    # indicators
    "rsi": _bi_rsi, "ema": _bi_ema, "sma": _bi_sma, "stdev": _bi_stdev,
    "zscore": _bi_zscore, "slope": _bi_slope,
    "adx": _bi_adx, "atr": _bi_atr, "vwap": _bi_vwap,
    "session_vwap": _bi_session_vwap,
    "bb_width": _bi_bb_width,
    "bb_lower": _bi_bb_lower, "bb_upper": _bi_bb_upper, "bb_mid": _bi_bb_mid,
    "highest": _bi_highest, "lowest": _bi_lowest,
    "corr": _bi_corr, "prev": _bi_prev,
    "roc": _bi_roc,
    # math
    "abs": _bi_abs, "sign": _bi_sign, "log": _bi_log, "sqrt": _bi_sqrt,
    "tanh": _bi_tanh, "min": _bi_min, "max": _bi_max, "clip": _bi_clip,
    # control flow
    "if": _bi_if,
}

# Common natural-language aliases for BUILTINS. The LLM (and humans
# porting from Pine Script / pandas-ta / TA-Lib) reach for `correlation`,
# `standard_deviation`, etc. Resolving these to the canonical name keeps
# generated expressions from breaking at eval time — and the canonical
# signal_id hash is unaffected because parse() preserves the literal
# token in the AST. Add new aliases here as they surface in error logs.
_BUILTIN_ALIASES: dict[str, str] = {
    "correlation": "corr",
    "standard_deviation": "stdev",
    "std": "stdev",
    "z_score": "zscore",
    "exponential_moving_average": "ema",
    "simple_moving_average": "sma",
    "moving_average": "sma",
    "average_true_range": "atr",
    "bollinger_width": "bb_width",
    "bollinger_band_width": "bb_width",
    "bollinger_lower": "bb_lower",
    "bollinger_upper": "bb_upper",
    "bollinger_middle": "bb_mid",
    "bollinger_mid": "bb_mid",
    "bb_middle": "bb_mid",
    "bb_basis": "bb_mid",
    "bbl": "bb_lower",
    "bbu": "bb_upper",
    "bbm": "bb_mid",
    "lower_band": "bb_lower",
    "upper_band": "bb_upper",
    "anchored_vwap": "session_vwap",
    "session_volume_weighted_average_price": "session_vwap",
    "svwap": "session_vwap",
    "rolling_max": "highest",
    "rolling_min": "lowest",
    "max_over": "highest",
    "min_over": "lowest",
    "highest_high": "highest",
    "lowest_low": "lowest",
    "ta_highest": "highest",
    "ta_lowest": "lowest",
    "relative_strength_index": "rsi",
    "linear_regression_slope": "slope",
    "linreg_slope": "slope",
    "previous": "prev",
    "lag": "prev",
    "abs_value": "abs",
    "absolute": "abs",
    "logarithm": "log",
    "natural_log": "log",
    "ln": "log",
    "square_root": "sqrt",
    "maximum": "max",
    "minimum": "min",
    "where": "if",
    "rate_of_change": "roc",
}


@dataclass
class EvalContext:
    frame: pd.DataFrame
    @property
    def index(self) -> pd.Index:
        return self.frame.index


def _resolve_series(name: str, ctx: EvalContext) -> pd.Series:
    """Bare identifier → column lookup. Missing columns return a zero
    series rather than raising — keeps expressions like
    `0.3 * close + 0.1 * funding` from breaking on assets without funding
    data instead of forcing the user to special-case it."""
    if name not in _SERIES_ACCESSORS:
        raise EvalError(f"Unknown identifier {name!r}. Did you mean a function call (e.g. {name}(close, 14))?")
    # Normalize the open_interest / oi alias to whichever column exists.
    if name == "oi" or name == "open_interest":
        if "open_interest" in ctx.frame.columns:
            return ctx.frame["open_interest"].astype(float)
        return pd.Series(0.0, index=ctx.index)
    # bar_index — 0-based integer position of each bar. Useful for
    # warm-up gates ("ignore first 50 bars while EMAs settle") and
    # for time-since-event patterns. Not a column on the frame, so
    # synthesize from the index length.
    if name == "bar_index":
        return pd.Series(range(len(ctx.index)), index=ctx.index, dtype=float)
    # Time-of-day accessors. Derived from the DatetimeIndex, always in UTC
    # — backtest frames are constructed with UTC timestamps by
    # build_mtf_features, so authors writing `hour_utc >= 0 and hour_utc < 4`
    # get the correct Asia-session mask without any timezone gymnastics.
    if name in ("hour_utc", "minute_utc", "dayofweek_utc"):
        idx = ctx.index
        if not isinstance(idx, pd.DatetimeIndex):
            # No datetime info → return zeros so any time-filter expression
            # is no-op rather than crashing. Backtest path always supplies
            # a DatetimeIndex so this only matters for synthetic test frames.
            return pd.Series(0.0, index=idx)
        utc_idx = idx.tz_convert("UTC") if idx.tz is not None else idx
        if name == "hour_utc":
            return pd.Series(utc_idx.hour, index=idx, dtype=float)
        if name == "minute_utc":
            return pd.Series(utc_idx.minute, index=idx, dtype=float)
        if name == "dayofweek_utc":
            return pd.Series(utc_idx.dayofweek, index=idx, dtype=float)
    if name in ctx.frame.columns:
        return ctx.frame[name].astype(float)
    return pd.Series(0.0, index=ctx.index)


def _eval(node: AstNode, ctx: EvalContext, depth: int = 0) -> Any:
    if depth > 64:
        raise EvalError("Expression nests too deeply (max 64 levels).")
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Var):
        return _resolve_series(node.name, ctx)
    if isinstance(node, UnaryNeg):
        v = _eval(node.operand, ctx, depth + 1)
        if isinstance(v, pd.Series):
            return -v
        return -float(v)
    if isinstance(node, BinOp):
        a = _eval(node.lhs, ctx, depth + 1)
        b = _eval(node.rhs, ctx, depth + 1)
        if node.op == "+": return a + b
        if node.op == "-": return a - b
        if node.op == "*": return a * b
        if node.op == "/":
            # Series / scalar or Series / Series — guard /0 by replacing
            # zero divisors with NaN, then forward-fill so the bar uses
            # the prior period's value rather than producing infinities.
            if isinstance(b, pd.Series):
                return (a / b.replace(0.0, np.nan)).fillna(0.0)
            if b == 0:
                raise EvalError("Division by zero literal.")
            return a / b
        raise EvalError(f"Unknown operator {node.op!r}")
    if isinstance(node, Call):
        # Common natural-language aliases the LLM (and humans porting from
        # Pine / pandas-ta / TA-Lib) reach for. Resolve before the strict
        # lookup so a `correlation(...)` call works as well as `corr(...)`.
        # Add new aliases here as we see them in the wild — they don't
        # affect canonical hashing because parse() preserves the literal
        # token in the AST until eval time.
        name = _BUILTIN_ALIASES.get(node.fn, node.fn)
        fn = BUILTINS.get(name)
        if fn is None:
            raise EvalError(f"Unknown function {node.fn!r}. Allowed: {sorted(BUILTINS)}.")
        args = [_eval(a, ctx, depth + 1) for a in node.args]
        return fn(ctx, args)
    if isinstance(node, Cmp):
        a = _eval(node.lhs, ctx, depth + 1)
        b = _eval(node.rhs, ctx, depth + 1)
        # pandas Series supports element-wise comparison via the normal
        # operators; numpy handles the scalar case. Result type is
        # Series-of-bool (or Python bool when both sides are scalar).
        if node.op == "<":  return a < b
        if node.op == "<=": return a <= b
        if node.op == ">":  return a > b
        if node.op == ">=": return a >= b
        if node.op == "==": return a == b
        if node.op == "!=": return a != b
        raise EvalError(f"Unknown comparison {node.op!r}")
    if isinstance(node, BoolOp):
        if node.op == "not":
            v = _eval(node.operands[0], ctx, depth + 1)
            return ~_to_bool(v, ctx.index)
        # Element-wise boolean fold. We deliberately don't short-circuit
        # on Series — there's no useful early-exit when the operands are
        # per-bar bool vectors of the same length.
        acc = _to_bool(_eval(node.operands[0], ctx, depth + 1), ctx.index)
        for next_node in node.operands[1:]:
            nxt = _to_bool(_eval(next_node, ctx, depth + 1), ctx.index)
            acc = (acc & nxt) if node.op == "and" else (acc | nxt)
        return acc
    raise EvalError(f"Unhandled AST node: {type(node).__name__}")


def _to_bool(v: Any, index: pd.Index) -> pd.Series:
    """Coerce a value into a boolean Series. Scalars broadcast across the
    frame index. Numeric Series are truthy where non-zero (so a user can
    write `not (close > ema(close, 20))` interchangeably with arithmetic
    expressions that incidentally produce numeric series)."""
    if isinstance(v, pd.Series):
        if v.dtype == bool:
            return v
        return v.astype(bool)
    return pd.Series(bool(v), index=index)


def evaluate(expression: str, frame: pd.DataFrame, *, clip: bool = True) -> pd.Series:
    """Compile + evaluate `expression` against `frame`.

    `frame` must contain the OHLCV columns at minimum (open, high, low,
    close, volume); funding_rate / open_interest are optional and fall
    back to a zero series when absent.

    Returns a Series of floats over `frame.index`. If `clip` is True
    (the default, matching the legacy weights pipeline) the output is
    clamped to [-1, +1] so downstream discretization on a published
    threshold still has a bounded surface.
    """
    if not isinstance(expression, str):
        raise EvalError("Expression must be a string.")
    if not expression.strip():
        raise EvalError("Empty expression.")
    ast = parse(expression)
    ctx = EvalContext(frame=frame)
    out = _eval(ast, ctx)
    series = _as_series(out, frame.index).astype(float)
    series = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if clip:
        series = series.clip(-1.0, 1.0)
    return series


def evaluate_bool(expression: str, frame: pd.DataFrame) -> pd.Series:
    """Compile + evaluate `expression` as a boolean Series. Used by
    long_when / short_when entry-condition fields. A scalar bool result
    broadcasts across the frame index; a numeric result is truthy where
    non-zero (so the user can mix arithmetic patterns like
    `slope(close, 20) * adx(14)` with explicit comparisons)."""
    if not isinstance(expression, str):
        raise EvalError("Expression must be a string.")
    if not expression.strip():
        raise EvalError("Empty expression.")
    ast = parse(expression)
    ctx = EvalContext(frame=frame)
    out = _eval(ast, ctx)
    return _to_bool(out, frame.index)


def list_builtins() -> list[str]:
    """For the AI prompt and the /api/signals?op=compile endpoint."""
    return sorted(BUILTINS) + sorted(_SERIES_ACCESSORS)
