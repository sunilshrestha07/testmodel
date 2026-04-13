# -*- coding: utf-8 -*-
"""
bias_combiner.py
----------------
Combines technical structure, news, and DXY into a single
scored trade decision for the day.

Scoring weights:
  Gold H4 bias  : +/-0.40
  News bias     : +/-0.35
  DXY bias      : +/-0.25  (inverse -- bearish DXY is bullish gold)
  Divergence    :  x0.30 penalty when XAUUSD and DXY move together
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone as _tz

import MetaTrader5 as mt5

from config import LOG_LEVEL, SYMBOL, DXY_SYMBOL
from market_structure import check_dxy_divergence
from news_engine import get_full_news_report

log = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- Weights ------------------------------------------------------------------
W_TECHNICAL           = 0.40
W_NEWS                = 0.35
W_DXY                 = 0.25
W_VWAP_ALIGNED        = 0.15   # bonus when price is in discount/premium zone for direction
W_VWAP_PENALTY        = 0.10   # penalty when price is in the wrong zone for direction
DIVERGENCE_MULTIPLIER = 0.30   # collapses score when gold and DXY agree

# VWAP distance lot-size modifiers (distance measured in ATR multiples)
VWAP_OVEREXTENDED_ATR = 2.0    # >2x ATR from VWAP → reduce lot to 75% (mean-reversion risk)
VWAP_FRESH_ATR        = 0.5    # <=0.5x ATR from VWAP → full lot (fresh imbalance, clean entry)
VWAP_LOT_OVEREXTENDED = 0.75   # lot modifier when overextended

# --- Thresholds ---------------------------------------------------------------
STRONG_THRESHOLD = 0.60
WEAK_THRESHOLD   = 0.30

# Per-window minimum |score| required before a trade is permitted
WINDOW_THRESHOLDS: dict[str, float] = {
    "ASIAN_SWEEP_WINDOW":        0.50,
    "LONDON_OPEN_SILVER_BULLET": 0.50,
    "NY_AM_SILVER_BULLET":       0.70,   # weakest historical window — requires stronger confluence
    "NY_PM_SILVER_BULLET":       0.50,
}

# Maximum score achievable without the news component.
# Use this as `available_weight` when calling is_score_sufficient() in backtest
# mode, so thresholds scale proportionally to what is actually computable.
# VWAP is computable in real-time so it's included here (max aligned bonus = +0.15).
W_MAX_PARTIAL: float = W_TECHNICAL + W_DXY + W_VWAP_ALIGNED   # 0.40 + 0.25 + 0.15 = 0.80


# --- Result dataclass ---------------------------------------------------------

@dataclass
class BiasResult:
    # Raw inputs
    technical_bias: str     # BULLISH / BEARISH / RANGING
    news_bias:      str     # BULLISH / BEARISH / NEUTRAL
    dxy_bias:       str     # BULLISH / BEARISH / RANGING
    divergence:     str     # ALIGNED / DIVERGENCE

    # Score components
    technical_score: float
    news_score:      float
    dxy_score:       float
    vwap_score:      float
    raw_score:       float
    final_score:     float

    # Decision
    signal:       str    # STRONG_LONG / WEAK_LONG / STRONG_SHORT / WEAK_SHORT / NEUTRAL
    direction:    str    # LONG / SHORT / NONE
    lot_modifier: float  # 1.0 / 0.5 / 0.0
    strength:     str    # STRONG / WEAK / NONE

    # VWAP (populated after score; defaults allow old callers to work)
    session_vwap:      float = 0.0
    vwap_bias:         str   = "NEUTRAL"   # BULLISH / BEARISH / NEUTRAL
    vwap_session:      str   = ""          # Asian / London / NY
    vwap_lot_modifier: float = 1.0         # 1.0 (normal) or 0.75 (overextended >2x ATR)
    vwap_distance_atr: float = 0.0         # abs(price - vwap) expressed in ATR multiples
    m5_atr:            float = 0.0         # 14-bar M5 ATR at time of calculation

    # Regime & SHORT qualification (populated after score; defaults allow old callers to work)
    current_price:   float = 0.0
    h4_sma_50:       float = 0.0
    h4_sma_200:      float = 0.0
    bull_regime:     bool  = False   # True when price > 200-period H4 SMA
    short_qualified: bool  = True    # False when a SHORT was blocked by qualification rules
    short_block_reason: str = ""     # human-readable reason if short_qualified is False


# --- Internal helpers ---------------------------------------------------------

def is_score_sufficient(abs_score: float, window: str, available_weight: float = 1.0) -> bool:
    """
    Return True when |score| meets the minimum threshold for `window`.

    available_weight  -- fraction of total weight actually present in the score:
      • 1.0           live trading (technical + news + DXY all computed)
      • W_MAX_PARTIAL backtest mode (news absent; scale threshold proportionally)

    Example: NY_AM threshold is 0.70. In backtest with available_weight=0.65,
    the effective threshold is 0.70 * 0.65 = 0.455, so only a BULLISH H4 with
    a confirming DXY (score 0.65) passes.
    """
    min_score = WINDOW_THRESHOLDS.get(window, WEAK_THRESHOLD) * available_weight
    return abs_score >= min_score


def _get_h4_sma(symbol: str, period: int) -> float:
    """Compute `period`-bar H4 SMA for `symbol` from live MT5 data.
    Returns 0.0 when MT5 data is unavailable or history is too short."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, period + 5)
    if rates is None or len(rates) < period:
        return 0.0
    closes = [float(r["close"]) for r in rates[-period:]]
    return round(sum(closes) / period, 5)


def _get_current_price() -> float:
    """Return current bid price for the gold symbol."""
    tick = mt5.symbol_info_tick(SYMBOL)
    return float(tick.bid) if tick else 0.0


def _get_session_vwap() -> tuple[float, str]:
    """
    Calculate session VWAP from the current session open using M5 candles.
    Typical price = (H + L + C) / 3, weighted by tick_volume.
    Falls back to equal-weight (simple mean) when tick_volume is zero.

    Sessions (UTC):
      Asian  00:00 – 07:00
      London 07:00 – 12:00
      NY     12:00 – close

    Returns (vwap_price, session_label).  Returns (0.0, label) on data failure.
    """
    now = datetime.now(_tz.utc)
    h = now.hour
    if h < 7:
        sess_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        sess_label = "Asian"
    elif h < 12:
        sess_start = now.replace(hour=7, minute=0, second=0, microsecond=0)
        sess_label = "London"
    else:
        sess_start = now.replace(hour=12, minute=0, second=0, microsecond=0)
        sess_label = "NY"

    elapsed_min = max(5, int((now - sess_start).total_seconds() / 60))
    n_bars = elapsed_min // 5 + 5

    rates = mt5.copy_rates_from(SYMBOL, mt5.TIMEFRAME_M5, sess_start, n_bars)
    if rates is None or len(rates) == 0:
        log.warning("VWAP: no M5 data for %s session (start=%s)", sess_label, sess_start)
        return 0.0, sess_label

    total_tpv = 0.0
    total_vol = 0.0
    for r in rates:
        tp  = (float(r["high"]) + float(r["low"]) + float(r["close"])) / 3.0
        vol = float(r["tick_volume"]) if r["tick_volume"] > 0 else 1.0
        total_tpv += tp * vol
        total_vol += vol

    vwap = round(total_tpv / total_vol, 2) if total_vol > 0 else 0.0
    return vwap, sess_label


def _get_m5_atr(period: int = 14) -> float:
    """Compute `period`-bar ATR for SYMBOL from live M5 data.
    Returns 0.0 when MT5 data is unavailable or history is too short."""
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, period + 1)
    if rates is None or len(rates) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(rates)):
        high       = float(rates[i]["high"])
        low        = float(rates[i]["low"])
        prev_close = float(rates[i - 1]["close"])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return round(sum(trs[-period:]) / period, 5)


def _vwap_to_score(price: float, vwap: float, tentative_direction: str) -> tuple[float, str]:
    """
    Return (score, bias) based on price zone relative to session VWAP.

    NEW LOGIC (ICT institutional flow):
      LONG  + price BELOW VWAP → +W_VWAP_ALIGNED (+0.15)  [discount zone, institutional buy]
      LONG  + price ABOVE VWAP → -W_VWAP_PENALTY (-0.10)  [premium zone, dangerous for longs]
      SHORT + price ABOVE VWAP → +W_VWAP_ALIGNED (+0.15)  [premium zone, institutional sell]
      SHORT + price BELOW VWAP → -W_VWAP_PENALTY (-0.10)  [discount zone, dangerous for shorts]

    Returns (0.0, 'NEUTRAL') when VWAP is unavailable or direction is NONE.
    """
    if vwap <= 0 or price <= 0 or tentative_direction == "NONE":
        return 0.0, "NEUTRAL"

    if tentative_direction == "LONG":
        if price < vwap:
            return W_VWAP_ALIGNED, "BULLISH"   # discount zone — ideal long entry
        return -W_VWAP_PENALTY, "BEARISH"       # premium zone — overextended for longs

    if tentative_direction == "SHORT":
        if price > vwap:
            return W_VWAP_ALIGNED, "BEARISH"   # premium zone — ideal short entry
        return -W_VWAP_PENALTY, "BULLISH"       # discount zone — overextended for shorts

    return 0.0, "NEUTRAL"


def _bias_to_technical_score(bias: str) -> float:
    return {"BULLISH": W_TECHNICAL, "BEARISH": -W_TECHNICAL, "RANGING": 0.0}.get(bias, 0.0)


def _bias_to_news_score(bias: str) -> float:
    return {"BULLISH": W_NEWS, "BEARISH": -W_NEWS, "NEUTRAL": 0.0}.get(bias, 0.0)


def _dxy_to_gold_score(dxy_bias: str) -> float:
    # DXY is inverse of gold: bearish DXY = bullish for gold
    return {"BEARISH": W_DXY, "BULLISH": -W_DXY, "RANGING": 0.0}.get(dxy_bias, 0.0)


def _classify(score: float) -> tuple[str, str, float, str]:
    """Returns (signal, direction, lot_modifier, strength)."""
    abs_score = abs(score)
    side = "LONG" if score > 0 else "SHORT"

    if abs_score >= STRONG_THRESHOLD:
        return f"STRONG_{side}", side, 1.0, "STRONG"
    if abs_score >= WEAK_THRESHOLD:
        return f"WEAK_{side}", side, 0.5, "WEAK"
    return "NEUTRAL", "NONE", 0.0, "NONE"


# --- 1. Calculate bias score --------------------------------------------------

def calculate_bias_score(
    news_report: dict | None = None,
    window:      str | None = None,
) -> BiasResult:
    """
    Run all bias engines, apply weights and divergence penalty,
    and return a fully-populated BiasResult.

    Pass a pre-fetched `news_report` dict (from get_full_news_report())
    to avoid a duplicate HTTP call. If None, it will be fetched here.

    Pass `window` (e.g. "NY_AM_SILVER_BULLET") to enforce the per-window
    minimum bias score; the direction is set to NONE when the threshold is not met.
    """
    # Technical + DXY (requires active MT5 connection)
    gold_bias, dxy_bias, divergence = check_dxy_divergence()

    # News
    if news_report is None:
        news_report = get_full_news_report()
    news_bias = news_report.get("final_bias", "NEUTRAL")

    # Score components
    tech_score = _bias_to_technical_score(gold_bias)
    news_score = _bias_to_news_score(news_bias)
    dxy_score  = _dxy_to_gold_score(dxy_bias)

    # Intraday VWAP (direction-aware — requires tentative direction from prelim score)
    prelim_score      = tech_score + news_score + dxy_score
    tentative_dir     = "LONG" if prelim_score > 0 else ("SHORT" if prelim_score < 0 else "NONE")
    current_price_for_vwap = _get_current_price()
    session_vwap, vwap_session = _get_session_vwap()
    vwap_score, vwap_bias = _vwap_to_score(current_price_for_vwap, session_vwap, tentative_dir)

    # VWAP distance lot modifier (ATR-based)
    m5_atr = _get_m5_atr(period=14)
    if m5_atr > 0 and session_vwap > 0 and current_price_for_vwap > 0:
        dist = abs(current_price_for_vwap - session_vwap)
        vwap_distance_atr = round(dist / m5_atr, 2)
        vwap_lot_modifier = (
            VWAP_LOT_OVEREXTENDED if vwap_distance_atr > VWAP_OVEREXTENDED_ATR else 1.0
        )
    else:
        vwap_distance_atr = 0.0
        vwap_lot_modifier = 1.0

    raw_score  = tech_score + news_score + dxy_score + vwap_score

    # Divergence penalty
    if divergence == "DIVERGENCE":
        final_score = raw_score * DIVERGENCE_MULTIPLIER
    else:
        final_score = raw_score

    signal, direction, lot_modifier, strength = _classify(final_score)
    # Apply VWAP distance modifier to lot size (overextended → 75% of normal)
    lot_modifier = round(lot_modifier * vwap_lot_modifier, 4)

    # ── Regime & SHORT qualification filters ──────────────────────────────────
    current_price   = current_price_for_vwap   # already fetched above
    h4_sma_50       = _get_h4_sma(SYMBOL, 50)
    h4_sma_200      = _get_h4_sma(SYMBOL, 200)
    bull_regime     = h4_sma_200 > 0 and current_price > h4_sma_200
    short_qualified = True
    short_block_reason = ""

    if direction == "SHORT":
        if bull_regime:
            # Strong bull regime: price above 200 SMA → only LONGs permitted
            short_block_reason = (
                f"Bull regime: price {current_price:.2f} > 200 H4 SMA {h4_sma_200:.2f}"
            )
            log.info("SHORT blocked -- %s", short_block_reason)
            signal, direction, lot_modifier, strength = "NEUTRAL", "NONE", 0.0, "NONE"
            short_qualified = False

        elif dxy_bias != "BULLISH":
            # Condition 2: DXY H4 must be clearly BULLISH for a valid SHORT
            short_block_reason = f"DXY H4 bias is {dxy_bias}, not BULLISH"
            log.info("SHORT blocked -- %s", short_block_reason)
            signal, direction, lot_modifier, strength = "NEUTRAL", "NONE", 0.0, "NONE"
            short_qualified = False

        elif h4_sma_50 > 0 and current_price >= h4_sma_50:
            # Condition 3: price must be BELOW the 50-period H4 SMA
            short_block_reason = (
                f"Price {current_price:.2f} >= 50 H4 SMA {h4_sma_50:.2f}"
            )
            log.info("SHORT blocked -- %s", short_block_reason)
            signal, direction, lot_modifier, strength = "NEUTRAL", "NONE", 0.0, "NONE"
            short_qualified = False
        # Condition 1 (Gold H4 clearly BEARISH) is already guaranteed:
        # _classify returns SHORT only when final_score is sufficiently negative,
        # which requires technical_bias == "BEARISH" (lower highs + lower lows).
    # ─────────────────────────────────────────────────────────────────────────

    # ── Window score threshold ────────────────────────────────────────────────
    if window and direction != "NONE":
        if not is_score_sufficient(abs(final_score), window):
            log.info(
                "WEAK_SCORE for %s: |score| %.2f < threshold %.2f -- no trade",
                window, abs(final_score), WINDOW_THRESHOLDS.get(window, WEAK_THRESHOLD),
            )
            if not short_qualified:
                pass  # already blocked; don't overwrite the short_block_reason
            signal, direction, lot_modifier, strength = "NEUTRAL", "NONE", 0.0, "NONE"
    # ─────────────────────────────────────────────────────────────────────────

    return BiasResult(
        technical_bias     = gold_bias,
        news_bias          = news_bias,
        dxy_bias           = dxy_bias,
        divergence         = divergence,
        technical_score    = tech_score,
        news_score         = news_score,
        dxy_score          = dxy_score,
        vwap_score         = vwap_score,
        raw_score          = raw_score,
        final_score        = final_score,
        signal             = signal,
        direction          = direction,
        lot_modifier       = lot_modifier,
        strength           = strength,
        session_vwap       = session_vwap,
        vwap_bias          = vwap_bias,
        vwap_session       = vwap_session,
        vwap_lot_modifier  = vwap_lot_modifier,
        vwap_distance_atr  = vwap_distance_atr,
        m5_atr             = m5_atr,
        current_price      = current_price,
        h4_sma_50          = h4_sma_50,
        h4_sma_200         = h4_sma_200,
        bull_regime        = bull_regime,
        short_qualified    = short_qualified,
        short_block_reason = short_block_reason,
    )


# --- 2 & 3. Convenience wrappers ---------------------------------------------

def get_trade_direction(result: BiasResult | None = None) -> str:
    """Return 'LONG', 'SHORT', or 'NONE'."""
    if result is None:
        result = calculate_bias_score()
    return result.direction


def get_lot_modifier(result: BiasResult | None = None) -> float:
    """Return 1.0 (full size), 0.5 (half size), or 0.0 (no trade)."""
    if result is None:
        result = calculate_bias_score()
    return result.lot_modifier


# --- 4. Print bias report -----------------------------------------------------

def print_bias_report(result: BiasResult) -> None:
    sign = lambda v: f"+{v:.2f}" if v >= 0 else f"{v:.2f}"
    bar  = "-" * 45

    tech_label = f"{result.technical_bias:<8} ({sign(result.technical_score)})"
    news_label = f"{result.news_bias:<8} ({sign(result.news_score)})"
    dxy_label  = f"{result.dxy_bias:<8} ({sign(result.dxy_score)})"
    if result.session_vwap > 0:
        dist_str = (f"  dist={result.vwap_distance_atr:.1f}x ATR"
                    if result.m5_atr > 0 else "")
        lot_str  = (f"  [LOT x{result.vwap_lot_modifier:.2f} -- overextended]"
                    if result.vwap_lot_modifier < 1.0 else "")
        vwap_label = (
            f"{result.vwap_bias:<8} ({sign(result.vwap_score)})  "
            f"VWAP={result.session_vwap:.2f}  [{result.vwap_session}]"
            f"{dist_str}{lot_str}"
        )
    else:
        vwap_label = "NEUTRAL  (unavailable)"

    if result.divergence == "DIVERGENCE":
        div_label = f"DIVERGENCE  (score x{DIVERGENCE_MULTIPLIER} penalty applied)"
    else:
        div_label = "ALIGNED     (no penalty)"

    if result.direction == "LONG":
        signal_icon = "[LONG]"
    elif result.direction == "SHORT":
        signal_icon = "[SHORT]"
    else:
        signal_icon = "[NO TRADE]"

    strong_weak  = result.strength if result.strength != "NONE" else ""
    signal_label = (
        f"{strong_weak} {result.direction}".strip()
        if result.direction != "NONE"
        else "NEUTRAL"
    )

    # Regime / SMA labels
    regime_label = "BULL REGIME (>200 SMA)" if result.bull_regime else "normal"
    sma50_label  = (
        f"{result.h4_sma_50:.2f}  (price {'BELOW' if result.current_price < result.h4_sma_50 else 'ABOVE'})"
        if result.h4_sma_50 > 0 else "unavailable"
    )
    sma200_label = (
        f"{result.h4_sma_200:.2f}  (price {'BELOW' if result.current_price < result.h4_sma_200 else 'ABOVE'})"
        if result.h4_sma_200 > 0 else "unavailable"
    )

    print()
    print("=" * 55)
    print("  Daily Bias Report - XAUUSD Bot")
    print("=" * 55)
    print(f"  Technical  : {tech_label}")
    print(f"  News       : {news_label}")
    print(f"  DXY        : {dxy_label}")
    print(f"  VWAP       : {vwap_label}")
    print(f"  Divergence : {div_label}")
    bar55 = "-" * 50
    print(f"  {bar55}")
    print(f"  Current Price : {result.current_price:.2f}")
    print(f"  H4 50 SMA     : {sma50_label}")
    print(f"  H4 200 SMA    : {sma200_label}")
    print(f"  Regime        : {regime_label}")
    if not result.short_qualified and result.short_block_reason:
        print(f"  SHORT blocked : {result.short_block_reason}")
    print(f"  {bar55}")
    print(f"  Raw Score  : {sign(result.raw_score)}")
    if result.divergence == "DIVERGENCE":
        print(f"  After Pen. : {sign(result.final_score)}")
    print(f"  {bar55}")
    print(f"  Final Score: {sign(result.final_score)}  ->  {signal_label}  {signal_icon}")
    print(f"  Lot Modifier: {result.lot_modifier}x")
    print("=" * 55)


# --- Entry Point --------------------------------------------------------------

if __name__ == "__main__":
    from mt5_connector import connect, disconnect

    print("Connecting to MT5...")
    if not connect():
        raise SystemExit("Could not connect to MT5.")

    try:
        print("Fetching news report...")
        news_report = get_full_news_report()

        print("\nCalculating bias score...")
        result = calculate_bias_score(news_report=news_report)

        print_bias_report(result)

        sign = "+" if result.final_score >= 0 else ""
        print(
            f"\n  >> TRADE DECISION: {result.signal}"
            f"  |  Score: {sign}{result.final_score:.2f}"
            f"  |  Lots: {result.lot_modifier}x"
        )
    finally:
        disconnect()
