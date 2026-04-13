# -*- coding: utf-8 -*-
"""
fvg_detector.py
---------------
Core ICT technical engine for XAUUSD.

Detects on M1/M5 candles:
  - Liquidity sweeps  (SSL / BSL taken)
  - Fair Value Gaps   (FVG / imbalance)
  - Market Structure Shifts (MSS / CHoCH)
  - Premium / Discount zones
  - Full entry setup with SL / TP / RR

Gold pip definition used here:
  1 pip = $1.00  (XAUUSD point = $0.01, 100 points = 1 pip)
  min_wick_pips=2 means the wick must exceed the level by >= $2.00
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

from config import SYMBOL, LOG_LEVEL
from mt5_connector import fetch_candles

log = logging.getLogger(__name__)

# --- Constants ----------------------------------------------------------------
PIP_VALUE    = 1.0   # $1.00 per pip for XAUUSD
SL_BUFFER    = 2.0   # extra pip buffer added beyond the FVG edge for SL
MIN_RR       = 1.5   # discard setups below this reward:risk ratio
M5_LOOKBACK  = 100   # M5 candles to scan for FVG
M1_LOOKBACK  = 200   # M1 candles to scan for MSS

# Quality-filter thresholds (applied in all detection functions)
MIN_FVG_PIPS          = 3.0   # FVG must be at least 3 pips wide
FVG_FRESHNESS_CANDLES = 5     # FVG must form within ±5 candles of the MSS
MIN_MSS_BREAK_PIPS    = 3.0   # MSS close must exceed the reference level by ≥ 3 pips
MIN_MSS_BODY_RATIO    = 0.60  # MSS candle body ≥ 60 % of its total H-L range
MIN_RETRACE_PCT       = 0.20  # after MSS, price must retrace ≥ 20 % of impulse  (was 30%)
MIN_OB_PIPS           = 5.0   # Order Block body must be ≥ 5 pips wide

# Strong-sweep MSS relaxation
# When the sweep wick extends more than STRONG_SWEEP_PIPS beyond the key level,
# the sweep itself is strong institutional confirmation. The MSS body-ratio filter
# is relaxed from 0.60 → 0.40 so that less decisive candles can still qualify.
# min_break_pips (3 pips) is NOT relaxed — the structure break must still be real.
STRONG_SWEEP_PIPS           = 5.0   # wick depth (pips) that qualifies as "strong"
STRONG_SWEEP_MSS_BODY_RATIO = 0.40  # relaxed body/range ratio for strong-sweep sessions


# --- Data classes -------------------------------------------------------------

@dataclass
class FVGZone:
    top:         float
    bottom:      float
    midpoint:    float
    candle_index: int
    direction:   str   # "bullish" or "bearish"
    filled:      bool = False

    def __str__(self) -> str:
        return (f"FVG({self.direction.upper()})  "
                f"top={self.top:.2f}  bot={self.bottom:.2f}  "
                f"mid={self.midpoint:.2f}  idx={self.candle_index}")


@dataclass
class SweepResult:
    found:        bool
    candle_index: int = -1
    sweep_price:  float = 0.0
    level:        float = 0.0
    direction:    str = ""   # "above" or "below"

    def __str__(self) -> str:
        if not self.found:
            return "SweepResult(not found)"
        return (f"SweepResult({self.direction})  "
                f"level={self.level:.2f}  wick_to={self.sweep_price:.2f}  "
                f"idx={self.candle_index}")


@dataclass
class MSSResult:
    found:        bool
    candle_index: int = -1
    break_price:  float = 0.0

    def __str__(self) -> str:
        if not self.found:
            return "MSSResult(not found)"
        return (f"MSSResult  break_price={self.break_price:.2f}  "
                f"idx={self.candle_index}")


@dataclass
class OrderBlockResult:
    """
    Represents an ICT Order Block (Method A) or MSS-candle entry zone (Method B).

    For an Order Block:
      body_top / body_bottom = candle body (max/min of open, close)
      midpoint               = body midpoint = 50% entry price
      candle_high / candle_low = full candle range — used as the SL anchor

    For an MSS-candle entry zone, body_top/bottom == candle_high/low (full candle).
    """
    found:        bool
    candle_index: int   = -1
    direction:    str   = ""    # "bullish" (serves LONG) or "bearish" (serves SHORT)
    body_top:     float = 0.0   # max(open, close)
    body_bottom:  float = 0.0   # min(open, close)
    midpoint:     float = 0.0   # 50 % entry: (body_top + body_bottom) / 2
    candle_high:  float = 0.0   # full candle high  (SL anchor for shorts)
    candle_low:   float = 0.0   # full candle low   (SL anchor for longs)

    def __str__(self) -> str:
        if not self.found:
            return "OrderBlockResult(not found)"
        return (
            f"OrderBlock({self.direction.upper()})  "
            f"body={self.body_bottom:.2f}-{self.body_top:.2f}  "
            f"mid={self.midpoint:.2f}  idx={self.candle_index}"
        )


@dataclass
class EntrySetup:
    valid:         bool
    direction:     str = ""
    entry:         float = 0.0
    sl:            float = 0.0
    tp:            float = 0.0
    rr:            float = 0.0
    fvg_zone:      Optional[FVGZone] = None
    sweep_level:   float = 0.0
    reason:        str = ""

    def __str__(self) -> str:
        if not self.valid:
            return f"EntrySetup(invalid: {self.reason})"
        return (
            f"EntrySetup({self.direction})  "
            f"entry={self.entry:.2f}  sl={self.sl:.2f}  tp={self.tp:.2f}  "
            f"RR=1:{self.rr:.1f}  sweep_lvl={self.sweep_level:.2f}  "
            f"reason={self.reason!r}"
        )


# --- 0. ATR helper ------------------------------------------------------------

def _compute_atr(candles: pd.DataFrame, period: int = 14) -> float:
    """Average True Range over the last `period` bars.
    Returns 0.0 when there is insufficient history."""
    if len(candles) < period + 1:
        return 0.0
    highs  = candles["high"].values
    lows   = candles["low"].values
    closes = candles["close"].values
    tr_vals = [
        max(highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]))
        for i in range(1, len(candles))
    ]
    return float(sum(tr_vals[-period:]) / period)


# --- 1. Liquidity Sweep -------------------------------------------------------

def detect_liquidity_sweep(
    candles:       pd.DataFrame,
    level:         float,
    direction:     str,
    min_wick_pips: float = 2.0,
) -> SweepResult:
    """
    Scan the last N candles for a wick that exceeds `level` by at least
    min_wick_pips * PIP_VALUE, then closes back on the other side.

    direction="above" -> sweep of buy-stops above the level (bearish sweep)
                         wick high > level + threshold, close < level
    direction="below" -> sweep of sell-stops below the level (bullish sweep)
                         wick low  < level - threshold, close > level

    Returns SweepResult. Scans from newest to oldest so the most recent
    sweep is returned first.
    """
    threshold = min_wick_pips * PIP_VALUE
    highs  = candles["high"].values
    lows   = candles["low"].values
    closes = candles["close"].values
    n      = len(candles)

    for i in range(n - 1, -1, -1):
        if direction == "above":
            if highs[i] >= level + threshold and closes[i] < level:
                return SweepResult(
                    found=True,
                    candle_index=i,
                    sweep_price=highs[i],
                    level=level,
                    direction="above",
                )
        elif direction == "below":
            if lows[i] <= level - threshold and closes[i] > level:
                return SweepResult(
                    found=True,
                    candle_index=i,
                    sweep_price=lows[i],
                    level=level,
                    direction="below",
                )

    return SweepResult(found=False)


# --- 1b. Sweep near-miss diagnostics -----------------------------------------

def diagnose_sweep_miss(
    candles:       pd.DataFrame,
    level:         float,
    direction:     str,
    min_wick_pips: float = 2.0,
    label:         str   = "",
) -> dict:
    """
    Called when detect_liquidity_sweep returned found=False.
    Analyses how close price got to completing the sweep and logs a
    DEBUG-level line for each candidate level.

    Returns a dict with the diagnostic fields for programmatic use
    (e.g. the backtester session funnel report).

    Keys returned
    -------------
    label          : level label ("ASL", "PDH", …)
    level          : the sweep reference price
    direction      : "above" | "below"
    threshold_pips : min_wick_pips required
    atr            : ATR(14) of the candle window  (in price, not pips)
    atr_pips       : ATR expressed in pips
    best_extreme   : the best wick price towards the level (high or low)
    reached_level  : True  — price crossed the level at least once
    pips_short     : pips the best wick fell short of level ± threshold
                     (0 when threshold was met but close-back failed)
    close_back_miss: count of candles that met the wick threshold but
                     failed the close-back-across-level condition
    verdict        : short human-readable summary
    """
    highs  = candles["high"].values
    lows   = candles["low"].values
    closes = candles["close"].values
    n      = len(candles)

    threshold = min_wick_pips * PIP_VALUE
    atr       = _compute_atr(candles)
    atr_pips  = round(atr / PIP_VALUE, 1) if atr > 0 else 0.0
    prefix    = f"[{label}]" if label else ""

    if direction == "above":
        # We need: high >= level + threshold  AND  close < level
        best_high        = float(highs.max())
        reached_level    = best_high >= level
        required         = level + threshold
        pips_short_raw   = required - best_high          # positive ⟹ short
        pips_short       = max(0.0, pips_short_raw) / PIP_VALUE
        close_back_miss  = sum(
            1 for i in range(n)
            if highs[i] >= required and closes[i] >= level
        )

        if not reached_level:
            verdict = (
                f"price never reached level -- "
                f"best high {best_high:.2f}, "
                f"gap {(level - best_high) / PIP_VALUE:.1f} pips"
            )
        elif pips_short > 0:
            verdict = (
                f"wick crossed level but fell {pips_short:.1f} pips short "
                f"of {min_wick_pips:.1f}-pip threshold "
                f"(best high {best_high:.2f}, needs {required:.2f})"
            )
        elif close_back_miss:
            verdict = (
                f"wick met threshold on {close_back_miss} candle(s) "
                f"but close did NOT pull back below {level:.2f} (engulf, not sweep)"
            )
        else:
            verdict = "unclassified miss"

        log.debug(
            "NO_SWEEP %s above %.2f | threshold %.1f pip | %s | ATR %.1f pip",
            prefix, level, min_wick_pips, verdict, atr_pips,
        )
        return {
            "label":           label,
            "level":           level,
            "direction":       direction,
            "threshold_pips":  min_wick_pips,
            "atr":             round(atr, 4),
            "atr_pips":        atr_pips,
            "best_extreme":    round(best_high, 2),
            "reached_level":   reached_level,
            "pips_short":      round(pips_short, 1),
            "close_back_miss": close_back_miss,
            "verdict":         verdict,
        }

    else:  # direction == "below"
        # We need: low <= level - threshold  AND  close > level
        best_low         = float(lows.min())
        reached_level    = best_low <= level
        required         = level - threshold
        pips_short_raw   = best_low - required               # positive ⟹ short
        pips_short       = max(0.0, pips_short_raw) / PIP_VALUE
        close_back_miss  = sum(
            1 for i in range(n)
            if lows[i] <= required and closes[i] <= level
        )

        if not reached_level:
            verdict = (
                f"price never reached level -- "
                f"best low {best_low:.2f}, "
                f"gap {(best_low - level) / PIP_VALUE:.1f} pips"
            )
        elif pips_short > 0:
            verdict = (
                f"wick crossed level but fell {pips_short:.1f} pips short "
                f"of {min_wick_pips:.1f}-pip threshold "
                f"(best low {best_low:.2f}, needs {required:.2f})"
            )
        elif close_back_miss:
            verdict = (
                f"wick met threshold on {close_back_miss} candle(s) "
                f"but close did NOT pull back above {level:.2f} (engulf, not sweep)"
            )
        else:
            verdict = "unclassified miss"

        log.debug(
            "NO_SWEEP %s below %.2f | threshold %.1f pip | %s | ATR %.1f pip",
            prefix, level, min_wick_pips, verdict, atr_pips,
        )
        return {
            "label":           label,
            "level":           level,
            "direction":       direction,
            "threshold_pips":  min_wick_pips,
            "atr":             round(atr, 4),
            "atr_pips":        atr_pips,
            "best_extreme":    round(best_low, 2),
            "reached_level":   reached_level,
            "pips_short":      round(pips_short, 1),
            "close_back_miss": close_back_miss,
            "verdict":         verdict,
        }


# --- 2. Fair Value Gap --------------------------------------------------------

def detect_fvg(
    candles:       pd.DataFrame,
    direction:     str,
    start_idx:     int   = 0,
    mss_idx:       int   = -1,
    min_pips:      float = MIN_FVG_PIPS,
    current_price: float = 0.0,
    swing_high:    float = 0.0,
    swing_low:     float = 0.0,
) -> Optional[FVGZone]:
    """
    Scan candles[start_idx:] for qualifying unfilled FVGs, then return the
    best candidate.

    Quality filters applied:
      • Minimum width  : gap >= min_pips * PIP_VALUE  (default 3 pips)
      • Freshness      : when mss_idx >= 0, FVG candle index must be within
                         ±FVG_FRESHNESS_CANDLES of mss_idx
      • Zone           : when swing_high/swing_low are non-zero, FVG midpoint
                         must be in discount zone (bullish: below equilibrium)
                         or premium zone (bearish: above equilibrium)
      • Not filled     : subsequent price action has not traded back into the gap

    Candidate selection (when multiple FVGs pass all filters):
      • If current_price > 0 → pick the FVG whose midpoint is closest to price
        (most relevant pullback target)
      • Otherwise → pick the most recent one (highest candle index)

    Returns the selected FVGZone, or None when no FVG passes all filters.
    """
    highs  = candles["high"].values
    lows   = candles["low"].values
    n      = len(candles)

    # Zone equilibrium (used when swing levels are provided)
    _zone_active = swing_high > 0 and swing_low > 0 and swing_high > swing_low
    _equilibrium = (swing_high + swing_low) / 2.0 if _zone_active else 0.0

    candidates: list[FVGZone] = []

    for i in range(n - 3, max(start_idx - 1, -1), -1):
        # Freshness: FVG must be within ±FVG_FRESHNESS_CANDLES of the MSS
        if mss_idx >= 0:
            if i < mss_idx - FVG_FRESHNESS_CANDLES:
                break   # scanning oldest → no point going further
            if i > mss_idx + FVG_FRESHNESS_CANDLES:
                continue

        if direction == "bullish":
            gap_bottom = highs[i]
            gap_top    = lows[i + 2]
            if gap_top <= gap_bottom:
                continue
            width = gap_top - gap_bottom

        elif direction == "bearish":
            gap_top    = lows[i]
            gap_bottom = highs[i + 2]
            if gap_top <= gap_bottom:
                continue
            width = gap_top - gap_bottom

        else:
            continue

        # Minimum width filter
        if width < min_pips * PIP_VALUE:
            continue

        midpoint = (gap_top + gap_bottom) / 2.0

        # Zone filter (discount for longs, premium for shorts)
        if _zone_active:
            if direction == "bullish" and midpoint >= _equilibrium:
                continue   # must be below equilibrium (discount zone)
            if direction == "bearish" and midpoint <= _equilibrium:
                continue   # must be above equilibrium (premium zone)

        # Filled check: gap violated by subsequent price action
        if direction == "bullish":
            filled = any(lows[j] <= gap_bottom for j in range(i + 3, n))
        else:
            filled = any(highs[j] >= gap_top for j in range(i + 3, n))

        if filled:
            continue

        candidates.append(FVGZone(
            top=gap_top, bottom=gap_bottom, midpoint=midpoint,
            candle_index=i, direction=direction, filled=False,
        ))

    if not candidates:
        return None

    # Select the best candidate
    if current_price > 0:
        # Closest midpoint to current price (most likely pullback target)
        return min(candidates, key=lambda z: abs(z.midpoint - current_price))

    # Default: most recent (highest candle index = most recently formed)
    return max(candidates, key=lambda z: z.candle_index)


# --- 3. Market Structure Shift ------------------------------------------------

def detect_market_structure_shift(
    candles:        pd.DataFrame,
    direction:      str,
    after_idx:      int   = 0,
    min_break_pips: float = MIN_MSS_BREAK_PIPS,
    min_body_ratio: float = MIN_MSS_BODY_RATIO,
) -> MSSResult:
    """
    Detect a Market Structure Shift (MSS / CHoCH) in candles[after_idx:].

    Bullish MSS: a candle closes ABOVE the highest high seen in the window
                 that precedes the sweep (break of structure to the upside).
    Bearish MSS: a candle closes BELOW the lowest low seen in the window
                 that precedes the sweep.

    Quality filters (applied to each candidate break candle):
      • min_break_pips  : close must exceed the reference level by at least
                          this many pips (default 3) — prevents 1-pip tickle breaks
      • min_body_ratio  : abs(close - open) / (high - low) >= ratio (default 0.60)
                          — requires a decisive body, not a wick MSS

    If a candidate fails quality, the scan continues searching (ref level still
    advances) so a later, stronger break can still qualify.

    Returns MSSResult with the first qualifying candle.
    """
    opens  = candles["open"].values
    closes = candles["close"].values
    highs  = candles["high"].values
    lows   = candles["low"].values
    n      = len(candles)
    start  = max(after_idx, 1)
    min_break = min_break_pips * PIP_VALUE

    if direction == "bullish":
        ref_high = highs[after_idx:start + 1].max() if start > after_idx else highs[after_idx]
        for i in range(start, n):
            if closes[i] > ref_high + min_break:
                body  = abs(closes[i] - opens[i])
                total = highs[i] - lows[i]
                if total <= 0 or (body / total) >= min_body_ratio:
                    return MSSResult(found=True, candle_index=i, break_price=closes[i])
            # Always advance ref so a stronger break later is measured correctly
            ref_high = max(ref_high, highs[i])

    elif direction == "bearish":
        ref_low = lows[after_idx:start + 1].min() if start > after_idx else lows[after_idx]
        for i in range(start, n):
            if closes[i] < ref_low - min_break:
                body  = abs(closes[i] - opens[i])
                total = highs[i] - lows[i]
                if total <= 0 or (body / total) >= min_body_ratio:
                    return MSSResult(found=True, candle_index=i, break_price=closes[i])
            ref_low = min(ref_low, lows[i])

    return MSSResult(found=False)


# --- 3b. MSS retrace validator -----------------------------------------------

def check_mss_retrace(
    candles:     pd.DataFrame,
    mss_idx:     int,
    mss_close:   float,
    sweep_price: float,
    direction:   str,
    min_pct:     float = MIN_RETRACE_PCT,
) -> bool:
    """
    After a MSS, verify that price retraced at least min_pct (30 %) of the
    impulse move back toward the sweep extreme before the FVG entry triggers.

    Bullish:
      impulse     = mss_close - sweep_low   (sweep_price is the wick low)
      retrace_tgt = mss_close - min_pct * impulse
      Pass when any candle after mss_idx has a low ≤ retrace_tgt.

    Bearish:
      impulse     = sweep_high - mss_close
      retrace_tgt = mss_close + min_pct * impulse
      Pass when any candle after mss_idx has a high ≥ retrace_tgt.

    Returns True (pass) in edge cases: no candles after MSS, or the impulse
    is less than 2 pips (too small to validate meaningfully).
    """
    n = len(candles)
    if mss_idx >= n - 1:
        return True   # no candles after MSS; can't verify — don't block

    impulse = abs(mss_close - sweep_price)
    if impulse < PIP_VALUE * 2:
        return True   # micro-move; skip check

    highs = candles["high"].values
    lows  = candles["low"].values

    if direction == "bullish":
        target = mss_close - min_pct * impulse
        return any(lows[i] <= target for i in range(mss_idx + 1, n))
    else:
        target = mss_close + min_pct * impulse
        return any(highs[i] >= target for i in range(mss_idx + 1, n))


# --- 4. Order Block & MSS-candle entry (fallback when no FVG) -----------------

def detect_order_block(
    candles:     pd.DataFrame,
    direction:   str,
    sweep_idx:   int,
    mss_idx:     int,
    min_ob_pips: float = MIN_OB_PIPS,
) -> OrderBlockResult:
    """
    Find the ICT Order Block — the LAST opposing candle immediately before
    the MSS impulse, scanning from MSS back toward the sweep.

    Bullish OB (serves LONG entry):
      Last BEARISH candle (close < open) in candles[sweep_idx:mss_idx].
      Entry at body midpoint.  SL reference = candle low.

    Bearish OB (serves SHORT entry):
      Last BULLISH candle (close > open) in candles[sweep_idx:mss_idx].
      Entry at body midpoint.  SL reference = candle high.

    Quality filters:
      • min_ob_pips  : OB body (|close - open|) must be ≥ 5 pips
      • Untested     : no candle after the OB (up to end of sub-window) has
                       closed back through the OB body, invalidating the level

    Scans newest-to-oldest (closest to MSS first).
    """
    n = len(candles)
    if mss_idx < 0 or mss_idx >= n:
        return OrderBlockResult(found=False)

    opens  = candles["open"].values
    closes = candles["close"].values
    highs  = candles["high"].values
    lows   = candles["low"].values

    end   = min(mss_idx - 1, n - 1)   # OB must be BEFORE the MSS candle itself
    start = max(sweep_idx, 0)
    min_body = min_ob_pips * PIP_VALUE

    for i in range(end, start - 1, -1):
        if direction == "bullish" and closes[i] < opens[i]:   # bearish candle → bullish OB
            body_top    = opens[i]
            body_bottom = closes[i]
            body_size   = body_top - body_bottom

            # Minimum size
            if body_size < min_body:
                continue

            # Untested: no subsequent candle has closed below the OB body bottom
            # (a close below body_bottom would invalidate the OB as support)
            if any(closes[j] < body_bottom for j in range(i + 1, n)):
                continue

            return OrderBlockResult(
                found=True, candle_index=i, direction="bullish",
                body_top=body_top, body_bottom=body_bottom,
                midpoint=(body_top + body_bottom) / 2,
                candle_high=highs[i], candle_low=lows[i],
            )

        if direction == "bearish" and closes[i] > opens[i]:   # bullish candle → bearish OB
            body_top    = closes[i]
            body_bottom = opens[i]
            body_size   = body_top - body_bottom

            # Minimum size
            if body_size < min_body:
                continue

            # Untested: no subsequent candle has closed above the OB body top
            if any(closes[j] > body_top for j in range(i + 1, n)):
                continue

            return OrderBlockResult(
                found=True, candle_index=i, direction="bearish",
                body_top=body_top, body_bottom=body_bottom,
                midpoint=(body_top + body_bottom) / 2,
                candle_high=highs[i], candle_low=lows[i],
            )

    return OrderBlockResult(found=False)


def detect_mss_candle_entry(
    candles: pd.DataFrame,
    mss_idx: int,
) -> OrderBlockResult:
    """
    Method B fallback: use the MSS candle itself as the entry zone.

    Entry at 50 % of the candle: (high + low) / 2.
    SL reference = candle low (for longs) or candle high (for shorts).
    Returns an OrderBlockResult (body == full candle for MSS entries).
    """
    n = len(candles)
    if mss_idx < 0 or mss_idx >= n:
        return OrderBlockResult(found=False)

    highs = candles["high"].values
    lows  = candles["low"].values
    mid   = (highs[mss_idx] + lows[mss_idx]) / 2

    return OrderBlockResult(
        found=True,
        candle_index=mss_idx,
        direction="",   # filled by caller based on trade direction
        body_top=highs[mss_idx],
        body_bottom=lows[mss_idx],
        midpoint=mid,
        candle_high=highs[mss_idx],
        candle_low=lows[mss_idx],
    )


# --- 5. Premium / Discount Zone Check ----------------------------------------

def is_fvg_in_correct_zone(
    fvg:        FVGZone,
    swing_high: float,
    swing_low:  float,
    direction:  str,
) -> bool:
    """
    Validate that the FVG sits in the correct premium / discount zone.

    equilibrium = 50% of the swing range

    LONG  setup: FVG midpoint must be BELOW equilibrium (discount zone)
    SHORT setup: FVG midpoint must be ABOVE equilibrium (premium zone)

    Returns True if the FVG is in the correct zone.
    """
    if swing_high <= swing_low:
        log.warning("is_fvg_in_correct_zone: swing_high <= swing_low, skipping")
        return False

    equilibrium = (swing_high + swing_low) / 2.0

    if direction == "bullish":
        return fvg.midpoint < equilibrium
    if direction == "bearish":
        return fvg.midpoint > equilibrium

    return False


# --- 5. Full Entry Setup ------------------------------------------------------

def get_entry_setup(
    symbol:         str,
    direction:      str,
    session_levels: dict,
) -> EntrySetup:
    """
    Build a complete trade setup from live M5 + M1 candles.

    session_levels dict expects keys:
      pdh, pdl           -- previous day high / low
      ash, asl           -- asian session high / low (may be None)

    Logic:
      1. Fetch M5 candles (FVG scan) and M1 candles (MSS scan)
      2. Pick sweep targets based on direction:
           LONG  -> check sweep of ASL / PDL  (sell-stops below)
           SHORT -> check sweep of ASH / PDH  (buy-stops above)
      3. Find a liquidity sweep of the best target
      4. Find MSS after the sweep
      5. Find an unfilled FVG after the sweep, in the correct zone
      6. Calculate entry / SL / TP / RR

    Entry  : FVG midpoint
    SL     : FVG bottom - buffer (LONG) | FVG top + buffer (SHORT)
    TP     : PDH (LONG) | PDL (SHORT), or 3x SL distance minimum
    """
    # Fetch candles
    df_m5 = fetch_candles(symbol, timeframe=mt5.TIMEFRAME_M5, count=M5_LOOKBACK)
    df_m1 = fetch_candles(symbol, timeframe=mt5.TIMEFRAME_M1, count=M1_LOOKBACK)

    if df_m5.empty or df_m1.empty:
        return EntrySetup(valid=False, reason="Failed to fetch M5/M1 candles")

    # Determine sweep direction and candidate levels
    if direction == "bullish":
        sweep_dir = "below"
        targets: list[tuple[str, float]] = []
        if session_levels.get("asl"):
            targets.append(("ASL", float(session_levels["asl"])))
        if session_levels.get("pdl"):
            targets.append(("PDL", float(session_levels["pdl"])))
        tp_target = session_levels.get("pdh")
    else:
        sweep_dir = "above"
        targets = []
        if session_levels.get("ash"):
            targets.append(("ASH", float(session_levels["ash"])))
        if session_levels.get("pdh"):
            targets.append(("PDH", float(session_levels["pdh"])))
        tp_target = session_levels.get("pdl")

    if not targets:
        return EntrySetup(valid=False, reason="No sweep targets available in session_levels")

    # --- Step 1: Find a liquidity sweep on M5 --------------------------------
    sweep: Optional[SweepResult] = None
    sweep_label = ""
    for label, level in targets:
        result = detect_liquidity_sweep(df_m5, level, sweep_dir)
        if result.found:
            sweep       = result
            sweep_label = label
            log.info("Sweep found: %s @ %.2f  (candle %d)", label, level, result.candle_index)
            break

    if sweep is None:
        return EntrySetup(
            valid=False,
            reason=f"No liquidity sweep found for {direction} setup "
                   f"(checked: {', '.join(l for l, _ in targets)})",
        )

    # --- Step 2: MSS on M1 after the sweep -----------------------------------
    # Map the M5 sweep candle time to an M1 index
    sweep_time = df_m5.index[sweep.candle_index]
    m1_after   = df_m1.index.searchsorted(sweep_time)

    mss = detect_market_structure_shift(df_m1, direction, after_idx=m1_after)
    if not mss.found:
        # Strong sweep: wick > STRONG_SWEEP_PIPS beyond the level gives extra
        # institutional confirmation, so relax the MSS body-ratio requirement.
        sweep_depth = abs(sweep.sweep_price - sweep.level) / PIP_VALUE
        if sweep_depth > STRONG_SWEEP_PIPS:
            mss = detect_market_structure_shift(
                df_m1, direction, after_idx=m1_after,
                min_break_pips=MIN_MSS_BREAK_PIPS,
                min_body_ratio=STRONG_SWEEP_MSS_BODY_RATIO,
            )
    if not mss.found:
        return EntrySetup(
            valid=False,
            sweep_level=sweep.level,
            reason=f"{sweep_label} swept but no MSS confirmed on M1",
        )
    log.info("MSS found at M1 candle %d, close=%.2f", mss.candle_index, mss.break_price)

    # --- Step 3: FVG on M5 after the sweep -----------------------------------
    fvg = detect_fvg(df_m5, direction, start_idx=sweep.candle_index)
    if fvg is None:
        return EntrySetup(
            valid=False,
            sweep_level=sweep.level,
            reason=f"{sweep_label} swept, MSS confirmed, but no {direction} FVG found",
        )
    if fvg.filled:
        return EntrySetup(
            valid=False,
            sweep_level=sweep.level,
            reason=f"FVG found but already filled (mid={fvg.midpoint:.2f})",
        )

    # --- Step 4: Zone check --------------------------------------------------
    # Use PDH / PDL as the swing range reference
    pdh = session_levels.get("pdh")
    pdl = session_levels.get("pdl")
    if pdh and pdl:
        in_zone = is_fvg_in_correct_zone(fvg, float(pdh), float(pdl), direction)
        if not in_zone:
            zone_name = "discount" if direction == "bullish" else "premium"
            return EntrySetup(
                valid=False,
                sweep_level=sweep.level,
                reason=f"FVG not in {zone_name} zone (mid={fvg.midpoint:.2f}, EQ={((float(pdh)+float(pdl))/2):.2f})",
            )

    # --- Step 5: Calculate entry / SL / TP -----------------------------------
    buf = SL_BUFFER * PIP_VALUE

    if direction == "bullish":
        entry = fvg.midpoint
        sl    = fvg.bottom - buf
        # TP: PDH first; fallback to 3x risk
        risk  = entry - sl
        if tp_target and float(tp_target) > entry:
            tp = float(tp_target)
        else:
            tp = entry + 3.0 * risk
    else:
        entry = fvg.midpoint
        sl    = fvg.top + buf
        risk  = sl - entry
        if tp_target and float(tp_target) < entry:
            tp = float(tp_target)
        else:
            tp = entry - 3.0 * risk

    reward = abs(tp - entry)
    rr     = round(reward / risk, 2) if risk > 0 else 0.0

    if rr < MIN_RR:
        return EntrySetup(
            valid=False,
            sweep_level=sweep.level,
            reason=f"RR too low: 1:{rr:.1f} (min 1:{MIN_RR})",
        )

    reason = (
        f"{sweep_label} swept, "
        f"{'bullish' if direction == 'bullish' else 'bearish'} MSS on M1, "
        f"FVG in {'discount' if direction == 'bullish' else 'premium'} zone"
    )

    return EntrySetup(
        valid=True,
        direction=direction.upper(),
        entry=round(entry, 2),
        sl=round(sl, 2),
        tp=round(tp, 2),
        rr=rr,
        fvg_zone=fvg,
        sweep_level=sweep.level,
        reason=reason,
    )


# --- Pretty printer -----------------------------------------------------------

def print_setup(setup: EntrySetup) -> None:
    print()
    print("=" * 55)
    if setup.valid:
        print(f"  ENTRY SETUP  [{setup.direction}]")
        print("=" * 55)
        print(f"  Entry        : {setup.entry:.2f}")
        print(f"  Stop Loss    : {setup.sl:.2f}")
        print(f"  Take Profit  : {setup.tp:.2f}")
        print(f"  R:R          : 1:{setup.rr:.1f}")
        if setup.fvg_zone:
            print(f"  FVG Zone     : {setup.fvg_zone.bottom:.2f} - {setup.fvg_zone.top:.2f}")
            print(f"  FVG Mid      : {setup.fvg_zone.midpoint:.2f}")
        print(f"  Sweep Level  : {setup.sweep_level:.2f}")
        print(f"  Reason       : {setup.reason}")
    else:
        print("  No valid setup found")
        print("=" * 55)
        print(f"  Reason : {setup.reason}")
    print("=" * 55)


# --- Entry Point --------------------------------------------------------------

if __name__ == "__main__":
    from mt5_connector import connect, disconnect
    from market_structure import get_previous_day_levels, get_asian_session_levels

    print("Connecting to MT5...")
    if not connect():
        raise SystemExit("Could not connect to MT5.")

    try:
        # Build session_levels from real data
        day   = get_previous_day_levels(SYMBOL)
        asian = get_asian_session_levels(SYMBOL)

        session_levels = {
            "pdh": day.pdh   if day   else None,
            "pdl": day.pdl   if day   else None,
            "ash": asian.ash if asian else None,
            "asl": asian.asl if asian else None,
        }

        print(f"\nSession levels:")
        print(f"  PDH = {session_levels['pdh']}  PDL = {session_levels['pdl']}")
        print(f"  ASH = {session_levels['ash']}  ASL = {session_levels['asl']}")

        # Fetch fresh M5 candles for a quick FVG scan preview
        df_m5 = fetch_candles(SYMBOL, timeframe=mt5.TIMEFRAME_M5, count=M5_LOOKBACK)
        df_m1 = fetch_candles(SYMBOL, timeframe=mt5.TIMEFRAME_M1, count=M1_LOOKBACK)

        current_price = df_m5["close"].iloc[-1]
        print(f"\nCurrent price : {current_price:.2f}")

        # --- Scan for FVGs in both directions --------------------------------
        print("\n--- M5 FVG Scan ---")
        bull_fvg = detect_fvg(df_m5, "bullish")
        bear_fvg = detect_fvg(df_m5, "bearish")

        if bull_fvg:
            print(f"  Latest Bullish FVG: {bull_fvg}  filled={bull_fvg.filled}")
        else:
            print("  No bullish FVG found in last 100 M5 candles")

        if bear_fvg:
            print(f"  Latest Bearish FVG: {bear_fvg}  filled={bear_fvg.filled}")
        else:
            print("  No bearish FVG found in last 100 M5 candles")

        # --- Sweep check on known levels ------------------------------------
        print("\n--- Liquidity Sweep Check (M5) ---")
        for label, level in [
            ("PDH", session_levels["pdh"]),
            ("PDL", session_levels["pdl"]),
            ("ASH", session_levels["ash"]),
            ("ASL", session_levels["asl"]),
        ]:
            if level is None:
                print(f"  {label}: N/A")
                continue
            s_above = detect_liquidity_sweep(df_m5, level, "above")
            s_below = detect_liquidity_sweep(df_m5, level, "below")
            above_str = f"sweep above @ {s_above.sweep_price:.2f}" if s_above.found else "no sweep above"
            below_str = f"sweep below @ {s_below.sweep_price:.2f}" if s_below.found else "no sweep below"
            print(f"  {label} ({level:.2f}): {above_str}  |  {below_str}")

        # --- Full entry setup -----------------------------------------------
        print("\n--- Full Entry Setup Check ---")
        for d in ("bullish", "bearish"):
            print(f"\n  Testing {d.upper()} setup...")
            setup = get_entry_setup(SYMBOL, d, session_levels)
            print_setup(setup)

    finally:
        disconnect()
