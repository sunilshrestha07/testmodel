"""
market_structure.py
-------------------
Reads H4 candles from MT5 and determines market structure.

Provides:
  get_swing_highs_lows()      -  pivot highs and lows from a candle DataFrame
  get_h4_bias()               -  BULLISH / BEARISH / RANGING for any symbol
  get_previous_day_levels()   -  PDH and PDL from MT5 D1 data
  get_asian_session_levels()  -  ASH and ASL from today's H1 candles (05:45-12:45 NPT)
  check_dxy_divergence()      -  ALIGNED or DIVERGENCE between XAUUSD and DXY
  get_full_structure_report() -  prints one-line summary of all the above
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import pandas as pd

from config import SYMBOL, DXY_SYMBOL, LOG_LEVEL
from mt5_connector import connect, disconnect, fetch_candles

log = logging.getLogger(__name__)

# --- Nepal Time constant (UTC+5:45) ------------------------------------------
NPT = timezone(timedelta(hours=5, minutes=45))

# Asian session boundaries in NPT
ASIAN_START_H, ASIAN_START_M = 5, 45
ASIAN_END_H,   ASIAN_END_M   = 12, 45


# --- Data classes -------------------------------------------------------------

@dataclass
class SwingPoint:
    time:  pd.Timestamp
    price: float
    kind:  str          # "HIGH" or "LOW"

    def __repr__(self) -> str:
        return f"SwingPoint({self.kind} @ {self.price:.2f}  [{self.time}])"


@dataclass
class DayLevels:
    pdh: float          # Previous Day High
    pdl: float          # Previous Day Low
    date: str           # The date those levels came from (YYYY-MM-DD)


@dataclass
class AsianLevels:
    ash: float          # Asian Session High
    asl: float          # Asian Session Low


# --- 1. Swing Highs / Lows ----------------------------------------------------

def get_swing_highs_lows(
    candles: pd.DataFrame,
    lookback: int = 2,
) -> list[SwingPoint]:
    """
    Detect swing highs and lows using a simple pivot algorithm.

    A swing HIGH at index i requires:
        candle[i].high > candle[i-1].high, candle[i-2].high (left side)
        candle[i].high > candle[i+1].high, candle[i+2].high (right side)

    A swing LOW at index i requires the mirror condition on .low.

    `lookback` controls how many candles on EACH side must be lower/higher.
    Default is 2 (matching the spec).
    """
    if candles.empty or len(candles) < 2 * lookback + 1:
        log.warning("Not enough candles for swing detection (need %d, got %d)",
                    2 * lookback + 1, len(candles))
        return []

    highs  = candles["high"].values
    lows   = candles["low"].values
    times  = candles.index
    n      = len(candles)
    points: list[SwingPoint] = []

    for i in range(lookback, n - lookback):
        # -- Swing High --
        left_ok  = all(highs[i] > highs[i - k] for k in range(1, lookback + 1))
        right_ok = all(highs[i] > highs[i + k] for k in range(1, lookback + 1))
        if left_ok and right_ok:
            points.append(SwingPoint(time=times[i], price=highs[i], kind="HIGH"))

        # -- Swing Low --
        left_ok  = all(lows[i] < lows[i - k] for k in range(1, lookback + 1))
        right_ok = all(lows[i] < lows[i + k] for k in range(1, lookback + 1))
        if left_ok and right_ok:
            points.append(SwingPoint(time=times[i], price=lows[i], kind="LOW"))

    # Sort chronologically
    points.sort(key=lambda p: p.time)
    return points


# --- 2. H4 Bias ---------------------------------------------------------------

def _classify_sequence(values: list[float]) -> str:
    """
    Given an ordered list of prices, return trend direction.
    Requires at least 2 successive higher or lower values to confirm.
    """
    if len(values) < 2:
        return "RANGING"

    higher = sum(1 for a, b in zip(values, values[1:]) if b > a)
    lower  = sum(1 for a, b in zip(values, values[1:]) if b < a)
    total  = len(values) - 1

    # Need ≥60 % of pivots moving in the same direction
    if higher / total >= 0.6:
        return "BULLISH"
    if lower / total >= 0.6:
        return "BEARISH"
    return "RANGING"


def get_h4_bias(symbol: str, count: int = 50) -> str:
    """
    Fetch `count` H4 candles for `symbol` and return the structural bias:
      "BULLISH"   -  higher highs + higher lows
      "BEARISH"   -  lower highs  + lower lows
      "RANGING"   -  no clear direction
    """
    df = fetch_candles(symbol, timeframe=mt5.TIMEFRAME_H4, count=count)
    if df.empty:
        log.warning("No H4 candles for %s  -  defaulting to RANGING", symbol)
        return "RANGING"

    swings = get_swing_highs_lows(df, lookback=2)
    if not swings:
        log.warning("No swing points found for %s  -  defaulting to RANGING", symbol)
        return "RANGING"

    swing_highs = [s.price for s in swings if s.kind == "HIGH"]
    swing_lows  = [s.price for s in swings if s.kind == "LOW"]

    hh_bias = _classify_sequence(swing_highs)
    ll_bias = _classify_sequence(swing_lows)

    # Both highs and lows must agree for a clean trend
    if hh_bias == "BULLISH" and ll_bias == "BULLISH":
        return "BULLISH"
    if hh_bias == "BEARISH" and ll_bias == "BEARISH":
        return "BEARISH"
    return "RANGING"


# --- 3. Previous Day Levels ---------------------------------------------------

def get_previous_day_levels(symbol: str = SYMBOL) -> DayLevels | None:
    """
    Fetch the last 2 completed D1 candles and return the PDH/PDL
    from the most recently completed day (index -2; index -1 is today).
    MT5 returns UTC-based daily candles.
    """
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 3)
    if rates is None or len(rates) < 2:
        log.error("Failed to fetch D1 candles for %s: %s", symbol, mt5.last_error())
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # Index -2 is the last fully closed day
    prev = df.iloc[-2]
    return DayLevels(
        pdh=float(prev["high"]),
        pdl=float(prev["low"]),
        date=str(prev["time"].date()),
    )


# --- 4. Asian Session Levels --------------------------------------------------

def get_asian_session_levels(symbol: str = SYMBOL) -> AsianLevels | None:
    """
    Derive ASH / ASL from today's H1 candles that fall inside the
    Asian session window (05:45-12:45 NPT = 00:00-07:00 UTC).

    Uses H1 candles for finer granularity than H4.
    """
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 48)
    if rates is None or len(rates) == 0:
        log.error("Failed to fetch H1 candles for Asian levels: %s", mt5.last_error())
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time_npt"] = df["time"].dt.tz_convert(NPT)

    now_npt  = datetime.now(tz=NPT)
    today    = now_npt.date()

    # Asian session start and end as NPT-aware datetimes for today
    asian_start = datetime(today.year, today.month, today.day,
                           ASIAN_START_H, ASIAN_START_M, tzinfo=NPT)
    asian_end   = datetime(today.year, today.month, today.day,
                           ASIAN_END_H, ASIAN_END_M, tzinfo=NPT)

    mask = (df["time_npt"] >= asian_start) & (df["time_npt"] < asian_end)
    session_candles = df[mask]

    if session_candles.empty:
        log.warning("No H1 candles found inside today's Asian session window.")
        return None

    return AsianLevels(
        ash=float(session_candles["high"].max()),
        asl=float(session_candles["low"].min()),
    )


# --- 5. DXY Divergence --------------------------------------------------------

def check_dxy_divergence() -> tuple[str, str, str]:
    """
    Compare XAUUSD H4 bias vs DXY H4 bias.

    Gold and DXY normally move inversely:
      Gold BULLISH + DXY BEARISH  → ALIGNED   (healthy bull setup)
      Gold BEARISH + DXY BULLISH  → ALIGNED   (healthy bear setup)
      Gold BULLISH + DXY BULLISH  → DIVERGENCE WARNING
      Gold BEARISH + DXY BEARISH  → DIVERGENCE WARNING
      Either RANGING              → ALIGNED   (no strong bias to conflict)

    Returns (gold_bias, dxy_bias, "ALIGNED" | "DIVERGENCE")
    """
    gold_bias = get_h4_bias(SYMBOL)
    dxy_bias  = get_h4_bias(DXY_SYMBOL)

    if gold_bias == "RANGING" or dxy_bias == "RANGING":
        divergence = "ALIGNED"
    elif gold_bias == dxy_bias:
        divergence = "DIVERGENCE"
    else:
        divergence = "ALIGNED"

    return gold_bias, dxy_bias, divergence


# --- 6. Full Structure Report -------------------------------------------------

def get_full_structure_report() -> dict:
    """
    Collects all structure data and prints a one-line summary.
    Also returns a dict for programmatic use.
    """
    gold_bias, dxy_bias, divergence = check_dxy_divergence()

    day_levels   = get_previous_day_levels(SYMBOL)
    asian_levels = get_asian_session_levels(SYMBOL)

    pdh = f"{day_levels.pdh:.2f}"   if day_levels   else "N/A"
    pdl = f"{day_levels.pdl:.2f}"   if day_levels   else "N/A"
    ash = f"{asian_levels.ash:.2f}" if asian_levels else "N/A"
    asl = f"{asian_levels.asl:.2f}" if asian_levels else "N/A"

    report_line = (
        f"GOLD H4: {gold_bias:<8} | "
        f"DXY H4: {dxy_bias:<8} | "
        f"Divergence: {divergence:<11} | "
        f"PDH: {pdh:>9} | PDL: {pdl:>9} | "
        f"ASH: {ash:>9} | ASL: {asl:>9}"
    )
    print(report_line)

    return {
        "gold_bias":   gold_bias,
        "dxy_bias":    dxy_bias,
        "divergence":  divergence,
        "pdh":         day_levels.pdh   if day_levels   else None,
        "pdl":         day_levels.pdl   if day_levels   else None,
        "pdl_date":    day_levels.date  if day_levels   else None,
        "ash":         asian_levels.ash if asian_levels else None,
        "asl":         asian_levels.asl if asian_levels else None,
    }


# --- Entry Point --------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 75)
    print("  Market Structure Report  -  XAUUSD Bot")
    print("=" * 75)

    if not connect():
        raise SystemExit("Could not connect to MT5.")

    try:
        report = get_full_structure_report()

        print()
        # -- Swing points preview ----------------------------------------------
        print("  Last 5 XAUUSD H4 swing points:")
        df = fetch_candles(SYMBOL, timeframe=mt5.TIMEFRAME_H4, count=50)
        swings = get_swing_highs_lows(df)
        for s in swings[-5:]:
            print(f"    {s}")

    finally:
        disconnect()
