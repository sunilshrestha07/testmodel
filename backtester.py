# -*- coding: utf-8 -*-
"""
backtester.py
-------------
Replays historical M5 XAUUSD candles through the full Silver Bullet
strategy logic and produces a trade-by-trade CSV + Obsidian summary.

Data sources (all from MT5 history):
  M5   -- entry signals, FVG detection, trade execution simulation
  H1   -- Asian session high/low computation
  H4   -- structural bias (swing highs/lows)
  D1   -- previous-day high/low

Strategy replicated:
  - Trading windows in NPT (Asian Sweep, NY AM, NY PM Silver Bullet)
  - Liquidity sweep of PDH/PDL or ASH/ASL
  - Market Structure Shift (MSS)
  - Fair Value Gap in premium/discount zone
  - Breakeven at 50% of TP distance
  - 1% risk per trade, lot capped at 0.10
"""

from __future__ import annotations

import csv
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from config import SYMBOL, DXY_SYMBOL, MAGIC_NUMBER, LOG_LEVEL
from mt5_connector import connect, disconnect
from fvg_detector import (
    detect_fvg, detect_liquidity_sweep, detect_market_structure_shift,
    is_fvg_in_correct_zone, diagnose_sweep_miss,
    detect_order_block, detect_mss_candle_entry, FVGZone,
)
from market_structure import get_swing_highs_lows
from risk_manager import calculate_lot_size

log = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- Constants ---------------------------------------------------------------
NPT          = timezone(timedelta(hours=5, minutes=45))
UTC          = timezone.utc
ACCOUNT_SIZE = 100_000.0      # starting balance for simulation
RISK_PCT     = 1.0            # % risk per trade
BE_TRIGGER   = 0.50           # move SL to entry at 50% of way to TP
MIN_RR         = 1.5
MAX_RR_CAP     = 8.0   # cap TP at 8R so it's reachable within a session
SL_BUFFER      = 3.0   # pip buffer beyond FVG edge for SL  (was 2.0 -- raised to reduce false SL hits)
MIN_WICK_PIPS  = 1.0   # min wick beyond sweep level        (was 1.5 -- lowered; diagnostics showed real sweeps missed at 1.0-1.5 pip)
ZONE_FILTER    = False  # require FVG in premium/discount zone (off -- too restrictive in trends)
PIP_VALUE      = 1.0   # $1.00 per pip for XAUUSD (in risk calc)
MAX_TRADES_PER_DAY = 4  # 4 windows can now fire on the same day

# --- Trading windows in NPT (h_start, m_start, h_end, m_end) ----------------
WINDOWS = {
    "ASIAN_RANGE":                ((5, 45),  (12, 45)),
    "ASIAN_SWEEP_WINDOW":         ((12, 45), (13, 45)),
    "LONDON_OPEN_SILVER_BULLET":  ((13, 45), (14, 45)),
    "NY_AM_SILVER_BULLET":        ((19, 45), (20, 45)),
    "NY_PM_SILVER_BULLET":        ((23, 45), (0, 45)),
}
TRADE_WINDOWS = {
    "ASIAN_SWEEP_WINDOW",
    "LONDON_OPEN_SILVER_BULLET",
    "NY_AM_SILVER_BULLET",
    "NY_PM_SILVER_BULLET",
}

# UTC open times for each trade window.
# NPT = UTC + 5:45 → ASIAN_SWEEP 12:45 NPT = 07:00 UTC etc.
_WIN_OPEN_UTC: dict[str, tuple[int, int]] = {
    "ASIAN_SWEEP_WINDOW":         (7,  0),   # 12:45 NPT
    "LONDON_OPEN_SILVER_BULLET":  (8,  0),   # 13:45 NPT
    "NY_AM_SILVER_BULLET":        (14, 0),   # 19:45 NPT
    "NY_PM_SILVER_BULLET":        (18, 0),   # 23:45 NPT
}

OUTPUT_DIR = Path(__file__).parent / "backtest_results"

# UTC hour at which VWAP resets for each trading window's parent session.
# Asian session  00:00 UTC  |  London 07:00 UTC  |  NY 12:00 UTC
_WIN_SESSION_VWAP_START_UTC: dict[str, int] = {
    "ASIAN_SWEEP_WINDOW":        0,   # Asian session VWAP
    "LONDON_OPEN_SILVER_BULLET": 7,   # London session VWAP
    "NY_AM_SILVER_BULLET":       12,  # NY session VWAP
    "NY_PM_SILVER_BULLET":       12,  # same NY session
}


# --- Data classes ------------------------------------------------------------

@dataclass
class BacktestTrade:
    trade_id:    int
    date:        str
    direction:   str
    window:      str
    entry:       float
    sl:          float
    tp:          float
    lot:         float
    exit_price:  float   = 0.0
    exit_reason: str     = ""    # "TP" | "SL" | "EOD" | "EOW"
    pnl:         float   = 0.0
    rr_achieved: float   = 0.0
    duration_m:  int     = 0     # minutes held
    entry_time:    str              = ""
    exit_time:     str              = ""
    sweep_level:   float            = 0.0
    fvg_top:       float            = 0.0
    fvg_bottom:    float            = 0.0
    h4_bias:       str              = ""
    entry_type:    str              = "FVG_ENTRY"   # "FVG_ENTRY" | "OB_ENTRY" | "MSS_ENTRY"
    be_triggered:  bool             = False
    vwap:          float            = 0.0    # session VWAP at trade entry
    vwap_aligned:  bool             = True   # True when entry is between VWAP and TP
    entry_ts_utc:  object           = None   # pd.Timestamp -- not in CSV output


@dataclass
class DayStats:
    date:       str
    trades:     int  = 0
    wins:       int  = 0
    losses:     int  = 0
    bes:        int  = 0
    pnl:        float = 0.0


# --- Session time helpers ----------------------------------------------------

def _in_window(t_npt: time, start: tuple, end: tuple) -> bool:
    s = time(start[0], start[1])
    e = time(end[0],   end[1])
    if s < e:
        return s <= t_npt < e
    return t_npt >= s or t_npt < e


def _get_window(dt_npt: datetime) -> str:
    t = dt_npt.time()
    for name, (start, end) in WINDOWS.items():
        if _in_window(t, start, end):
            return name
    return "NONE"


# --- H4 bias from historical candles -----------------------------------------

def _h4_bias_at(h4_df: pd.DataFrame, as_of: pd.Timestamp) -> str:
    """Compute H4 bias using only candles available up to `as_of`."""
    sub = h4_df[h4_df.index < as_of].tail(50)
    if len(sub) < 10:
        return "RANGING"

    # Wrap in a dummy DataFrame to reuse get_swing_highs_lows
    swings = get_swing_highs_lows(sub, lookback=2)
    if not swings:
        return "RANGING"

    sh_prices = [s.price for s in swings if s.kind == "HIGH"]
    sl_prices = [s.price for s in swings if s.kind == "LOW"]

    def _trend(vals):
        if len(vals) < 2:
            return "RANGING"
        higher = sum(1 for a, b in zip(vals, vals[1:]) if b > a)
        lower  = sum(1 for a, b in zip(vals, vals[1:]) if b < a)
        total  = len(vals) - 1
        if higher / total >= 0.6: return "BULLISH"
        if lower  / total >= 0.6: return "BEARISH"
        return "RANGING"

    hh = _trend(sh_prices)
    hl = _trend(sl_prices)
    if hh == "BULLISH" and hl == "BULLISH": return "BULLISH"
    if hh == "BEARISH" and hl == "BEARISH": return "BEARISH"
    return "RANGING"


def _h1_bias_at(h1_df: pd.DataFrame, as_of: pd.Timestamp) -> str:
    """Compute H1 structural bias using only candles available up to `as_of`.
    Used as fallback when H4 is RANGING."""
    sub = h1_df[h1_df.index < as_of].tail(50)
    if len(sub) < 10:
        return "RANGING"

    swings = get_swing_highs_lows(sub, lookback=2)
    if not swings:
        return "RANGING"

    sh_prices = [s.price for s in swings if s.kind == "HIGH"]
    sl_prices = [s.price for s in swings if s.kind == "LOW"]

    def _trend(vals):
        if len(vals) < 2:
            return "RANGING"
        higher = sum(1 for a, b in zip(vals, vals[1:]) if b > a)
        lower  = sum(1 for a, b in zip(vals, vals[1:]) if b < a)
        total  = len(vals) - 1
        if higher / total >= 0.6: return "BULLISH"
        if lower  / total >= 0.6: return "BEARISH"
        return "RANGING"

    hh = _trend(sh_prices)
    hl = _trend(sl_prices)
    if hh == "BULLISH" and hl == "BULLISH": return "BULLISH"
    if hh == "BEARISH" and hl == "BEARISH": return "BEARISH"
    return "RANGING"


def _h4_sma_at(h4_df: pd.DataFrame, as_of: pd.Timestamp, period: int) -> float:
    """Compute H4 SMA of `period` closes available strictly before `as_of`.
    Returns 0.0 when there are fewer candles than the period (insufficient history)."""
    closes = h4_df[h4_df.index < as_of]["close"].tail(period)
    if len(closes) < period:
        return 0.0
    return float(closes.mean())


def _session_vwap_at(
    m5_df: pd.DataFrame,
    as_of: pd.Timestamp,
    session_start: pd.Timestamp,
) -> float:
    """
    Calculate session VWAP from session_start up to (not including) as_of.

    Typical price = (H + L + C) / 3.
    Weights by tick_volume; falls back to equal-weight when volume is zero.
    Returns 0.0 when fewer than 2 M5 candles are available in the session.
    """
    mask = (m5_df.index >= session_start) & (m5_df.index < as_of)
    sub  = m5_df[mask]
    if len(sub) < 2:
        return 0.0
    typical   = (sub["high"] + sub["low"] + sub["close"]) / 3.0
    vol_col   = "tick_volume" if "tick_volume" in sub.columns else None
    total_vol = float(sub[vol_col].sum()) if vol_col else 0.0
    if total_vol > 0:
        vwap = float((typical * sub[vol_col]).sum() / total_vol)
    else:
        vwap = float(typical.mean())
    return round(vwap, 2)


# --- Session level computers -------------------------------------------------

def _prev_day_levels(d1_df: pd.DataFrame, as_of_date: date) -> tuple[float, float]:
    """Return (PDH, PDL) for the day before as_of_date."""
    prev = as_of_date - timedelta(days=1)
    # Walk backwards up to 4 days to skip weekends
    for _ in range(4):
        key = pd.Timestamp(prev, tz=UTC)
        row = d1_df[d1_df.index.date == prev]
        if not row.empty:
            return float(row["high"].iloc[0]), float(row["low"].iloc[0])
        prev -= timedelta(days=1)
    return 0.0, 0.0


def _asian_levels(h1_df: pd.DataFrame, as_of_date: date) -> tuple[float, float]:
    """
    Return (ASH, ASL) for the Asian session of as_of_date.
    Asian session NPT = 05:45-12:45, which is UTC 00:00-07:00.
    """
    day_start = pd.Timestamp(as_of_date, tz=UTC)
    day_end   = day_start + timedelta(hours=7)
    mask = (h1_df.index >= day_start) & (h1_df.index < day_end)
    sub  = h1_df[mask]
    if sub.empty:
        return 0.0, 0.0
    return float(sub["high"].max()), float(sub["low"].min())


def _weekly_levels(d1_df: pd.DataFrame, as_of_date: date) -> tuple[float, float]:
    """
    Return (weekly_high, weekly_low) = the highest high and lowest low of the
    completed daily candles in the current week (Monday through yesterday).

    Weekly liquidity levels are primary institutional targets.  They are passed
    to the entry scanner as additional sweep candidates so that NY and London
    sessions can trade a sweep of the week's range, not only PDH/PDL or ASH/ASL.
    Returns (0.0, 0.0) when no completed weekly candles are available yet
    (e.g. Monday before the first close of the week).
    """
    # Monday of the current week
    days_since_monday = as_of_date.weekday()   # Mon=0 … Sun=6
    week_start = as_of_date - timedelta(days=days_since_monday)
    week_start_ts = pd.Timestamp(week_start, tz=UTC)
    today_ts      = pd.Timestamp(as_of_date, tz=UTC)

    # Candles in [week_start, today) — completed days only
    mask = (d1_df.index >= week_start_ts) & (d1_df.index < today_ts)
    sub  = d1_df[mask]
    if sub.empty:
        return 0.0, 0.0
    return float(sub["high"].max()), float(sub["low"].min())


# --- Trade simulation --------------------------------------------------------

def _simulate_trade(
    m5_df:      pd.DataFrame,
    entry_idx:  int,
    direction:  str,
    entry:      float,
    sl:         float,
    tp:         float,
    lot:        float,
    pip_val:    float = 10.0,   # $10 per pip per lot for XAUUSD
) -> tuple[float, float, str, int]:
    """
    Walk forward from entry_idx and return (exit_price, pnl, reason, duration_m).
    Simulates breakeven at BE_TRIGGER.
    Closes at end of day if neither SL nor TP is hit.
    """
    effective_sl = sl
    be_set       = False
    pip_size     = 0.10   # $0.10 per pip

    candles = m5_df.iloc[entry_idx + 1:]
    entry_dt  = m5_df.index[entry_idx]
    entry_date = entry_dt.date()

    for i, (ts, row) in enumerate(candles.iterrows()):
        high  = row["high"]
        low   = row["low"]
        close = row["close"]

        # End of day (midnight UTC ≈ 18:15 NPT, close open trades)
        if ts.date() != entry_date:
            exit_p = close
            pnl    = _calc_pnl(direction, entry, exit_p, lot, pip_size, pip_val)
            return exit_p, pnl, "EOD", i * 5

        # Breakeven trigger
        if not be_set:
            tp_dist  = abs(tp - entry)
            progress = abs(close - entry) / tp_dist if tp_dist > 0 else 0
            if progress >= BE_TRIGGER:
                effective_sl = entry
                be_set = True

        if direction == "LONG":
            if low <= effective_sl:
                exit_p = effective_sl
                pnl    = _calc_pnl("LONG", entry, exit_p, lot, pip_size, pip_val)
                reason = "BE" if be_set and abs(exit_p - entry) < 0.10 else "SL"
                return exit_p, pnl, reason, i * 5
            if high >= tp:
                pnl = _calc_pnl("LONG", entry, tp, lot, pip_size, pip_val)
                return tp, pnl, "TP", i * 5
        else:
            if high >= effective_sl:
                exit_p = effective_sl
                pnl    = _calc_pnl("SHORT", entry, exit_p, lot, pip_size, pip_val)
                reason = "BE" if be_set and abs(exit_p - entry) < 0.10 else "SL"
                return exit_p, pnl, reason, i * 5
            if low <= tp:
                pnl = _calc_pnl("SHORT", entry, tp, lot, pip_size, pip_val)
                return tp, pnl, "TP", i * 5

    # Ran out of data
    exit_p = candles["close"].iloc[-1] if len(candles) else entry
    pnl    = _calc_pnl(direction, entry, exit_p, lot, pip_size, pip_val)
    return exit_p, pnl, "EOW", len(candles) * 5


def _calc_pnl(direction, entry, exit_p, lot, pip_size, pip_val):
    pips = (exit_p - entry) / pip_size if direction == "LONG" else (entry - exit_p) / pip_size
    return round(pips * pip_val * lot, 2)


# --- Entry logic -------------------------------------------------------------

# Ordered filter stages (higher index = further into the pipeline)
STAGE_ORDER = [
    "IN_TRADE",                    #  0 - couldn't scan: trade already open
    "MAX_TRADES",                  #  1 - couldn't scan: daily trade limit hit
    "NO_PDH_PDL",                  #  2 - missing previous-day levels
    "RANGING",                     #  3 - H4 bias not directional
    "PREV_SESSION_LOSS",           #  4 - London Open skipped: Asian Sweep already lost today
    "BULL_REGIME_SHORT_BLOCKED",   #  5 - price > 200 SMA → only LONGs allowed
    "SHORT_NO_BULL_DXY",           #  6 - SHORT skipped: DXY H4 not clearly BULLISH
    "SHORT_ABOVE_50SMA",           #  7 - SHORT skipped: price is above 50-period H4 SMA
    "LONG_BELOW_50SMA",            #  8 - LONG skipped: price is below 50-period H4 SMA
    "NO_SWEEP",                    #  9 - no liquidity sweep found
    "NO_MSS",                      # 10 - sweep found, but no MSS
    "NO_FVG",                      # 11 - MSS found, but no valid FVG
    "WRONG_ZONE",                  # 12 - FVG exists but in wrong premium/discount zone
    "LOW_RR",                      # 13 - zone OK but reward:risk below minimum
    "TRADE",                       # 14 - setup complete, trade taken
]
_STAGE_RANK = {s: i for i, s in enumerate(STAGE_ORDER)}


def _scan_for_entry_verbose(
    m5_df:       pd.DataFrame,
    cur_idx:     int,
    lookback:    int,
    direction:   str,
    pdh: float, pdl: float,
    ash: float, asl: float,
    wh:  float = 0.0,   # weekly high (additional sweep candidate)
    wl:  float = 0.0,   # weekly low  (additional sweep candidate)
    window_open: Optional[pd.Timestamp] = None,  # kept for API compat; unused
) -> tuple[Optional[tuple], str]:
    """
    Same logic as _scan_for_entry but also returns a reason string from
    STAGE_ORDER describing how far the setup got before failing (or "TRADE"
    on success).  Used both for trade entry and for session-funnel accounting.
    """
    sub = m5_df.iloc[max(0, cur_idx - lookback): cur_idx]
    if len(sub) < 10:
        return None, "NO_SWEEP"

    sweep_dir  = "below" if direction == "bullish" else "above"
    candidates = []
    if direction == "bullish":
        if asl > 0:  candidates.append(("ASL", asl))
        if pdl > 0:  candidates.append(("PDL", pdl))
        if wl  > 0:  candidates.append(("WL",  wl))
    else:
        if ash > 0:  candidates.append(("ASH", ash))
        if pdh > 0:  candidates.append(("PDH", pdh))
        if wh  > 0:  candidates.append(("WH",  wh))

    sweep_res  = None
    sweep_lvl  = 0.0
    sweep_time: Optional[pd.Timestamp] = None
    for _label, level in candidates:
        r = detect_liquidity_sweep(sub, level, sweep_dir, min_wick_pips=MIN_WICK_PIPS)
        if r.found:
            sweep_res  = r
            sweep_lvl  = level
            sweep_time = sub.index[r.candle_index]
            break
    if sweep_res is None:
        if log.isEnabledFor(logging.DEBUG):
            for _label, level in candidates:
                diagnose_sweep_miss(sub, level, sweep_dir, MIN_WICK_PIPS, label=_label)
        return None, "NO_SWEEP"

    mss = detect_market_structure_shift(sub, direction, after_idx=sweep_res.candle_index)
    if not mss.found:
        return None, "NO_MSS"

    fvg = detect_fvg(sub, direction, start_idx=sweep_res.candle_index)

    # -- When no clean FVG: try Method A (Order Block) then Method B (MSS candle) --
    if fvg is None or fvg.filled:
        ob = detect_order_block(sub, direction,
                                sweep_idx=sweep_res.candle_index,
                                mss_idx=mss.candle_index)
        fallback = ob if ob.found else detect_mss_candle_entry(sub, mss.candle_index)

        if fallback.found:
            buf = SL_BUFFER * PIP_VALUE
            if direction == "bullish":
                entry    = fallback.midpoint
                sl       = fallback.candle_low - buf
                risk     = entry - sl
                tp_ideal = pdh if pdh > entry else entry + 3 * risk
                tp       = min(tp_ideal, entry + MAX_RR_CAP * risk)
                # Pseudo-FVG: stores the OB/MSS zone for fvg_top/bottom fields
                zone = FVGZone(top=fallback.body_top, bottom=fallback.candle_low,
                               midpoint=fallback.midpoint,
                               candle_index=fallback.candle_index,
                               direction=direction, filled=False)
            else:
                entry    = fallback.midpoint
                sl       = fallback.candle_high + buf
                risk     = sl - entry
                tp_ideal = pdl if (pdl > 0 and pdl < entry) else entry - 3 * risk
                tp       = max(tp_ideal, entry - MAX_RR_CAP * risk)
                zone = FVGZone(top=fallback.candle_high, bottom=fallback.body_bottom,
                               midpoint=fallback.midpoint,
                               candle_index=fallback.candle_index,
                               direction=direction, filled=False)

            risk   = abs(entry - sl)
            reward = abs(tp - entry)
            if risk > 0 and reward / risk >= MIN_RR:
                entry_type = "OB_ENTRY" if ob.found else "MSS_ENTRY"
                return (entry, sl, tp, sweep_lvl, zone, entry_type, sweep_time), "TRADE"

        return None, "NO_FVG"

    if ZONE_FILTER and pdh > 0 and pdl > 0:
        if not is_fvg_in_correct_zone(fvg, pdh, pdl, direction):
            return None, "WRONG_ZONE"

    buf = SL_BUFFER * PIP_VALUE
    if direction == "bullish":
        entry    = fvg.midpoint
        sl       = fvg.bottom - buf
        risk     = entry - sl
        tp_ideal = pdh if pdh > entry else entry + 3 * risk
        tp       = min(tp_ideal, entry + MAX_RR_CAP * risk)
    else:
        entry    = fvg.midpoint
        sl       = fvg.top + buf
        risk     = sl - entry
        tp_ideal = pdl if (pdl > 0 and pdl < entry) else entry - 3 * risk
        tp       = max(tp_ideal, entry - MAX_RR_CAP * risk)

    risk   = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 0 or reward / risk < MIN_RR:
        return None, "LOW_RR"

    return (entry, sl, tp, sweep_lvl, fvg, "FVG_ENTRY", sweep_time), "TRADE"


def _scan_for_entry(
    m5_df: pd.DataFrame, cur_idx: int, lookback: int,
    direction: str, pdh: float, pdl: float, ash: float, asl: float,
    wh: float = 0.0, wl: float = 0.0,
    window_open: Optional[pd.Timestamp] = None,
) -> Optional[tuple]:
    """Thin wrapper — returns result only (no reason code)."""
    result, _ = _scan_for_entry_verbose(m5_df, cur_idx, lookback, direction,
                                        pdh, pdl, ash, asl, wh=wh, wl=wl,
                                        window_open=window_open)
    return result


# --- Main backtest -----------------------------------------------------------

def run_backtest(
    months: int = 6,
    start_date: Optional[datetime] = None,
    end_date:   Optional[datetime] = None,
) -> tuple[list[BacktestTrade], pd.DataFrame]:
    """
    Run the full Silver Bullet backtest.

    Parameters
    ----------
    months     : how many months of history to replay (ignored if dates given)
    start_date : explicit start (UTC-aware datetime)
    end_date   : explicit end   (UTC-aware datetime)

    Returns
    -------
    (trades, equity_curve_df)
    """
    if end_date is None:
        end_date = datetime.now(UTC)
    if start_date is None:
        start_date = end_date - timedelta(days=30 * months)

    log.info("Loading historical data  %s  ->  %s",
             start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    # Fetch data
    m5_raw = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M5, start_date, end_date)
    # 30-day lookback so H1 bias is computable from day 1 of the backtest period
    h1_raw = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H1,
                                  start_date - timedelta(days=30), end_date)
    # Extra 60-day lookback so the 200-period H4 SMA is computable from day 1
    h4_raw = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H4,
                                  start_date - timedelta(days=60), end_date)
    d1_raw = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_D1,
                                  start_date - timedelta(days=10), end_date)
    dxy_h4_raw = mt5.copy_rates_range(DXY_SYMBOL, mt5.TIMEFRAME_H4,
                                      start_date - timedelta(days=60), end_date)

    if m5_raw is None or len(m5_raw) == 0:
        raise RuntimeError("No M5 data returned from MT5")

    def _to_df(raw):
        df = pd.DataFrame(raw)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df.set_index("time")

    m5_df     = _to_df(m5_raw)
    h1_df     = _to_df(h1_raw)     if h1_raw     is not None else pd.DataFrame()
    h4_df     = _to_df(h4_raw)     if h4_raw     is not None else pd.DataFrame()
    d1_df     = _to_df(d1_raw)     if d1_raw     is not None else pd.DataFrame()
    dxy_h4_df = _to_df(dxy_h4_raw) if (dxy_h4_raw is not None and len(dxy_h4_raw) > 0) \
                                    else pd.DataFrame()

    log.info("M5: %d candles  H1: %d  H4: %d  D1: %d  DXY_H4: %d",
             len(m5_df), len(h1_df), len(h4_df), len(d1_df), len(dxy_h4_df))
    if dxy_h4_df.empty:
        log.warning("DXY H4 data unavailable (%s) -- SHORT trades will be blocked.", DXY_SYMBOL)

    # --- Replay --------------------------------------------------------------
    trades:        list[BacktestTrade] = []
    trade_counter  = 0
    balance        = ACCOUNT_SIZE
    equity_rows:   list[dict] = []

    # Per-day state
    current_day:       Optional[date] = None
    trades_today       = 0
    day_pdh = day_pdl  = 0.0
    day_ash = day_asl  = 0.0
    day_wh  = day_wl   = 0.0   # weekly high/low (additional sweep candidates)
    day_used_sweep_times: set = set()   # sweep candle timestamps already traded today
    day_window_losses:    set = set()   # windows that closed a trade at SL today
    in_trade           = False
    open_trade:        Optional[BacktestTrade] = None

    M5_LOOKBACK = 100   # candles to scan for patterns

    # Session-funnel accounting.
    # Key: (date_str, window_name)  Value: furthest STAGE_ORDER stage reached
    # We record the MAX stage seen across all candles in each session so that
    # e.g. "sweep found on candle 3, no FVG on any candle" → "NO_FVG" for the
    # whole session (not "NO_SWEEP").
    session_best: dict[tuple, str] = {}

    log.info("Replaying %d M5 candles ...", len(m5_df))

    for idx in range(M5_LOOKBACK, len(m5_df)):
        ts_utc = m5_df.index[idx]
        dt_npt = ts_utc.astimezone(NPT)
        today  = ts_utc.date()

        # Day rollover
        if today != current_day:
            current_day  = today
            trades_today = 0
            day_pdh, day_pdl = _prev_day_levels(d1_df, today)
            day_ash, day_asl = _asian_levels(h1_df, today)
            day_wh,  day_wl  = _weekly_levels(d1_df, today)
            day_used_sweep_times = set()   # reset daily sweep dedup
            day_window_losses    = set()   # reset daily window-loss tracker

            # Record equity at each day boundary
            equity_rows.append({
                "date":    today.isoformat(),
                "balance": round(balance, 2),
            })

        # Session key for funnel tracking (window + calendar date).
        # Use UTC date to avoid midnight NPT crossing bug in NY PM window
        # (23:45-00:45 NPT = 18:00-19:00 UTC -- no UTC midnight crossing).
        _win_now  = _get_window(dt_npt)
        _sess_key = (ts_utc.strftime("%Y-%m-%d"), _win_now)

        def _update_session(stage: str) -> None:
            """Advance the session's best stage if this one is further along."""
            cur = session_best.get(_sess_key)
            if cur is None or _STAGE_RANK[stage] > _STAGE_RANK[cur]:
                session_best[_sess_key] = stage

        # Already in a trade -- check for exit
        if in_trade and open_trade:
            row   = m5_df.iloc[idx]
            high  = row["high"]
            low   = row["low"]
            close = row["close"]
            entry = open_trade.entry
            sl    = open_trade.sl
            tp    = open_trade.tp
            direction = open_trade.direction

            # BE check
            if not open_trade.be_triggered:
                tp_dist  = abs(tp - entry)
                progress = abs(close - entry) / tp_dist if tp_dist > 0 else 0
                if progress >= BE_TRIGGER:
                    open_trade.sl = entry
                    open_trade.be_triggered = True

            exited = False
            if direction == "LONG":
                if low <= open_trade.sl:
                    open_trade.exit_price  = open_trade.sl
                    open_trade.exit_reason = "BE" if open_trade.be_triggered and abs(open_trade.sl - entry) < 0.10 else "SL"
                    exited = True
                elif high >= tp:
                    open_trade.exit_price  = tp
                    open_trade.exit_reason = "TP"
                    exited = True
            else:
                if high >= open_trade.sl:
                    open_trade.exit_price  = open_trade.sl
                    open_trade.exit_reason = "BE" if open_trade.be_triggered and abs(open_trade.sl - entry) < 0.10 else "SL"
                    exited = True
                elif low <= tp:
                    open_trade.exit_price  = tp
                    open_trade.exit_reason = "TP"
                    exited = True

            # End of day -- close at NY session end (22:00 UTC) or date rollover
            if not exited and open_trade.entry_ts_utc is not None:
                eod_utc = (
                    pd.Timestamp(open_trade.entry_ts_utc.date(), tz=UTC)
                    + timedelta(hours=22)
                )
                if ts_utc >= eod_utc:
                    open_trade.exit_price  = close
                    open_trade.exit_reason = "EOD"
                    exited = True

            if exited:
                ep   = open_trade.exit_price
                ent  = open_trade.entry
                lot  = open_trade.lot
                pips = (ep - ent) / 0.10 if direction == "LONG" else (ent - ep) / 0.10
                pnl  = round(pips * 10.0 * lot, 2)
                # Use the ORIGINAL sl stored on the trade (before any BE move)
                orig_sl   = open_trade.fvg_bottom - SL_BUFFER if direction == "LONG" else open_trade.fvg_top + SL_BUFFER
                orig_risk = abs(ent - orig_sl)
                rr_a = round(abs(ep - ent) / orig_risk, 2) if orig_risk > 0 else 0.0

                open_trade.pnl         = pnl
                open_trade.rr_achieved = rr_a
                open_trade.exit_time   = dt_npt.strftime("%Y-%m-%d %H:%M NPT")
                dur = m5_df.index.get_loc(ts_utc) - m5_df.index.get_loc(
                    pd.Timestamp(open_trade.entry_time.replace(" NPT", ""))
                    .tz_localize(NPT).tz_convert(UTC)
                ) * 5
                open_trade.duration_m  = max(5, abs(dur))

                balance += pnl
                # Track SL exits per window so later windows can filter
                if open_trade.exit_reason == "SL":
                    day_window_losses.add(open_trade.window)
                trades.append(open_trade)
                in_trade   = False
                open_trade = None
            continue

        # Gate: trading window (only track sessions we care about)
        window = _win_now
        if window not in TRADE_WINDOWS:
            continue

        # Gate: deduplication -- one trade per (date, window) pair.
        # session_best already records "TRADE" the moment a trade is opened in
        # this window.  Without this check, a fast TP within the same window
        # would allow the same FVG to re-trigger on the next M5 candle.
        if session_best.get(_sess_key) == "TRADE":
            continue

        # Gate: max trades per day
        if trades_today >= MAX_TRADES_PER_DAY:
            _update_session("MAX_TRADES")
            continue

        # Gate: London Open skipped when Asian Sweep already hit SL today.
        # Both windows trade the same H4 bias on the same morning; a loss in
        # Asian Sweep signals the intraday move is against the bias.
        if (window == "LONDON_OPEN_SILVER_BULLET"
                and "ASIAN_SWEEP_WINDOW" in day_window_losses):
            _update_session("PREV_SESSION_LOSS")
            continue

        # Gate: need valid PDH/PDL
        if day_pdh == 0 or day_pdl == 0:
            _update_session("NO_PDH_PDL")
            continue

        # Determine direction: H4 primary, H1 fallback when H4 is ranging
        h4_bias = _h4_bias_at(h4_df, ts_utc) if not h4_df.empty else "RANGING"
        h4_ranging_fallback = False   # True when we're using H1 bias instead of H4

        if h4_bias == "RANGING":
            # H1 fallback: only for the Asian Sweep window.
            # NY AM / NY PM sweep PDH/PDL — these need H4 structural context.
            # Asian Sweep sweeps the Asian session range — H1 bias is more aligned.
            if window != "ASIAN_SWEEP_WINDOW":
                _update_session("RANGING")
                continue
            h1_bias_now = _h1_bias_at(h1_df, ts_utc) if not h1_df.empty else "RANGING"
            if h1_bias_now == "RANGING":
                _update_session("RANGING")
                continue
            direction = "bullish" if h1_bias_now == "BULLISH" else "bearish"
            h4_ranging_fallback = True
            h4_bias_label = f"H1:{h1_bias_now}"
        else:
            direction = "bullish" if h4_bias == "BULLISH" else "bearish"
            h4_bias_label = h4_bias

        # ── Regime & direction quality filters ───────────────────────────────
        # Compute current M5 close and H4 SMAs (no look-ahead: index < ts_utc)
        cur_price  = float(m5_df.iloc[idx]["close"])
        h4_sma_50  = _h4_sma_at(h4_df, ts_utc, 50)
        h4_sma_200 = _h4_sma_at(h4_df, ts_utc, 200)
        dxy_bias   = _h4_bias_at(dxy_h4_df, ts_utc) if not dxy_h4_df.empty else "RANGING"

        if direction == "bearish":
            # Bull regime: price above 200-period H4 SMA → only LONGs permitted
            if h4_sma_200 > 0 and cur_price > h4_sma_200:
                _update_session("BULL_REGIME_SHORT_BLOCKED")
                continue

            # Condition 1: Gold H4 clearly BEARISH (lower highs + lower lows)
            #   For H4-directed trades: guaranteed by _h4_bias_at returning "BEARISH".
            #   For H1 fallback trades: H4 is ranging but H1 shows bearish structure.

            # Condition 2: DXY H4 must be clearly BULLISH
            if dxy_bias != "BULLISH":
                _update_session("SHORT_NO_BULL_DXY")
                continue

            # Condition 3: Price must be BELOW the 50-period H4 SMA
            if h4_sma_50 > 0 and cur_price >= h4_sma_50:
                _update_session("SHORT_ABOVE_50SMA")
                continue

        if direction == "bullish":
            # For H4-directed trades: only buy when above 50 SMA (structural
            # weakness filter).  For H1 fallback (H4 ranging), price naturally
            # oscillates around the 50 SMA — the filter doesn't apply.
            if not h4_ranging_fallback:
                if h4_sma_50 > 0 and cur_price < h4_sma_50:
                    _update_session("LONG_BELOW_50SMA")
                    continue

        # ─────────────────────────────────────────────────────────────────────

        # Mark that this session was blocked by an open trade (lowest priority —
        # only recorded if nothing further was reached on another candle)
        if in_trade:
            _update_session("IN_TRADE")
            continue

        # Compute window open time so the scanner can reject sweeps that
        # happened in an earlier session (prevents NY PM from re-firing a
        # sweep that NY AM already acted on).
        _wh, _wm = _WIN_OPEN_UTC[window]
        _win_open = pd.Timestamp(today, tz=UTC).replace(hour=_wh, minute=_wm)

        # Scan for setup (verbose so we get the failure reason)
        result, reason = _scan_for_entry_verbose(
            m5_df, idx, M5_LOOKBACK, direction,
            day_pdh, day_pdl, day_ash, day_asl,
            wh=day_wh, wl=day_wl,
            window_open=_win_open,
        )
        _update_session(reason)
        if result is None:
            continue

        entry, sl, tp, sweep_lvl, fvg, entry_type, sweep_ts = result

        # Cross-window duplicate guard: if this exact sweep candle was already
        # used for a trade today (e.g. NY AM traded it, NY PM finds it again),
        # skip — same setup, different window.
        if sweep_ts is not None and sweep_ts in day_used_sweep_times:
            _update_session("NO_SWEEP")   # treat as if no sweep found
            continue

        # --- Session VWAP at entry -------------------------------------------
        _vwap_sess_hour = _WIN_SESSION_VWAP_START_UTC.get(window, 0)
        _vwap_sess_start = pd.Timestamp(today, tz=UTC).replace(hour=_vwap_sess_hour, minute=0)
        trade_vwap = _session_vwap_at(m5_df, ts_utc, _vwap_sess_start)

        # VWAP alignment: entry must be between VWAP and TP
        # LONG:  VWAP < entry < TP   (bullish confirmation)
        # SHORT: TP   < entry < VWAP (bearish confirmation)
        if trade_vwap > 0:
            if direction == "bullish":
                trade_vwap_aligned = entry > trade_vwap
            else:
                trade_vwap_aligned = entry < trade_vwap
        else:
            trade_vwap_aligned = True   # VWAP unavailable → don't filter
        # ---------------------------------------------------------------------

        # Lot size
        sl_pips = abs(entry - sl) / 0.10
        lot     = calculate_lot_size(balance, RISK_PCT, sl_pips, lot_modifier=1.0)

        trade_counter += 1
        trades_today  += 1
        if sweep_ts is not None:
            day_used_sweep_times.add(sweep_ts)   # mark this sweep as traded

        t = BacktestTrade(
            trade_id     = trade_counter,
            date         = today.isoformat(),
            direction    = "LONG" if direction == "bullish" else "SHORT",
            window       = window,
            entry        = round(entry, 2),
            sl           = round(sl, 2),
            tp           = round(tp, 2),
            lot          = lot,
            entry_time   = dt_npt.strftime("%Y-%m-%d %H:%M NPT"),
            sweep_level  = round(sweep_lvl, 2),
            fvg_top      = round(fvg.top, 2),
            fvg_bottom   = round(fvg.bottom, 2),
            h4_bias      = h4_bias_label,
            entry_type   = entry_type,
            vwap         = trade_vwap,
            vwap_aligned = trade_vwap_aligned,
            entry_ts_utc = ts_utc,
        )
        in_trade   = True
        open_trade = t
        log.debug("Trade opened #%d  %s  entry=%.2f  sl=%.2f  tp=%.2f",
                  trade_counter, direction.upper(), entry, sl, tp)

    # Close any trade still open at end of data
    if in_trade and open_trade:
        close_p = m5_df["close"].iloc[-1]
        ent     = open_trade.entry
        lot     = open_trade.lot
        direction = open_trade.direction
        pips    = (close_p - ent) / 0.10 if direction == "LONG" else (ent - close_p) / 0.10
        pnl     = round(pips * 10.0 * lot, 2)
        open_trade.exit_price  = round(close_p, 2)
        open_trade.exit_reason = "EOD"
        open_trade.pnl         = pnl
        open_trade.exit_time   = "end of data"
        balance += pnl
        trades.append(open_trade)

    equity_rows.append({"date": end_date.date().isoformat(), "balance": round(balance, 2)})
    equity_df = pd.DataFrame(equity_rows).set_index("date")
    log.info("Backtest complete: %d trades  final balance=%.2f", len(trades), balance)
    return trades, equity_df, session_best


# --- Statistics --------------------------------------------------------------

def _compute_session_report(session_best: dict) -> dict:
    """
    Aggregate the per-session 'furthest stage reached' dict into counts.

    A session is counted in exactly ONE bucket — the deepest stage it reached
    before being rejected (or TRADE if a trade was taken).
    Sessions that never entered a trade window are excluded entirely.
    """
    counts = {s: 0 for s in STAGE_ORDER}
    for stage in session_best.values():
        counts[stage] += 1
    total = sum(counts.values())
    return {
        "total_sessions":               total,
        "skip_in_trade":                counts["IN_TRADE"],
        "skip_max_trades":              counts["MAX_TRADES"],
        "skip_no_pdh_pdl":             counts["NO_PDH_PDL"],
        "skip_ranging":                 counts["RANGING"],
        "skip_prev_session_loss":       counts["PREV_SESSION_LOSS"],
        "skip_bull_regime_short":       counts["BULL_REGIME_SHORT_BLOCKED"],
        "skip_short_no_bull_dxy":       counts["SHORT_NO_BULL_DXY"],
        "skip_short_above_50sma":       counts["SHORT_ABOVE_50SMA"],
        "skip_long_below_50sma":        counts["LONG_BELOW_50SMA"],
        "skip_no_sweep":                counts["NO_SWEEP"],
        "skip_no_mss":                  counts["NO_MSS"],
        "skip_no_fvg":                  counts["NO_FVG"],
        "skip_wrong_zone":              counts["WRONG_ZONE"],
        "skip_low_rr":                  counts["LOW_RR"],
        "sessions_traded":              counts["TRADE"],
        # Conversion rate of scannable sessions → trades
        "scannable":           total - counts["IN_TRADE"] - counts["MAX_TRADES"]
                               - counts["NO_PDH_PDL"] - counts["RANGING"],
    }


def _compute_stats(trades: list[BacktestTrade], equity_df: pd.DataFrame) -> dict:
    if not trades:
        return {"error": "No trades generated"}

    pnls     = [t.pnl for t in trades]
    total    = len(trades)
    # WIN = positive P&L (TP hit OR profitable EOD/EOW close)
    # LOSS = negative P&L (SL hit OR losing EOD/EOW close)
    # BE   = ~zero P&L (breakeven exit OR EOD with near-zero pnl)
    wins     = sum(1 for t in trades if t.pnl > 0.01)
    losses   = sum(1 for t in trades if t.pnl < -0.01)
    bes      = sum(1 for t in trades if abs(t.pnl) <= 0.01)
    win_rate = wins / total * 100
    # Directional WR: wins / (wins + losses), excluding BE exits.
    # BEs protect capital and aren't losses — this metric shows the raw
    # quality of setups that actually resolved as wins or losses.
    directional_wr = round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0.0
    # Exit-reason breakdown (informational)
    tp_count  = sum(1 for t in trades if t.exit_reason == "TP")
    sl_count  = sum(1 for t in trades if t.exit_reason == "SL")
    eod_count = sum(1 for t in trades if t.exit_reason in ("EOD", "EOW"))
    be_count  = sum(1 for t in trades if t.exit_reason == "BE")

    # Direction breakdown
    longs  = [t for t in trades if t.direction == "LONG"]
    shorts = [t for t in trades if t.direction == "SHORT"]
    long_wins   = sum(1 for t in longs  if t.pnl > 0.01)
    short_wins  = sum(1 for t in shorts if t.pnl > 0.01)
    long_pnl    = round(sum(t.pnl for t in longs),  2)
    short_pnl   = round(sum(t.pnl for t in shorts), 2)
    long_wr     = round(long_wins  / len(longs)  * 100, 1) if longs  else 0.0
    short_wr    = round(short_wins / len(shorts) * 100, 1) if shorts else 0.0

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss   = abs(sum(p for p in pnls if p < 0))
    net_pnl      = sum(pnls)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    rr_vals  = [t.rr_achieved for t in trades if t.rr_achieved > 0]
    avg_rr   = sum(rr_vals) / len(rr_vals) if rr_vals else 0.0

    # Max drawdown (on equity curve)
    eq = equity_df["balance"].values
    peak  = eq[0]
    max_dd = 0.0
    for v in eq:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (daily returns)
    eq_series     = equity_df["balance"]
    daily_returns = eq_series.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * math.sqrt(252)
    else:
        sharpe = 0.0

    # Monthly breakdown
    monthly: dict[str, dict] = {}
    for t in trades:
        month = t.date[:7]
        if month not in monthly:
            monthly[month] = {"trades": 0, "wins": 0, "pnl": 0.0}
        monthly[month]["trades"] += 1
        monthly[month]["wins"]   += 1 if t.pnl > 0.01 else 0
        monthly[month]["pnl"]    += t.pnl

    best_month  = max(monthly, key=lambda m: monthly[m]["pnl"]) if monthly else "--"
    worst_month = min(monthly, key=lambda m: monthly[m]["pnl"]) if monthly else "--"

    # Window breakdown
    window_stats: dict[str, dict] = {}
    for w in TRADE_WINDOWS:
        wt = [t for t in trades if t.window == w]
        ww = sum(1 for t in wt if t.pnl > 0.01)
        window_stats[w] = {
            "trades": len(wt),
            "wins":   ww,
            "losses": sum(1 for t in wt if t.pnl < -0.01),
            "bes":    sum(1 for t in wt if abs(t.pnl) <= 0.01),
            "pnl":    round(sum(t.pnl for t in wt), 2),
            "win_rate": round(ww / len(wt) * 100, 1) if wt else 0.0,
        }

    # Entry-type breakdown
    entry_type_stats: dict[str, dict] = {}
    for et in ("FVG_ENTRY", "OB_ENTRY", "MSS_ENTRY"):
        et_trades = [t for t in trades if t.entry_type == et]
        et_wins   = sum(1 for t in et_trades if t.pnl > 0.01)
        et_losses = sum(1 for t in et_trades if t.pnl < -0.01)
        entry_type_stats[et] = {
            "count":    len(et_trades),
            "wins":     et_wins,
            "losses":   et_losses,
            "bes":      sum(1 for t in et_trades if abs(t.pnl) <= 0.01),
            "pnl":      round(sum(t.pnl for t in et_trades), 2),
            "win_rate": round(et_wins / len(et_trades) * 100, 1) if et_trades else 0.0,
            "dir_wr":   round(et_wins / (et_wins + et_losses) * 100, 1)
                        if (et_wins + et_losses) > 0 else 0.0,
        }

    return {
        "total_trades":   total,
        "wins":           wins,
        "losses":         losses,
        "breakevens":     bes,
        "win_rate":       round(win_rate, 1),
        "directional_wr": directional_wr,
        "tp_count":       tp_count,
        "sl_count":       sl_count,
        "eod_count":      eod_count,
        "be_count":       be_count,
        # Direction breakdown
        "long_count":     len(longs),
        "short_count":    len(shorts),
        "long_wins":      long_wins,
        "short_wins":     short_wins,
        "long_win_rate":  long_wr,
        "short_win_rate": short_wr,
        "long_pnl":       long_pnl,
        "short_pnl":      short_pnl,
        # Aggregate
        "net_pnl":        round(net_pnl, 2),
        "gross_profit":   round(gross_profit, 2),
        "gross_loss":     round(gross_loss, 2),
        "profit_factor":  round(profit_factor, 2),
        "avg_rr":         round(avg_rr, 2),
        "max_drawdown":   round(max_dd, 2),
        "sharpe_ratio":   round(sharpe, 3),
        "best_month":     best_month,
        "best_month_pnl": round(monthly[best_month]["pnl"], 2) if monthly else 0,
        "worst_month":    worst_month,
        "worst_month_pnl":round(monthly[worst_month]["pnl"], 2) if monthly else 0,
        "monthly":        monthly,
        "window_stats":      window_stats,
        "entry_type_stats":  entry_type_stats,
        "final_balance":  round(ACCOUNT_SIZE + net_pnl, 2),
        "return_pct":     round(net_pnl / ACCOUNT_SIZE * 100, 2),
    }


# --- VWAP comparison ---------------------------------------------------------

def _compute_vwap_comparison(trades: list[BacktestTrade]) -> dict:
    """
    Compare performance with vs. without VWAP alignment filter.

    VWAP-aligned: entry is between session VWAP and TP target
      LONG  → entry > VWAP  (bullish confirmation)
      SHORT → entry < VWAP  (bearish confirmation)

    Returns a dict with side-by-side stats for the report.
    """
    def _quick_stats(subset: list[BacktestTrade]) -> dict:
        if not subset:
            return {"trades": 0, "wins": 0, "losses": 0, "bes": 0,
                    "win_rate": 0.0, "dir_wr": 0.0, "net_pnl": 0.0}
        wins   = sum(1 for t in subset if t.pnl > 0.01)
        losses = sum(1 for t in subset if t.pnl < -0.01)
        bes    = sum(1 for t in subset if abs(t.pnl) <= 0.01)
        wr     = round(wins / len(subset) * 100, 1)
        dir_wr = round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0.0
        net    = round(sum(t.pnl for t in subset), 2)
        return {"trades": len(subset), "wins": wins, "losses": losses, "bes": bes,
                "win_rate": wr, "dir_wr": dir_wr, "net_pnl": net}

    aligned     = [t for t in trades if t.vwap_aligned]
    not_aligned = [t for t in trades if not t.vwap_aligned]
    no_vwap     = [t for t in trades if t.vwap == 0.0]   # VWAP unavailable

    return {
        "all":         _quick_stats(trades),
        "aligned":     _quick_stats(aligned),
        "not_aligned": _quick_stats(not_aligned),
        "no_vwap":     len(no_vwap),
    }


# --- Output writers ----------------------------------------------------------

_CSV_SKIP = {"entry_ts_utc"}   # internal fields, not for CSV

def _save_csv(trades: list[BacktestTrade], path: Path) -> None:
    if not trades:
        return
    fields = [f for f in trades[0].__dataclass_fields__.keys() if f not in _CSV_SKIP]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in trades:
            writer.writerow({k: getattr(t, k) for k in fields})
    log.info("CSV saved: %s  (%d trades)", path, len(trades))


def _save_md(stats: dict, trades: list[BacktestTrade], path: Path,
             start: datetime, end: datetime,
             session_report: Optional[dict] = None,
             vwap_cmp: Optional[dict] = None) -> None:
    monthly = stats.get("monthly", {})

    monthly_table = "| Month | Trades | Wins | P&L |\n|-------|--------|------|-----|\n"
    for m, data in sorted(monthly.items()):
        wr = data["wins"] / data["trades"] * 100 if data["trades"] else 0
        monthly_table += (
            f"| {m} | {data['trades']} | {data['wins']} ({wr:.0f}%) "
            f"| ${data['pnl']:+.2f} |\n"
        )

    _WIN = {"ASIAN_SWEEP_WINDOW": "Asian Sweep", "NY_AM_SILVER_BULLET": "NY AM", "NY_PM_SILVER_BULLET": "NY PM"}
    ws = stats.get("window_stats", {})
    window_table = "| Window | Trades | W / L / BE | Win Rate | Net P&L |\n|--------|--------|-----------|----------|--------|\n"
    for key in ("ASIAN_SWEEP_WINDOW", "NY_AM_SILVER_BULLET", "NY_PM_SILVER_BULLET"):
        d = ws.get(key, {"trades": 0, "wins": 0, "losses": 0, "bes": 0, "pnl": 0.0, "win_rate": 0.0})
        window_table += (
            f"| {_WIN[key]} | {d['trades']} | {d['wins']} / {d['losses']} / {d['bes']} "
            f"| {d['win_rate']:.1f}% | ${d['pnl']:+.2f} |\n"
        )

    ets = stats.get("entry_type_stats", {})
    entry_type_table = (
        "| Entry Type | Trades | W / L / BE | WR% | Dir WR | Net P&L |\n"
        "|------------|--------|-----------|-----|--------|--------|\n"
    )
    for et in ("FVG_ENTRY", "OB_ENTRY", "MSS_ENTRY"):
        d = ets.get(et, {"count": 0, "wins": 0, "losses": 0, "bes": 0, "pnl": 0.0, "win_rate": 0.0, "dir_wr": 0.0})
        entry_type_table += (
            f"| {et} | {d['count']} | {d['wins']} / {d['losses']} / {d['bes']} "
            f"| {d['win_rate']:.1f}% | {d['dir_wr']:.1f}% | ${d['pnl']:+.2f} |\n"
        )

    content = f"""\
---
type: backtest
symbol: XAUUSD
strategy: Silver Bullet (ICT)
period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}
generated: {datetime.now(NPT).strftime('%Y-%m-%d %H:%M NPT')}
---

# Backtest Report -- XAUUSD Silver Bullet

**Period:** {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}

---

## Summary Statistics

| Metric           | Value              |
|------------------|--------------------|
| Total Trades     | {stats['total_trades']}  |
| Wins (pnl > 0)   | {stats['wins']}    |
| Losses (pnl < 0) | {stats['losses']}  |
| Breakevens       | {stats['breakevens']}  |
| Win Rate         | {stats['win_rate']}%  |
| Directional WR   | {stats['directional_wr']}% (excl. BE)  |
| Exit: TP/SL/EOD/BE | {stats['tp_count']}/{stats['sl_count']}/{stats['eod_count']}/{stats['be_count']}  |
| LONG trades        | {stats['long_count']} ({stats['long_win_rate']}% WR, ${stats['long_pnl']:+,.2f})  |
| SHORT trades       | {stats['short_count']} ({stats['short_win_rate']}% WR, ${stats['short_pnl']:+,.2f})  |
| Net P&L          | ${stats['net_pnl']:+,.2f}  |
| Return           | {stats['return_pct']:+.2f}%  |
| Final Balance    | ${stats['final_balance']:,.2f}  |
| Profit Factor    | {stats['profit_factor']}  |
| Avg R:R          | 1:{stats['avg_rr']}  |
| Max Drawdown     | {stats['max_drawdown']}%  |
| Sharpe Ratio     | {stats['sharpe_ratio']}  |
| Best Month       | {stats['best_month']} (${stats['best_month_pnl']:+,.2f})  |
| Worst Month      | {stats['worst_month']} (${stats['worst_month_pnl']:+,.2f})  |

---

## Monthly Breakdown

{monthly_table}

---

## Session Window Breakdown

{window_table}

---

## Entry Type Breakdown

{entry_type_table}

---

## Session Funnel Analysis

{"_No session data available._" if not session_report else (
"| Stage | Sessions |\\n|-------|----------|\n"
+ f"| Total sessions scanned | {session_report['total_sessions']} |\\n"
+ f"| Skipped: already in trade | {session_report['skip_in_trade']} |\\n"
+ f"| Skipped: max trades/day | {session_report['skip_max_trades']} |\\n"
+ f"| Skipped: no PDH/PDL data | {session_report['skip_no_pdh_pdl']} |\\n"
+ f"| Skipped: H4 ranging | {session_report['skip_ranging']} |\\n"
+ f"| **Scannable sessions** | **{session_report['scannable']}** |\\n"
+ f"| London skipped: prior session SL | {session_report['skip_prev_session_loss']} |\\n"
+ f"| SHORT blocked: bull regime (price > 200 SMA) | {session_report['skip_bull_regime_short']} |\\n"
+ f"| SHORT blocked: DXY H4 not bullish | {session_report['skip_short_no_bull_dxy']} |\\n"
+ f"| SHORT blocked: price above 50 SMA | {session_report['skip_short_above_50sma']} |\\n"
+ f"| LONG blocked: price below 50 SMA | {session_report['skip_long_below_50sma']} |\\n"
+ f"| No liquidity sweep | {session_report['skip_no_sweep']} |\\n"
+ f"| Sweep found, no MSS | {session_report['skip_no_mss']} |\\n"
+ f"| MSS found, no FVG | {session_report['skip_no_fvg']} |\\n"
+ f"| FVG in wrong zone | {session_report['skip_wrong_zone']} |\\n"
+ f"| RR below minimum | {session_report['skip_low_rr']} |\\n"
+ f"| **Trades taken** | **{session_report['sessions_traded']}** |\\n"
+ (f"| Session-to-trade rate | {session_report['sessions_traded']/session_report['scannable']*100:.1f}% |"
   if session_report['scannable'] > 0 else "")
)}

---

## VWAP Filter Comparison

Session VWAP resets at each session open (Asian 00:00 UTC, London 07:00 UTC, NY 12:00 UTC).
VWAP-aligned = FVG entry is between session VWAP and TP target.

{"_VWAP data not available._" if not vwap_cmp else (
"| Metric | All Trades | VWAP-Aligned | Not Aligned |\\n"
+ "|--------|-----------|-------------|-------------|\\n"
+ f"| Total trades | {vwap_cmp['all']['trades']} | {vwap_cmp['aligned']['trades']} | {vwap_cmp['not_aligned']['trades']} |\\n"
+ f"| Wins | {vwap_cmp['all']['wins']} | {vwap_cmp['aligned']['wins']} | {vwap_cmp['not_aligned']['wins']} |\\n"
+ f"| Losses | {vwap_cmp['all']['losses']} | {vwap_cmp['aligned']['losses']} | {vwap_cmp['not_aligned']['losses']} |\\n"
+ f"| Breakevens | {vwap_cmp['all']['bes']} | {vwap_cmp['aligned']['bes']} | {vwap_cmp['not_aligned']['bes']} |\\n"
+ f"| Win Rate | {vwap_cmp['all']['win_rate']}% | {vwap_cmp['aligned']['win_rate']}% | {vwap_cmp['not_aligned']['win_rate']}% |\\n"
+ f"| Dir WR (excl BE) | {vwap_cmp['all']['dir_wr']}% | {vwap_cmp['aligned']['dir_wr']}% | {vwap_cmp['not_aligned']['dir_wr']}% |\\n"
+ f"| Net P&L | ${vwap_cmp['all']['net_pnl']:+,.2f} | ${vwap_cmp['aligned']['net_pnl']:+,.2f} | ${vwap_cmp['not_aligned']['net_pnl']:+,.2f} |\\n"
+ (f"\\n_Note: {vwap_cmp['no_vwap']} trade(s) had no VWAP data available (counted as aligned)._"
   if vwap_cmp['no_vwap'] > 0 else "")
)}

---

## Notes

- Strategy: ICT Silver Bullet -- Liquidity Sweep + MSS + FVG
- Account: ${ACCOUNT_SIZE:,.0f} starting balance, {RISK_PCT}% risk per trade
- Sessions traded: Asian Sweep (12:45-13:45 NPT), London Open (13:45-14:45 NPT), NY AM (19:45-20:45 NPT), NY PM (23:45-00:45 NPT)
- Breakeven triggered at {int(BE_TRIGGER*100)}% of way to TP
- VWAP: session VWAP computed from M5 candles; resets at Asian/London/NY session open
- News filter NOT applied in backtest (conservative: would reduce trades)
- High-risk days (Mon/Fri) NOT filtered in backtest

---

## Interpretation

{"[CAUTION] Win rate is suspiciously high -- check for look-ahead bias" if stats['win_rate'] > 75 else "[OK] Win rate appears realistic"}
{"[CAUTION] Profit factor > 3 may indicate overfitting" if stats['profit_factor'] > 3 else "[OK] Profit factor within normal range"}
{"[OK] Max drawdown is manageable" if stats['max_drawdown'] < 10 else "[WARN] High drawdown -- review position sizing"}

---

## Tags

#backtest #xauusd #silver-bullet #ict #vwap
"""
    path.write_text(content, encoding="utf-8")
    log.info("Markdown report saved: %s", path)


# --- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    import sys

    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Parse optional CLI args: python backtester.py 3  (months) ----------
    months = int(sys.argv[1]) if len(sys.argv) > 1 else 6

    print("=" * 60)
    print(f"  XAUUSD Silver Bullet Backtester")
    print(f"  Symbol: {SYMBOL}  |  Months: {months}")
    print(f"  Account: ${ACCOUNT_SIZE:,.0f}  |  Risk: {RISK_PCT}%/trade")
    print("=" * 60)

    if not connect():
        raise SystemExit("Could not connect to MT5.")

    try:
        end_dt   = datetime(2026, 4, 11, tzinfo=UTC)
        start_dt = end_dt - timedelta(days=30 * months)

        trades, equity_df, session_best = run_backtest(months, start_dt, end_dt)

        if not trades:
            print("\n  No trades generated in this period.")
            print("  Possible causes: no active trading windows in the data,")
            print("  all H4 biases returned RANGING, or FVG conditions not met.")
        else:
            stats = _compute_stats(trades, equity_df)

            # Print to console
            print(f"\n{'=' * 60}")
            print(f"  BACKTEST RESULTS")
            print(f"{'=' * 60}")
            print(f"  Period        : {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
            print(f"  Total Trades  : {stats['total_trades']}")
            print(f"  Win Rate      : {stats['win_rate']}%  ({stats['wins']}W  {stats['losses']}L  {stats['breakevens']}BE)")
            print(f"  Directional WR: {stats['directional_wr']}%  (excl. BE exits)")
            print(f"  Exit Reasons  : TP={stats['tp_count']}  SL={stats['sl_count']}  EOD={stats['eod_count']}  BE={stats['be_count']}")
            print(f"  Net P&L       : ${stats['net_pnl']:+,.2f}  ({stats['return_pct']:+.2f}%)")
            print(f"  Final Balance : ${stats['final_balance']:,.2f}")
            print(f"  Avg R:R       : 1:{stats['avg_rr']}")
            print(f"  Max Drawdown  : {stats['max_drawdown']}%")
            print(f"  Sharpe Ratio  : {stats['sharpe_ratio']}")
            print(f"  Profit Factor : {stats['profit_factor']}")
            print(f"  Best Month    : {stats['best_month']}  (${stats['best_month_pnl']:+,.2f})")
            print(f"  Worst Month   : {stats['worst_month']}  (${stats['worst_month_pnl']:+,.2f})")
            print(f"{'=' * 60}")
            print(f"  Direction Breakdown")
            print(f"  {'':2} {'Dir':<6} {'Count':>5}  {'Wins':>4}  {'WR%':>5}  {'P&L':>10}")
            print(f"  {'':2} {'-'*6} {'-'*5}  {'-'*4}  {'-'*5}  {'-'*10}")
            print(f"  {'':2} {'LONG':<6} {stats['long_count']:>5}  "
                  f"{stats['long_wins']:>4}  {stats['long_win_rate']:>4.1f}%  "
                  f"${stats['long_pnl']:>+9,.2f}")
            print(f"  {'':2} {'SHORT':<6} {stats['short_count']:>5}  "
                  f"{stats['short_wins']:>4}  {stats['short_win_rate']:>4.1f}%  "
                  f"${stats['short_pnl']:>+9,.2f}")
            if stats['short_count'] == 0:
                print(f"  [NOTE] No SHORT trades -- filtered out by regime/qualification rules.")
            elif stats['short_count'] < stats['long_count'] * 0.20:
                pct = round(stats['short_count'] / stats['total_trades'] * 100, 1)
                print(f"  [NOTE] SHORTs are {pct}% of trades -- market trending bullish.")
            print(f"{'=' * 60}")
            print(f"  Session Window Breakdown")
            _WIN_LABEL = {
                "ASIAN_SWEEP_WINDOW":         "Asian Sweep  (12:45-13:45)",
                "LONDON_OPEN_SILVER_BULLET":  "London Open  (13:45-14:45)",
                "NY_AM_SILVER_BULLET":        "NY AM        (19:45-20:45)",
                "NY_PM_SILVER_BULLET":        "NY PM        (23:45-00:45)",
            }
            print(f"  {'':2} {'Window':<28} {'Cnt':>4}  {'W/L/BE':>8}  {'WR%':>5}  {'P&L':>10}")
            print(f"  {'':2} {'-'*28} {'-'*4}  {'-'*8}  {'-'*5}  {'-'*10}")
            for key in ("ASIAN_SWEEP_WINDOW", "NY_AM_SILVER_BULLET", "NY_PM_SILVER_BULLET"):
                d = stats['window_stats'].get(key, {})
                if not d or d['trades'] == 0:
                    print(f"  {'':2} {_WIN_LABEL[key]:<28} {'0':>4}  {'--':>8}  {'--':>5}  {'$0.00':>10}")
                    continue
                wl = f"{d['wins']}/{d['losses']}/{d['bes']}"
                print(f"  {'':2} {_WIN_LABEL[key]:<28} {d['trades']:>4}  {wl:>8}  "
                      f"{d['win_rate']:>4.1f}%  ${d['pnl']:>+9,.2f}")
            print(f"{'=' * 60}")

            # Entry-type breakdown
            print(f"  Entry Type Breakdown")
            ets = stats.get('entry_type_stats', {})
            print(f"  {'':2} {'Type':<12} {'Cnt':>4}  {'W/L/BE':>8}  {'WR%':>5}  {'Dir WR':>7}  {'P&L':>10}")
            print(f"  {'':2} {'-'*12} {'-'*4}  {'-'*8}  {'-'*5}  {'-'*7}  {'-'*10}")
            for et_key in ("FVG_ENTRY", "OB_ENTRY", "MSS_ENTRY"):
                d = ets.get(et_key, {})
                if not d or d['count'] == 0:
                    print(f"  {'':2} {et_key:<12} {'0':>4}  {'--':>8}  {'--':>5}  {'--':>7}  {'$0.00':>10}")
                    continue
                wl = f"{d['wins']}/{d['losses']}/{d['bes']}"
                print(f"  {'':2} {et_key:<12} {d['count']:>4}  {wl:>8}  "
                      f"{d['win_rate']:>4.1f}%  {d['dir_wr']:>5.1f}%  ${d['pnl']:>+9,.2f}")
            print(f"{'=' * 60}")

            # All trades table
            print(f"\n  All trades ({stats['total_trades']} total):")
            print(f"  {'#':<4} {'Date':<12} {'Dir':<6} {'Entry':>8} {'Exit':>8} {'PnL':>9} {'Reason':<6} {'RR':>5}  {'VWAP':>8} {'Align':<6} Window")
            for t in trades:
                win_marker = " *" if t.pnl > 0.01 else ("  " if abs(t.pnl) <= 0.01 else "  ")
                vwap_str  = f"{t.vwap:>8.2f}" if t.vwap > 0 else "       -"
                align_str = "YES" if t.vwap_aligned else "NO "
                print(f"  {t.trade_id:<4} {t.date:<12} {t.direction:<6} "
                      f"{t.entry:>8.2f} {t.exit_price:>8.2f} "
                      f"${t.pnl:>+8.2f} {t.exit_reason:<6} 1:{t.rr_achieved:.1f}"
                      f"  {vwap_str} {align_str}   {t.window}{win_marker}")

            # VWAP comparison
            vwap_cmp = _compute_vwap_comparison(trades)
            vc_a  = vwap_cmp['all']
            vc_al = vwap_cmp['aligned']
            vc_na = vwap_cmp['not_aligned']
            print(f"\n{'=' * 60}")
            print(f"  VWAP FILTER COMPARISON")
            print(f"  (aligned = FVG entry between session VWAP and TP)")
            print(f"{'=' * 60}")
            print(f"  {'Metric':<22} {'All':>10}  {'VWAP-Aligned':>12}  {'Not Aligned':>11}")
            print(f"  {'-'*22} {'-'*10}  {'-'*12}  {'-'*11}")
            print(f"  {'Trades':<22} {vc_a['trades']:>10}  {vc_al['trades']:>12}  {vc_na['trades']:>11}")
            print(f"  {'Wins':<22} {vc_a['wins']:>10}  {vc_al['wins']:>12}  {vc_na['wins']:>11}")
            print(f"  {'Losses':<22} {vc_a['losses']:>10}  {vc_al['losses']:>12}  {vc_na['losses']:>11}")
            print(f"  {'Win Rate':<22} {vc_a['win_rate']:>9.1f}%  {vc_al['win_rate']:>11.1f}%  {vc_na['win_rate']:>10.1f}%")
            print(f"  {'Dir WR (excl BE)':<22} {vc_a['dir_wr']:>9.1f}%  {vc_al['dir_wr']:>11.1f}%  {vc_na['dir_wr']:>10.1f}%")
            print(f"  {'Net P&L':<22} ${vc_a['net_pnl']:>+9,.2f}  ${vc_al['net_pnl']:>+11,.2f}  ${vc_na['net_pnl']:>+10,.2f}")
            if vwap_cmp['no_vwap']:
                print(f"  [NOTE] {vwap_cmp['no_vwap']} trade(s) had no VWAP data -- counted as aligned.")
            print(f"{'=' * 60}")

            # Session funnel report
            sr = _compute_session_report(session_best)
            conv = (sr['sessions_traded'] / sr['scannable'] * 100
                    if sr['scannable'] > 0 else 0.0)
            print(f"\n{'=' * 60}")
            print(f"  SESSION FUNNEL ANALYSIS")
            print(f"{'=' * 60}")
            print(f"  Total sessions scanned    : {sr['total_sessions']}")
            print(f"  -- Skipped (in trade)     : {sr['skip_in_trade']}")
            print(f"  -- Skipped (max trades/d) : {sr['skip_max_trades']}")
            print(f"  -- Skipped (no PDH/PDL)   : {sr['skip_no_pdh_pdl']}")
            print(f"  -- Skipped (H4 ranging)   : {sr['skip_ranging']}")
            print(f"  ----------------------------------------")
            print(f"  Scannable sessions        : {sr['scannable']}")
            print(f"  London skipped (prev loss): {sr['skip_prev_session_loss']}")
            print(f"  SHORT: bull regime block  : {sr['skip_bull_regime_short']}")
            print(f"  SHORT: DXY not BULLISH    : {sr['skip_short_no_bull_dxy']}")
            print(f"  SHORT: price > 50 SMA     : {sr['skip_short_above_50sma']}")
            print(f"  LONG:  price < 50 SMA     : {sr['skip_long_below_50sma']}")
            print(f"  -- No liquidity sweep     : {sr['skip_no_sweep']}")
            print(f"  -- Sweep found, no MSS    : {sr['skip_no_mss']}")
            print(f"  -- MSS found, no FVG      : {sr['skip_no_fvg']}")
            print(f"  -- FVG wrong zone         : {sr['skip_wrong_zone']}")
            print(f"  -- RR too low             : {sr['skip_low_rr']}")
            print(f"  ----------------------------------------")
            print(f"  Trades taken              : {sr['sessions_traded']}")
            print(f"  Session -> trade rate     : {conv:.1f}%")
            print(f"{'=' * 60}")

            # Save outputs
            ts_str   = datetime.now().strftime("%Y%m%d_%H%M")
            csv_path = OUTPUT_DIR / f"backtest_{ts_str}.csv"
            md_path  = OUTPUT_DIR / f"backtest_{ts_str}.md"

            _save_csv(trades, csv_path)
            _save_md(stats, trades, md_path, start_dt, end_dt, sr, vwap_cmp)

            # Also copy MD to Obsidian vault
            try:
                from obsidian_logger import VAULT_PATH
                obs_bt = VAULT_PATH / "trades" / "backtests"
                obs_bt.mkdir(parents=True, exist_ok=True)
                obs_md = obs_bt / f"backtest_{ts_str}.md"
                obs_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"\n  Obsidian note: {obs_md}")
            except Exception as exc:
                log.warning("Could not copy to Obsidian: %s", exc)

            print(f"\n  CSV     : {csv_path}")
            print(f"  Report  : {md_path}")

    finally:
        disconnect()
