# -*- coding: utf-8 -*-
"""
telegram_alerts.py
------------------
Sends formatted trade alerts and reports to a Telegram chat.

Uses python-telegram-bot v20+ (async).  All public functions are
synchronous wrappers so the rest of the bot can call them without
managing an event loop.

Credentials from .env:
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv
from telegram import Bot, InputFile
from telegram.constants import ParseMode
from telegram.error import TelegramError

from config import LOG_LEVEL

load_dotenv()
log = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID",   "")
NPT      = timezone(timedelta(hours=5, minutes=45))

if not _TOKEN or not _CHAT_ID:
    raise EnvironmentError(
        "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env"
    )


# --- Async helpers ----------------------------------------------------------

def _run(coro):
    """Run an async coroutine from sync code."""
    try:
        loop = asyncio.get_running_loop()
        # Already inside a running loop (e.g. Jupyter) — schedule and wait
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)


async def _send_text(text: str, parse_mode: str = ParseMode.HTML) -> bool:
    try:
        async with Bot(token=_TOKEN) as bot:
            await bot.send_message(
                chat_id=_CHAT_ID,
                text=text,
                parse_mode=parse_mode,
            )
        return True
    except TelegramError as exc:
        log.error("Telegram send_message failed: %s", exc)
        return False


async def _send_photo(
    image_path: Union[str, Path],
    caption:    str,
    parse_mode: str = ParseMode.HTML,
) -> bool:
    path = Path(image_path)
    if not path.exists():
        log.warning("Screenshot not found: %s  -- sending text only", path)
        return await _send_text(caption, parse_mode)
    try:
        async with Bot(token=_TOKEN) as bot:
            with open(path, "rb") as f:
                await bot.send_photo(
                    chat_id=_CHAT_ID,
                    photo=f,
                    caption=caption,
                    parse_mode=parse_mode,
                )
        return True
    except TelegramError as exc:
        log.error("Telegram send_photo failed: %s", exc)
        return False


# --- Emoji helpers ----------------------------------------------------------

def _bias_icon(bias: str) -> str:
    b = bias.upper()
    if b == "BULLISH":  return "up"
    if b == "BEARISH":  return "down"
    return "right"


def _signal_icon(signal: str) -> str:
    s = signal.upper()
    if "STRONG_LONG"  in s: return "green circle  STRONG LONG"
    if "WEAK_LONG"    in s: return "yellow circle WEAK LONG"
    if "STRONG_SHORT" in s: return "red circle    STRONG SHORT"
    if "WEAK_SHORT"   in s: return "yellow circle WEAK SHORT"
    return "white circle   NEUTRAL"


def _result_icon(result: str) -> str:
    r = result.upper()
    if r in ("WIN", "TP", "PROFIT"):   return "[WIN]"
    if r in ("LOSS", "SL", "STOPPED"): return "[LOSS]"
    if r in ("BE", "BREAKEVEN"):       return "[BE]"
    return "[OPEN]"


def _windows_label(signal: str) -> str:
    s = signal.upper()
    if "STRONG" in s: return "ALL 3"
    if "WEAK"   in s: return "NY AM only"
    return "NONE"


# --- 1. Daily bias report ---------------------------------------------------

def send_bias_report(bias_data: dict) -> bool:
    """
    Send the morning bias summary.

    bias_data keys (from bias_combiner.BiasResult or equivalent dict):
      technical_bias, news_bias, dxy_bias, divergence,
      technical_score, news_score, dxy_score, final_score, signal
    """
    now      = datetime.now(tz=NPT)
    gold     = bias_data.get("technical_bias", "--")
    news     = bias_data.get("news_bias",      "--")
    dxy      = bias_data.get("dxy_bias",       "--")
    div      = bias_data.get("divergence",     "--")
    score    = float(bias_data.get("final_score", 0))
    signal   = bias_data.get("signal", "NEUTRAL")
    g_score  = float(bias_data.get("technical_score", 0))
    n_score  = float(bias_data.get("news_score",      0))
    d_score  = float(bias_data.get("dxy_score",       0))
    news_rsn = bias_data.get("news_reason", "")
    windows  = _windows_label(signal)
    div_warn = "  [!] DIVERGENCE PENALTY APPLIED" if div == "DIVERGENCE" else ""

    sign = "+" if score >= 0 else ""
    msg = (
        f"[MORNING] DAILY BIAS REPORT\n"
        f"{now.strftime('%Y-%m-%d  %I:%M %p NPT')}\n"
        f"{'=' * 32}\n"
        f"Gold H4 : [{_bias_icon(gold)}] {gold}  ({'+' if g_score>=0 else ''}{g_score:.2f})\n"
        f"News    : [{_bias_icon(news)}] {news}"
        + (f"  ({news_rsn})" if news_rsn else "") + f"  ({'+' if n_score>=0 else ''}{n_score:.2f})\n"
        f"DXY     : [{_bias_icon(dxy)}] {dxy}  ({'+' if d_score>=0 else ''}{d_score:.2f})\n"
        f"{'=' * 32}\n"
        f"Signal  : {_signal_icon(signal)}\n"
        f"Score   : {sign}{score:.2f}"
        + (div_warn) + "\n"
        f"Windows : {windows}"
    )

    ok = _run(_send_text(msg))
    if ok:
        log.info("Telegram: bias report sent")
    return ok


# --- 2. Trade entered alert -------------------------------------------------

def send_trade_alert(
    trade_data:      dict,
    screenshot_path: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Send alert when a trade is entered, optionally with entry screenshot.

    trade_data keys:
      direction, strategy, entry, sl, tp, rr, lot_size,
      final_score, sweep_level, fvg_top, fvg_bottom, session
    """
    d         = trade_data
    direction = d.get("direction", "LONG").upper()
    strategy  = d.get("strategy",  "--")
    entry     = d.get("entry",     "--")
    sl        = d.get("sl",        "--")
    tp        = d.get("tp",        "--")
    rr        = d.get("rr",        "--")
    lot       = d.get("lot_size",  "--")
    score     = d.get("final_score", 0)
    session   = d.get("session",   "--")
    sweep     = d.get("sweep_level","--")
    fvg_top   = d.get("fvg_top",   "--")
    fvg_bot   = d.get("fvg_bottom","--")

    icon  = "[LONG]" if direction == "LONG" else "[SHORT]"
    sign  = "+" if float(score) >= 0 else ""

    msg = (
        f"{icon} TRADE ENTERED\n"
        f"XAUUSD {direction}\n"
        f"{'=' * 32}\n"
        f"Strategy : {strategy}\n"
        f"Session  : {session}\n"
        f"Sweep Lvl: {sweep}\n"
        f"FVG Zone : {fvg_top} - {fvg_bot}\n"
        f"{'=' * 32}\n"
        f"Entry : {entry}\n"
        f"SL    : {sl}  |  TP: {tp}\n"
        f"RR    : 1:{rr}  |  Lot: {lot}\n"
        f"Score : {sign}{score:.2f}"
    )

    if screenshot_path:
        ok = _run(_send_photo(screenshot_path, msg))
    else:
        ok = _run(_send_text(msg))

    if ok:
        log.info("Telegram: trade alert sent (%s %s entry=%s)", direction, strategy, entry)
    return ok


# --- 3. Trade result alert --------------------------------------------------

def send_trade_result(
    trade_data:      dict,
    screenshot_path: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Send alert when a trade closes.

    trade_data keys:
      direction, entry, exit_price, result, pnl, duration,
      wins_today, losses_today, total_pnl_today
    """
    d         = trade_data
    direction = d.get("direction",  "LONG").upper()
    entry     = d.get("entry",      "--")
    exit_p    = d.get("exit_price", "--")
    result    = d.get("result",     "OPEN").upper()
    pnl       = d.get("pnl",        "--")
    duration  = d.get("duration",   "--")
    wins      = d.get("wins_today",    0)
    losses    = d.get("losses_today",  0)
    day_pnl   = d.get("total_pnl_today", "--")

    icon = _result_icon(result)

    msg = (
        f"{icon} TRADE CLOSED -- {result}\n"
        f"XAUUSD {direction}\n"
        f"{'=' * 32}\n"
        f"Entry : {entry}  >>  Exit: {exit_p}\n"
        f"P&L   : {pnl}\n"
        f"Time  : {duration}\n"
        f"{'=' * 32}\n"
        f"Today : {wins}W {losses}L  |  {day_pnl}"
    )

    if screenshot_path:
        ok = _run(_send_photo(screenshot_path, msg))
    else:
        ok = _run(_send_text(msg))

    if ok:
        log.info("Telegram: trade result sent (result=%s pnl=%s)", result, pnl)
    return ok


# --- 4. Warning / blocked trade ---------------------------------------------

def send_warning(message: str) -> bool:
    """
    Send a warning message (blocked trade, news proximity, drawdown alert).

    message : plain-text reason, e.g.
      "High impact news in 30 min (US CPI)"
      "Daily drawdown limit reached (2.1%)"
    """
    text = f"[!] WARNING\n{message}"
    ok   = _run(_send_text(text))
    if ok:
        log.info("Telegram: warning sent: %s", message)
    return ok


# --- 5. Daily summary -------------------------------------------------------

def send_daily_summary(summary_data: dict) -> bool:
    """
    Send end-of-day performance summary.

    summary_data keys:
      trades, wins, losses, breakevens,
      total_pnl, win_rate, best_rr, worst_pnl,
      date (optional)
    """
    d         = summary_data
    trades    = d.get("trades",    0)
    wins      = d.get("wins",      0)
    losses    = d.get("losses",    0)
    bes       = d.get("breakevens",0)
    total_pnl = d.get("total_pnl", "--")
    win_rate  = d.get("win_rate",  0.0)
    best_rr   = d.get("best_rr",   "--")
    worst_pnl = d.get("worst_pnl", "--")
    date_str  = d.get("date", datetime.now(tz=NPT).strftime("%Y-%m-%d"))

    no_trades = trades == 0

    if no_trades:
        msg = (
            f"[REPORT] DAILY SUMMARY -- {date_str}\n"
            f"{'=' * 32}\n"
            f"No trades taken today."
        )
    else:
        msg = (
            f"[REPORT] DAILY SUMMARY -- {date_str}\n"
            f"{'=' * 32}\n"
            f"Trades   : {trades}  ({wins}W  {losses}L  {bes}BE)\n"
            f"P&L      : {total_pnl}\n"
            f"Win Rate : {win_rate:.1f}%\n"
            f"Best R:R : 1:{best_rr}\n"
            f"Worst P&L: {worst_pnl}"
        )

    ok = _run(_send_text(msg))
    if ok:
        log.info("Telegram: daily summary sent (%d trades)", trades)
    return ok


# --- Entry Point ------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    print("=" * 55)
    print("  Telegram Alerts - XAUUSD Bot  (live test)")
    print("=" * 55)

    # 1. Bias report
    print("\n[1] Sending bias report...")
    send_bias_report({
        "technical_bias":  "BULLISH",
        "news_bias":       "BULLISH",
        "dxy_bias":        "BEARISH",
        "divergence":      "ALIGNED",
        "technical_score": 0.40,
        "news_score":      0.35,
        "dxy_score":       0.25,
        "final_score":     1.00,
        "signal":          "STRONG_LONG",
        "news_reason":     "CPI miss",
    })
    print("    Sent.")

    # 2. Trade alert with screenshot
    print("\n[2] Sending trade entry alert...")
    demo_screenshot = (
        Path(__file__).parent
        / "trades" / "2026-04" / "trade_001" / "pre_entry.png"
    )
    send_trade_alert(
        trade_data={
            "direction":   "LONG",
            "strategy":    "Silver Bullet NY AM",
            "session":     "NY_AM_SILVER_BULLET",
            "entry":       4711.10,
            "sl":          4707.20,
            "tp":          4801.22,
            "rr":          23.1,
            "lot_size":    0.05,
            "final_score": 1.00,
            "sweep_level": 4698.62,
            "fvg_top":     4712.80,
            "fvg_bottom":  4709.40,
        },
        screenshot_path=demo_screenshot,
    )
    print("    Sent.")

    # 3. Trade result
    print("\n[3] Sending trade result (WIN)...")
    send_trade_result(
        trade_data={
            "direction":       "LONG",
            "entry":           4711.10,
            "exit_price":      4801.22,
            "result":          "WIN",
            "pnl":             "+$182.50",
            "duration":        "47 min",
            "wins_today":      1,
            "losses_today":    0,
            "total_pnl_today": "+$182.50",
        },
        screenshot_path=None,
    )
    print("    Sent.")

    # 4. Warning
    print("\n[4] Sending warning...")
    send_warning("High impact news in 30 min (US CPI at 08:30 EST)")
    print("    Sent.")

    # 5. Daily summary
    print("\n[5] Sending daily summary...")
    send_daily_summary({
        "trades":    1,
        "wins":      1,
        "losses":    0,
        "breakevens":0,
        "total_pnl": "+$182.50",
        "win_rate":  100.0,
        "best_rr":   23.1,
        "worst_pnl": "+$182.50",
    })
    print("    Sent.")

    print("\n  Check your Telegram for all 5 messages.")
    print("=" * 55)
