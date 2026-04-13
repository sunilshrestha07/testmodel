# -*- coding: utf-8 -*-
"""
main.py  --  XAUUSD Silver Bullet Bot  v1.0
============================================
Startup  : connect MT5, bias report, Telegram
Loop     : every 30 s -- session check, risk gates,
           FVG scan, order placement, BE management
End-of-day: summary, graceful shutdown

PAPER_TRADE_MODE is read from config.py.
Start with True; flip to False only after 3+ clean paper days.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import MetaTrader5 as mt5

# --- Bot modules -------------------------------------------------------------
from config import (
    SYMBOL, MAGIC_NUMBER, PAPER_TRADE_MODE, LOG_LEVEL,
)
from mt5_connector        import connect, disconnect, fetch_candles
from session_manager      import (
    get_current_npt, get_active_session,
    get_minutes_until_next_window, is_high_risk_day,
    TRADING_WINDOWS,
)
from bias_combiner        import calculate_bias_score, print_bias_report, BiasResult
from market_structure     import (
    get_previous_day_levels, get_asian_session_levels,
)
from news_engine          import get_full_news_report, is_news_in_next_minutes
from risk_manager         import (
    check_daily_drawdown, check_daily_trade_count,
    check_spread, validate_rr, calculate_lot_size, get_trades_today,
)
from fvg_detector         import get_entry_setup, EntrySetup
from trade_executor       import (
    place_limit_order, monitor_open_trades, get_trade_result,
    modify_sl_to_breakeven,
)
from screenshot_manager   import capture_chart, get_screenshot_paths
from obsidian_logger      import create_trade_note, update_trade_outcome
from telegram_alerts      import (
    send_bias_report, send_trade_alert, send_trade_result,
    send_warning, send_daily_summary,
)

# --- Logging -----------------------------------------------------------------
LOG_FILE = Path(__file__).parent / "bot.log"
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("main")

# --- Constants ---------------------------------------------------------------
NPT             = timezone(timedelta(hours=5, minutes=45))
LOOP_INTERVAL   = 30          # seconds between each scan
MAX_RECONNECTS  = 3           # MT5 reconnect attempts before alerting
EOD_HOUR_NPT    = 1           # end-of-day at 01:00 AM NPT
BOT_START_HOUR  = 5           # restart watch starts at 05:45 AM NPT
VERSION         = "1.0"

# --- Bot state (mutable, in-memory only) -------------------------------------
class BotState:
    def __init__(self) -> None:
        self.bias:          Optional[BiasResult] = None
        self.session_levels: dict               = {}
        self.initial_balance: float             = 0.0
        self.trades_today:  int                 = 0
        self.wins_today:    int                 = 0
        self.losses_today:  int                 = 0
        self.day_pnl:       float               = 0.0
        self.open_tickets:  list[int]           = []
        self.trade_counter: int                 = 0
        self.last_eod_date: Optional[str]       = None
        self.connected:     bool                = False

state = BotState()


# ===========================================================================
# STARTUP
# ===========================================================================

def startup() -> bool:
    """
    One-time initialisation.
    Returns True if startup succeeds, False if MT5 cannot connect.
    """
    log.info("=" * 55)
    log.info("  XAUUSD Silver Bullet Bot  v%s", VERSION)
    log.info("  Mode: %s", "PAPER TRADE" if PAPER_TRADE_MODE else "LIVE EXECUTION")
    log.info("=" * 55)

    # 1. Connect MT5
    if not _connect_with_retry():
        return False

    # 2. Account snapshot
    acct = mt5.account_info()
    if acct:
        state.initial_balance = acct.balance
        log.info("Account: %s  Balance: $%.2f  Leverage: 1:%d",
                 acct.name, acct.balance, acct.leverage)

    # 3. Session levels (PDH/PDL/ASH/ASL)
    _refresh_session_levels()

    # 4. Daily bias
    try:
        log.info("Calculating daily bias...")
        news_report  = get_full_news_report()
        state.bias   = calculate_bias_score(news_report=news_report)
        print_bias_report(state.bias)
        send_bias_report(_bias_to_dict(state.bias))
    except Exception as exc:
        log.error("Bias calculation failed: %s", exc)
        send_warning(f"Bot startup: bias engine error -- {exc}")

    # 5. Pre-entry context screenshot (H4)
    try:
        capture_chart("startup", "pre_entry")
        log.info("Startup H4 screenshot saved.")
    except Exception as exc:
        log.warning("Startup screenshot failed (non-fatal): %s", exc)

    log.info("Startup complete.  Entering main loop.")
    return True


def _refresh_session_levels() -> None:
    """Update PDH/PDL and ASH/ASL from MT5."""
    try:
        day   = get_previous_day_levels(SYMBOL)
        asian = get_asian_session_levels(SYMBOL)
        state.session_levels = {
            "pdh": day.pdh   if day   else None,
            "pdl": day.pdl   if day   else None,
            "ash": asian.ash if asian else None,
            "asl": asian.asl if asian else None,
        }
        log.info(
            "Session levels: PDH=%.2f  PDL=%.2f  ASH=%s  ASL=%s",
            state.session_levels.get("pdh") or 0,
            state.session_levels.get("pdl") or 0,
            state.session_levels.get("ash") or "N/A",
            state.session_levels.get("asl") or "N/A",
        )
    except Exception as exc:
        log.warning("Could not refresh session levels: %s", exc)


def _connect_with_retry(max_tries: int = MAX_RECONNECTS) -> bool:
    for attempt in range(1, max_tries + 1):
        log.info("MT5 connect attempt %d/%d ...", attempt, max_tries)
        if connect():
            state.connected = True
            return True
        time.sleep(5)
    log.error("MT5 connection failed after %d attempts.", max_tries)
    send_warning("MT5 connection failed -- bot is NOT running.")
    state.connected = False
    return False


# ===========================================================================
# MAIN LOOP
# ===========================================================================

def run_loop() -> None:
    """Blocking loop. Exits cleanly on KeyboardInterrupt or EOD."""
    log.info("Main loop started  (interval: %ds)", LOOP_INTERVAL)

    while True:
        try:
            _loop_tick()
        except KeyboardInterrupt:
            log.info("Keyboard interrupt -- shutting down.")
            _shutdown()
            break
        except Exception as exc:
            log.error("Unhandled exception in loop: %s\n%s", exc, traceback.format_exc())
            send_warning(f"Bot loop error: {exc}")

        time.sleep(LOOP_INTERVAL)


def _loop_tick() -> None:
    """Single iteration of the main loop."""
    now     = get_current_npt()
    session = get_active_session()

    # --- End-of-day check ----------------------------------------------------
    if _is_eod(now):
        _end_of_day(now)
        return

    # --- Refresh ASH/ASL once the Asian session has closed (12:45 NPT) ------
    if now.hour == 12 and now.minute == 45:
        _refresh_session_levels()

    # --- Dashboard -----------------------------------------------------------
    _print_dashboard(now, session)

    # --- MT5 health check ----------------------------------------------------
    if not _ensure_connected():
        return

    # --- No active trading window --------------------------------------------
    if session not in TRADING_WINDOWS:
        return

    # =========================================================================
    # GATE CHECKS -- skip if any fails
    # =========================================================================
    acct = mt5.account_info()
    if acct is None:
        log.warning("Could not fetch account info -- skipping tick")
        return

    balance = acct.balance

    # Gate 1: Daily drawdown
    dd_breached, dd_pct = check_daily_drawdown(balance, state.initial_balance)
    if dd_breached:
        log.warning("Drawdown gate: %.2f%% -- no new entries", dd_pct)
        return

    # Gate 2: Daily trade count
    state.trades_today = get_trades_today()
    limit_reached, remaining = check_daily_trade_count(state.trades_today)
    if limit_reached:
        return

    # Gate 3: High-impact news
    events = None
    try:
        from news_engine import get_todays_events
        events = get_todays_events()
        news_near, next_event = is_news_in_next_minutes(45, events)
        if news_near and next_event:
            msg = f"Trade blocked: {next_event.name} in <45 min ({next_event.time_est})"
            log.warning(msg)
            send_warning(msg)
            return
    except Exception as exc:
        log.warning("News check failed (non-fatal): %s", exc)

    # Gate 4: Spread
    spread_ok, spread_pips = check_spread(SYMBOL)
    if not spread_ok:
        log.warning("Spread too wide: %.1f pips -- skipping", spread_pips)
        return

    # Gate 5: Bias must not be NEUTRAL
    if state.bias is None or state.bias.direction == "NONE":
        log.debug("Bias is NEUTRAL -- no trade today")
        return

    direction = state.bias.direction.lower()   # "long" or "short"

    # =========================================================================
    # ENTRY SCAN
    # =========================================================================
    try:
        setup: EntrySetup = get_entry_setup(SYMBOL, direction, state.session_levels)
    except Exception as exc:
        log.error("FVG detector error: %s", exc)
        return

    if not setup.valid:
        log.debug("Scanning %s | %s | No setup: %s", session, direction.upper(), setup.reason)
        return

    # =========================================================================
    # SETUP FOUND -- final validation
    # =========================================================================
    log.info("Setup found: %s", setup)

    # RR check
    rr_ok, rr = validate_rr(setup.entry, setup.sl, setup.tp, min_rr=2.0)
    if not rr_ok:
        log.warning("RR too low (%.2f) -- skipping", rr)
        return

    # Lot size
    sl_pips  = abs(setup.entry - setup.sl) / 0.10
    modifier = state.bias.lot_modifier
    lot      = calculate_lot_size(balance, risk_percent=1.0,
                                  sl_pips=sl_pips, lot_modifier=modifier)

    # Trade ID
    state.trade_counter += 1
    trade_id = f"{datetime.now(tz=NPT).strftime('%Y%m%d')}_{state.trade_counter:02d}"

    # Screenshot
    screenshot_paths = get_screenshot_paths(trade_id)
    try:
        entry_img = capture_chart(trade_id, "entry")
        log.info("Entry screenshot saved: %s", entry_img)
    except Exception as exc:
        log.warning("Entry screenshot failed: %s", exc)
        entry_img = None

    # Build trade data dict
    fvg  = setup.fvg_zone
    trade_data = {
        "trade_id":       trade_id,
        "direction":      setup.direction,
        "session":        session,
        "strategy":       _session_to_strategy(session),
        "gold_bias":      state.bias.technical_bias,
        "gold_score":     state.bias.technical_score,
        "news_bias":      state.bias.news_bias,
        "news_score":     state.bias.news_score,
        "dxy_bias":       state.bias.dxy_bias,
        "dxy_score":      state.bias.dxy_score,
        "divergence":     state.bias.divergence,
        "final_score":    state.bias.final_score,
        "signal_strength":state.bias.signal,
        "sweep_level":    setup.sweep_level,
        "fvg_top":        fvg.top      if fvg else "--",
        "fvg_bottom":     fvg.bottom   if fvg else "--",
        "entry":          setup.entry,
        "sl":             setup.sl,
        "tp":             setup.tp,
        "rr":             setup.rr,
        "lot_size":       lot,
        "result":         "OPEN",
    }

    # Place order
    order_result = place_limit_order(
        symbol    = SYMBOL,
        direction = setup.direction,
        entry     = setup.entry,
        sl        = setup.sl,
        tp        = setup.tp,
        lot_size  = lot,
        comment   = f"sb_bot_{trade_id}",
    )

    if not order_result.success:
        log.error("Order failed: %s", order_result.message)
        send_warning(f"Order failed: {order_result.message}")
        return

    ticket = order_result.ticket
    state.open_tickets.append(ticket)
    log.info("Order placed: ticket=%d  %s  entry=%.2f  SL=%.2f  TP=%.2f  lot=%.2f",
             ticket, setup.direction, setup.entry, setup.sl, setup.tp, lot)

    # Telegram alert
    send_trade_alert(trade_data, screenshot_path=entry_img)

    # Obsidian note
    try:
        create_trade_note(trade_data, screenshot_paths)
    except Exception as exc:
        log.warning("Obsidian note failed: %s", exc)


def _monitor_positions() -> None:
    """Check open trades for BE trigger and closed trades."""
    if not state.open_tickets:
        return

    try:
        open_trades = monitor_open_trades(SYMBOL, MAGIC_NUMBER)
        open_tix    = {t.ticket for t in open_trades}
    except Exception as exc:
        log.warning("monitor_open_trades error: %s", exc)
        return

    # Detect newly closed trades
    still_open = []
    for ticket in state.open_tickets:
        if ticket not in open_tix:
            _handle_closed_trade(ticket)
        else:
            still_open.append(ticket)
    state.open_tickets = still_open


def _handle_closed_trade(ticket: int) -> None:
    """Pull result from history, screenshot, update Obsidian, send Telegram."""
    log.info("Position closed: ticket=%d", ticket)

    cr = get_trade_result(ticket, SYMBOL)
    if cr is None:
        log.warning("Could not fetch result for ticket=%d", ticket)
        return

    result   = cr.result
    pnl_val  = cr.pnl
    duration = cr.duration

    if result == "WIN":
        state.wins_today   += 1
    elif result == "LOSS":
        state.losses_today += 1
    state.day_pnl += pnl_val

    # Outcome screenshot
    trade_id = _find_trade_id_by_ticket(ticket)
    try:
        outcome_img = capture_chart(trade_id or str(ticket), "outcome")
    except Exception as exc:
        log.warning("Outcome screenshot failed: %s", exc)
        outcome_img = None

    # Update Obsidian
    if trade_id:
        try:
            update_trade_outcome(trade_id, {
                "result":      result,
                "pnl":         f"${pnl_val:+.2f}",
                "duration":    duration,
                "exit_price":  f"{cr.exit:.2f}" if cr.exit else "--",
            })
        except Exception as exc:
            log.warning("Obsidian update failed: %s", exc)

    # Telegram result
    send_trade_result(
        trade_data={
            "direction":       "LONG",   # stored in state; simplified here
            "entry":           f"{cr.entry:.2f}",
            "exit_price":      f"{cr.exit:.2f}" if cr.exit else "--",
            "result":          result,
            "pnl":             f"${pnl_val:+.2f}",
            "duration":        duration,
            "wins_today":      state.wins_today,
            "losses_today":    state.losses_today,
            "total_pnl_today": f"${state.day_pnl:+.2f}",
        },
        screenshot_path=outcome_img,
    )


# --- Helpers -----------------------------------------------------------------

def _ensure_connected() -> bool:
    """Ping MT5; reconnect if needed. Returns True if connected."""
    if mt5.terminal_info() is not None:
        return True
    log.warning("MT5 connection lost -- reconnecting...")
    if _connect_with_retry():
        return True
    return False


def _is_eod(now: datetime) -> bool:
    """True at exactly 01:00 AM NPT and only once per day."""
    date_str = now.strftime("%Y-%m-%d")
    if now.hour == EOD_HOUR_NPT and now.minute < 1 and state.last_eod_date != date_str:
        return True
    return False


def _end_of_day(now: datetime) -> None:
    """Daily wrap-up: summary telegram, refresh bias, sleep until 05:45."""
    date_str = now.strftime("%Y-%m-%d")
    state.last_eod_date = date_str
    log.info("End of day -- sending daily summary.")

    total   = state.wins_today + state.losses_today
    wr      = (state.wins_today / total * 100) if total else 0.0
    send_daily_summary({
        "date":      date_str,
        "trades":    total,
        "wins":      state.wins_today,
        "losses":    state.losses_today,
        "breakevens":state.trades_today - total,
        "total_pnl": f"${state.day_pnl:+.2f}",
        "win_rate":  wr,
        "best_rr":   "--",
        "worst_pnl": "--",
    })

    # Reset daily counters
    state.wins_today    = 0
    state.losses_today  = 0
    state.day_pnl       = 0.0
    state.trade_counter = 0
    state.initial_balance = mt5.account_info().balance if mt5.account_info() else state.initial_balance

    # Sleep until 05:45 AM NPT (next trading day start)
    now       = get_current_npt()
    wake_npt  = now.replace(hour=5, minute=45, second=0, microsecond=0)
    if wake_npt <= now:
        wake_npt += timedelta(days=1)
    sleep_sec = (wake_npt - now).total_seconds()
    log.info("Sleeping %.0f min until %s NPT", sleep_sec / 60,
             wake_npt.strftime("%I:%M %p"))
    disconnect()
    state.connected = False
    time.sleep(sleep_sec)
    _connect_with_retry()
    _refresh_session_levels()


def _shutdown() -> None:
    """Graceful shutdown."""
    log.info("Shutting down bot...")
    if state.open_tickets:
        log.warning("Open positions at shutdown: %s", state.open_tickets)
    send_warning("Bot shutting down (manual stop).")
    disconnect()


def _bias_to_dict(bias: BiasResult) -> dict:
    return {
        "technical_bias":  bias.technical_bias,
        "news_bias":       bias.news_bias,
        "dxy_bias":        bias.dxy_bias,
        "divergence":      bias.divergence,
        "technical_score": bias.technical_score,
        "news_score":      bias.news_score,
        "dxy_score":       bias.dxy_score,
        "final_score":     bias.final_score,
        "signal":          bias.signal,
    }


def _session_to_strategy(session: str) -> str:
    return {
        "ASIAN_SWEEP_WINDOW":  "Asian Sweep (Strategy 1)",
        "NY_AM_SILVER_BULLET": "ICT Silver Bullet (NY AM)",
        "NY_PM_SILVER_BULLET": "ICT Silver Bullet (NY PM)",
    }.get(session, session)


_trade_id_map: dict[int, str] = {}   # ticket -> trade_id

def _find_trade_id_by_ticket(ticket: int) -> Optional[str]:
    return _trade_id_map.get(ticket)


# ===========================================================================
# CONSOLE DASHBOARD
# ===========================================================================

def _print_dashboard(now: datetime, session: str) -> None:
    acct        = mt5.account_info()
    balance_str = f"${acct.balance:,.2f}" if acct else "N/A"
    pnl_sign    = "+" if state.day_pnl >= 0 else ""
    pnl_str     = f"{pnl_sign}${state.day_pnl:.2f}"

    if session in TRADING_WINDOWS:
        session_status = f"{session}  [ACTIVE]"
    elif session == "ASIAN_RANGE":
        session_status = "ASIAN RANGE  [marking]"
    else:
        next_win, mins = get_minutes_until_next_window()
        h, m = divmod(mins, 60)
        session_status = f"NONE  -- next: {next_win} in {h}h {m}m"

    bias_str   = "N/A"
    signal_str = "N/A"
    if state.bias:
        sign = "+" if state.bias.final_score >= 0 else ""
        bias_str   = f"{sign}{state.bias.final_score:.2f}"
        signal_str = state.bias.signal

    risk_flag  = "  [HIGH RISK DAY]" if is_high_risk_day() else ""
    mode_flag  = "  [PAPER]" if PAPER_TRADE_MODE else "  [LIVE]"

    open_count = len(state.open_tickets)

    print()
    print("=" * 50)
    print(f"  XAUUSD Silver Bullet Bot  v{VERSION}{mode_flag}")
    print(f"  {now.strftime('%Y-%m-%d  %I:%M:%S %p')} NPT{risk_flag}")
    print(f"  {'-' * 46}")
    print(f"  Session  : {session_status}")
    print(f"  Bias     : {signal_str}  ({bias_str})")
    print(f"  Trades   : {state.trades_today}/2  |  P&L: {pnl_str}")
    print(f"  Balance  : {balance_str}")
    print(f"  Open Pos : {open_count}")
    print("=" * 50)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    if not startup():
        log.critical("Startup failed -- exiting.")
        sys.exit(1)

    run_loop()
