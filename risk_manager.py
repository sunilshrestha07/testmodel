# -*- coding: utf-8 -*-
"""
risk_manager.py
---------------
All position-sizing and capital-protection logic for the XAUUSD bot.

XAUUSD pip reference:
  1 pip       = $0.10 price movement  (10 points, point = $0.01)
  pip_value   = $10.00 per pip per 1.0 lot  (100 oz x $0.10)
  spread      = mt5.symbol_info().spread / 10  (points -> pips)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import MetaTrader5 as mt5

from config import SYMBOL, LOG_LEVEL

log = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- XAUUSD pip constants -----------------------------------------------------
PIP_POINTS        = 10          # 10 MT5 points = 1 pip for XAUUSD
PIP_VALUE_PER_LOT = 10.0        # $10 per pip per 1.0 lot
POINT_SIZE        = 0.01        # smallest price increment

# --- Hard limits (configurable here, not in config.py) ------------------------
LOT_HARD_MAX      = 0.10        # never exceed this regardless of calculation
LOT_HARD_MIN      = 0.01        # broker minimum for XAUUSD
DEFAULT_RISK_PCT  = 1.0         # % of balance to risk per trade
MAX_DAILY_DD_PCT  = 2.0         # halt trading if daily loss >= this %
MAX_DAILY_TRADES  = 2           # max entries per calendar day


# --- 1. Lot size calculator ---------------------------------------------------

def calculate_lot_size(
    account_balance: float,
    risk_percent:    float,
    sl_pips:         float,
    lot_modifier:    float = 1.0,
) -> float:
    """
    Calculate position size using fixed fractional risk.

    Formula:
        risk_amount = balance * (risk_percent / 100)
        raw_lot     = risk_amount / (sl_pips * PIP_VALUE_PER_LOT)
        final_lot   = raw_lot * lot_modifier

    Parameters
    ----------
    account_balance : float
        Current account balance in USD.
    risk_percent : float
        Percentage of balance to risk (e.g. 1.0 = 1%).
    sl_pips : float
        Stop-loss distance in pips.  Convert price distance with:
            sl_pips = abs(entry_price - sl_price) / 0.10
    lot_modifier : float
        1.0 for STRONG signal, 0.5 for WEAK signal, 0.0 = no trade.

    Returns
    -------
    float : lot size rounded to 2 d.p., clamped to [LOT_HARD_MIN, LOT_HARD_MAX].
    """
    if sl_pips <= 0:
        log.error("sl_pips must be > 0, got %.2f", sl_pips)
        return 0.0
    if lot_modifier <= 0:
        return 0.0

    risk_amount = account_balance * (risk_percent / 100.0)
    raw_lot     = risk_amount / (sl_pips * PIP_VALUE_PER_LOT)
    final_lot   = raw_lot * lot_modifier

    # Clamp to broker limits and hard cap
    final_lot = max(LOT_HARD_MIN, min(LOT_HARD_MAX, round(final_lot, 2)))

    log.debug(
        "lot_size: balance=%.2f risk=%.1f%% sl=%.1fpips modifier=%.1f -> %.2f lots",
        account_balance, risk_percent, sl_pips, lot_modifier, final_lot,
    )
    return final_lot


# --- 2. Daily drawdown guard --------------------------------------------------

def check_daily_drawdown(
    account_balance:   float,
    initial_balance:   float,
    max_drawdown_percent: float = MAX_DAILY_DD_PCT,
) -> tuple[bool, float]:
    """
    Check whether the daily drawdown limit has been breached.

    Parameters
    ----------
    account_balance   : current equity / balance
    initial_balance   : balance at the start of the trading day
    max_drawdown_percent : daily loss ceiling (default 2%)

    Returns
    -------
    (breached: bool, current_drawdown_pct: float)
        breached=True means trading must halt for the day.
    """
    if initial_balance <= 0:
        log.error("initial_balance must be > 0")
        return True, 0.0

    daily_pnl_pct = ((account_balance - initial_balance) / initial_balance) * 100.0
    drawdown_pct  = -daily_pnl_pct   # positive = a loss

    breached = drawdown_pct >= max_drawdown_percent
    if breached:
        log.warning(
            "Daily drawdown limit reached: %.2f%% (limit: %.2f%%)",
            drawdown_pct, max_drawdown_percent,
        )
    return breached, round(drawdown_pct, 3)


# --- 3. Daily trade-count guard -----------------------------------------------

def check_daily_trade_count(
    trades_today: int,
    max_trades:   int = MAX_DAILY_TRADES,
) -> tuple[bool, int]:
    """
    Check whether the maximum number of trades for today has been reached.

    Returns
    -------
    (limit_reached: bool, remaining: int)
    """
    remaining     = max(0, max_trades - trades_today)
    limit_reached = trades_today >= max_trades
    if limit_reached:
        log.warning("Daily trade limit reached (%d/%d). No more entries today.", trades_today, max_trades)
    return limit_reached, remaining


# --- 4. Spread check ----------------------------------------------------------

def check_spread(
    symbol:          str   = SYMBOL,
    max_spread_pips: float = 3.0,
) -> tuple[bool, float]:
    """
    Fetch the current spread from MT5 and compare against the limit.

    Parameters
    ----------
    symbol          : trading symbol
    max_spread_pips : maximum acceptable spread in pips (default 3.0)

    Returns
    -------
    (acceptable: bool, current_spread_pips: float)
        acceptable=True means spread is within limits.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        log.error("symbol_info() failed for %s: %s", symbol, mt5.last_error())
        return False, 0.0

    spread_pips = info.spread / PIP_POINTS   # points -> pips
    acceptable  = spread_pips <= max_spread_pips

    if not acceptable:
        log.warning(
            "Spread too wide: %.1f pips (limit: %.1f pips)", spread_pips, max_spread_pips
        )
    return acceptable, round(spread_pips, 2)


# --- 5. RR validator ----------------------------------------------------------

def validate_rr(
    entry:  float,
    sl:     float,
    tp:     float,
    min_rr: float = 2.0,
) -> tuple[bool, float]:
    """
    Calculate the actual risk:reward ratio and validate it meets the minimum.

    Returns
    -------
    (valid: bool, rr: float)
        valid=True means the trade meets the minimum RR requirement.
    """
    risk   = abs(entry - sl)
    reward = abs(tp - entry)

    if risk == 0:
        log.error("validate_rr: risk is 0 (entry == sl)")
        return False, 0.0

    rr = round(reward / risk, 2)
    valid = rr >= min_rr

    if not valid:
        log.warning("RR too low: 1:%.2f (minimum: 1:%.1f)", rr, min_rr)
    return valid, rr


# --- 6. Trades-today counter from MT5 history --------------------------------

def get_trades_today() -> int:
    """
    Count closed trades (deals) executed since midnight UTC today.
    Uses MT5 deal history filtered by the bot's MAGIC_NUMBER.
    """
    from config import MAGIC_NUMBER
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    deals = mt5.history_deals_get(today_start, datetime.now(timezone.utc))
    if deals is None:
        return 0
    # Count only entry deals placed by this bot
    return sum(
        1 for d in deals
        if d.magic == MAGIC_NUMBER and d.entry == mt5.DEAL_ENTRY_IN
    )


# --- 7. Full risk report -------------------------------------------------------

def get_risk_report(
    initial_balance: float | None = None,
) -> dict:
    """
    Pull live account data from MT5 and print a one-line status.

    Parameters
    ----------
    initial_balance : balance at the start of the day.
                      If None, uses the current balance (assumes no trades yet).
    """
    acct = mt5.account_info()
    if acct is None:
        log.error("Could not fetch account info: %s", mt5.last_error())
        return {}

    balance     = acct.balance
    equity      = acct.equity
    if initial_balance is None:
        initial_balance = balance

    daily_pnl      = balance - initial_balance
    daily_pnl_pct  = (daily_pnl / initial_balance * 100) if initial_balance else 0.0

    dd_breached, dd_pct = check_daily_drawdown(balance, initial_balance)
    trades_today        = get_trades_today()
    limit_reached, remaining = check_daily_trade_count(trades_today)
    spread_ok, spread_pips   = check_spread(SYMBOL)

    all_clear = (not dd_breached) and (not limit_reached) and spread_ok
    status    = "ACTIVE" if all_clear else "HALTED"

    pnl_sign  = "+" if daily_pnl >= 0 else ""
    dd_str    = f"{dd_pct:.2f}%"
    pnl_str   = f"{pnl_sign}${daily_pnl:.2f} ({pnl_sign}{daily_pnl_pct:.2f}%)"

    print()
    print("=" * 65)
    print("  Risk Manager Report - XAUUSD Bot")
    print("=" * 65)
    print(f"  Balance          : ${balance:,.2f}")
    print(f"  Equity           : ${equity:,.2f}")
    print(f"  Daily P&L        : {pnl_str}")
    print(f"  Drawdown         : {dd_str} (limit: {MAX_DAILY_DD_PCT}%)")
    print(f"  Trades Today     : {trades_today}/{MAX_DAILY_TRADES}  ({remaining} remaining)")
    print(f"  Spread           : {spread_pips:.1f} pips  ({'OK' if spread_ok else 'WIDE'})")
    print(f"  Status           : {status}")
    print("=" * 65)

    return {
        "balance":         balance,
        "equity":          equity,
        "daily_pnl":       daily_pnl,
        "daily_pnl_pct":   round(daily_pnl_pct, 3),
        "drawdown_pct":    dd_pct,
        "dd_breached":     dd_breached,
        "trades_today":    trades_today,
        "limit_reached":   limit_reached,
        "spread_pips":     spread_pips,
        "spread_ok":       spread_ok,
        "status":          status,
    }


# --- Entry Point --------------------------------------------------------------

if __name__ == "__main__":
    from mt5_connector import connect, disconnect

    if not connect():
        raise SystemExit("Could not connect to MT5.")

    try:
        acct = mt5.account_info()
        balance = acct.balance

        # --- Scenario 1: Normal conditions ------------------------------------
        print("\n[Scenario 1] Normal conditions (1% risk, 50-pip SL, strong signal)")
        lot = calculate_lot_size(balance, risk_percent=1.0, sl_pips=50.0, lot_modifier=1.0)
        print(f"  Balance : ${balance:,.2f}")
        print(f"  Lot     : {lot}")

        # --- Scenario 2: Weak signal (half size) ------------------------------
        print("\n[Scenario 2] Weak signal (lot_modifier=0.5)")
        lot2 = calculate_lot_size(balance, risk_percent=1.0, sl_pips=50.0, lot_modifier=0.5)
        print(f"  Lot     : {lot2}")

        # --- Scenario 3: Drawdown guard ---------------------------------------
        print("\n[Scenario 3] Drawdown check")
        scenarios = [
            (balance, balance,          "start of day (0% DD)"),
            (balance * 0.99, balance,   "1% loss"),
            (balance * 0.98, balance,   "2% loss  -> should HALT"),
            (balance * 0.97, balance,   "3% loss  -> should HALT"),
            (balance * 1.01, balance,   "1% gain  -> should be fine"),
        ]
        for curr, init, label in scenarios:
            breached, pct = check_daily_drawdown(curr, init)
            icon = "[HALT]" if breached else "[ OK ]"
            print(f"  {icon}  {label:<32}  drawdown={pct:.2f}%")

        # --- Scenario 4: Trade count ------------------------------------------
        print("\n[Scenario 4] Daily trade count")
        for n_trades in [0, 1, 2, 3]:
            halted, rem = check_daily_trade_count(n_trades)
            icon = "[HALT]" if halted else "[ OK ]"
            print(f"  {icon}  {n_trades} trades today  ({rem} remaining)")

        # --- Scenario 5: RR validation ----------------------------------------
        print("\n[Scenario 5] RR validation (min 2.0)")
        rr_cases = [
            (4760.0, 4750.0, 4800.0, "entry=4760 sl=4750 tp=4800"),
            (4760.0, 4750.0, 4780.0, "entry=4760 sl=4750 tp=4780"),
            (4760.0, 4755.0, 4790.0, "entry=4760 sl=4755 tp=4790"),
        ]
        for entry, sl, tp, label in rr_cases:
            valid, rr = validate_rr(entry, sl, tp)
            icon = "[ OK ]" if valid else "[FAIL]"
            print(f"  {icon}  {label}  -> RR=1:{rr:.1f}")

        # --- Scenario 6: Spread check -----------------------------------------
        print("\n[Scenario 6] Live spread check")
        ok, spread = check_spread(SYMBOL)
        icon = "[ OK ]" if ok else "[WIDE]"
        print(f"  {icon}  Current spread: {spread:.1f} pips")

        # --- Full risk report --------------------------------------------------
        print("\n[Full Risk Report]")
        get_risk_report(initial_balance=balance)

    finally:
        disconnect()
