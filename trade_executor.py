# -*- coding: utf-8 -*-
"""
trade_executor.py
-----------------
Places, closes, and monitors MT5 orders for the XAUUSD bot.

SAFETY FLAGS
------------
  PAPER_TRADE_MODE = True  (set in config.py)
      All order logic runs but nothing is sent to MT5.
      Every action is logged as "PAPER TRADE: Would ...".

  PAPER_TRADE_MODE = False
      Live execution.  Only enable after full paper testing.

Account notes (MetaQuotes-Demo):
  - NETTING account  (one position per symbol at a time)
  - Filling mode: FOK | IOC  (RETURN not supported)
  - point = $0.01,  1 pip = 10 points = $0.10
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import MetaTrader5 as mt5

from config import (
    SYMBOL, MAGIC_NUMBER, DEVIATION,
    LOG_LEVEL, PAPER_TRADE_MODE, BREAKEVEN_TRIGGER,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

UTC = timezone.utc

# MT5 filling mode: prefer IOC (2), fallback FOK (1)
_FILLING = mt5.ORDER_FILLING_IOC


# --- Data classes ------------------------------------------------------------

@dataclass
class OrderResult:
    success:  bool
    ticket:   int   = 0
    order:    int   = 0      # MT5 order id (before it becomes a deal)
    message:  str   = ""
    paper:    bool  = False  # True when PAPER_TRADE_MODE was active

    def __str__(self) -> str:
        mode = " [PAPER]" if self.paper else ""
        if self.success:
            return f"OrderResult(OK{mode}  ticket={self.ticket}  {self.message})"
        return f"OrderResult(FAIL{mode}  {self.message})"


@dataclass
class TradeInfo:
    ticket:        int
    symbol:        str
    direction:     str      # "LONG" or "SHORT"
    entry:         float
    sl:            float
    tp:            float
    lot:           float
    open_time:     datetime
    current_price: float    = 0.0
    current_pnl:   float    = 0.0
    pnl_pct_to_tp: float    = 0.0   # 0.0 -> 1.0 (how far to TP)


@dataclass
class ClosedTradeResult:
    ticket:     int
    entry:      float
    exit:       float
    pnl:        float
    duration:   str
    result:     str        # "WIN" | "LOSS" | "BE"
    lot:        float
    open_time:  Optional[datetime] = None
    close_time: Optional[datetime] = None


# --- MT5 error decoder -------------------------------------------------------

def _mt5_err() -> str:
    code, msg = mt5.last_error()
    return f"[{code}] {msg}"


# --- 1. Place limit order ----------------------------------------------------

def place_limit_order(
    symbol:    str   = SYMBOL,
    direction: str   = "LONG",
    entry:     float = 0.0,
    sl:        float = 0.0,
    tp:        float = 0.0,
    lot_size:  float = 0.01,
    magic:     int   = MAGIC_NUMBER,
    comment:   str   = "xauusd_bot",
) -> OrderResult:
    """
    Place a BUY_LIMIT or SELL_LIMIT pending order on MT5.

    Parameters
    ----------
    direction : "LONG" or "SHORT"
    entry     : limit price to enter
    sl        : stop-loss price
    tp        : take-profit price
    lot_size  : position size in lots

    Returns
    -------
    OrderResult with ticket on success.
    """
    direction = direction.upper()
    order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "LONG" else mt5.ORDER_TYPE_SELL_LIMIT
    type_label = "BUY_LIMIT" if direction == "LONG" else "SELL_LIMIT"

    # --- Validate prices -----------------------------------------------------
    if sl <= 0 or tp <= 0 or entry <= 0:
        return OrderResult(False, message="Invalid price: entry/sl/tp must be > 0")

    if direction == "LONG":
        if sl >= entry:
            return OrderResult(False, message=f"LONG: SL ({sl}) must be below entry ({entry})")
        if tp <= entry:
            return OrderResult(False, message=f"LONG: TP ({tp}) must be above entry ({entry})")
        tick = mt5.symbol_info_tick(symbol)
        if tick and entry >= tick.ask:
            return OrderResult(False,
                message=f"LONG limit entry ({entry}) must be below current ask ({tick.ask:.2f})")
    else:
        if sl <= entry:
            return OrderResult(False, message=f"SHORT: SL ({sl}) must be above entry ({entry})")
        if tp >= entry:
            return OrderResult(False, message=f"SHORT: TP ({tp}) must be below entry ({entry})")
        tick = mt5.symbol_info_tick(symbol)
        if tick and entry <= tick.bid:
            return OrderResult(False,
                message=f"SHORT limit entry ({entry}) must be above current bid ({tick.bid:.2f})")

    # --- Paper trade ---------------------------------------------------------
    if PAPER_TRADE_MODE:
        msg = (
            f"PAPER TRADE: Would place {type_label}"
            f"  symbol={symbol}"
            f"  entry={entry:.2f}"
            f"  SL={sl:.2f}"
            f"  TP={tp:.2f}"
            f"  lot={lot_size}"
            f"  magic={magic}"
        )
        log.info(msg)
        fake_ticket = int(datetime.now(UTC).timestamp())
        return OrderResult(True, ticket=fake_ticket, message=msg, paper=True)

    # --- Live order ----------------------------------------------------------
    request = {
        "action":      mt5.TRADE_ACTION_PENDING,
        "symbol":      symbol,
        "volume":      lot_size,
        "type":        order_type,
        "price":       entry,
        "sl":          sl,
        "tp":          tp,
        "deviation":   DEVIATION,
        "magic":       magic,
        "comment":     comment,
        "type_time":   mt5.ORDER_TIME_GTC,
        "type_filling":_FILLING,
    }

    result = mt5.order_send(request)
    if result is None:
        return OrderResult(False, message=f"order_send returned None: {_mt5_err()}")

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return OrderResult(
            False,
            message=(
                f"order_send failed: retcode={result.retcode}"
                f"  comment={result.comment}"
                f"  error={_mt5_err()}"
            ),
        )

    log.info(
        "Order placed: ticket=%d  %s  %s  entry=%.2f  SL=%.2f  TP=%.2f  lot=%s",
        result.order, type_label, symbol, entry, sl, tp, lot_size,
    )
    return OrderResult(True, ticket=result.order, order=result.order,
                       message=f"retcode={result.retcode} comment={result.comment}")


# --- 2. Close trade ----------------------------------------------------------

def close_trade(
    ticket: int,
    symbol: str  = SYMBOL,
    magic:  int  = MAGIC_NUMBER,
) -> OrderResult:
    """
    Close an open position by ticket number.

    For NETTING accounts the position is closed with an opposite
    market order (no position ticket needed in the request).
    """
    if PAPER_TRADE_MODE:
        msg = f"PAPER TRADE: Would close ticket={ticket} on {symbol}"
        log.info(msg)
        return OrderResult(True, ticket=ticket, message=msg, paper=True)

    # Find the position
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return OrderResult(False, message=f"No open positions found for {symbol}")

    # On a netting account there is only one position per symbol.
    # Match by ticket when possible.
    pos = next((p for p in positions if p.ticket == ticket), positions[0])

    close_type  = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick        = mt5.symbol_info_tick(symbol)
    close_price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

    request = {
        "action":      mt5.TRADE_ACTION_DEAL,
        "symbol":      symbol,
        "volume":      pos.volume,
        "type":        close_type,
        "price":       close_price,
        "deviation":   DEVIATION,
        "magic":       magic,
        "comment":     "close_by_bot",
        "type_filling":_FILLING,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else "None"
        return OrderResult(False, message=f"Close failed: retcode={code}  {_mt5_err()}")

    log.info("Position closed: ticket=%d  symbol=%s  deal=%d", ticket, symbol, result.deal)
    return OrderResult(True, ticket=ticket, order=result.order,
                       message=f"deal={result.deal}")


# --- 3. Move SL to breakeven -------------------------------------------------

def modify_sl_to_breakeven(
    ticket:      int,
    entry_price: float,
    symbol:      str = SYMBOL,
) -> OrderResult:
    """
    Move the stop-loss to the entry price (breakeven).

    Only modifies if the current SL is not already at or beyond entry.
    Works on both pending orders and open positions.
    """
    if PAPER_TRADE_MODE:
        msg = f"PAPER TRADE: Would move SL to breakeven={entry_price:.2f}  ticket={ticket}"
        log.info(msg)
        return OrderResult(True, ticket=ticket, message=msg, paper=True)

    # Try positions first
    positions = mt5.positions_get(symbol=symbol)
    pos = None
    if positions:
        pos = next((p for p in positions if p.ticket == ticket), None)

    if pos:
        # Already at or better than BE?
        if pos.type == mt5.POSITION_TYPE_BUY and pos.sl >= entry_price:
            return OrderResult(True, ticket=ticket, message="SL already at/above entry (LONG)")
        if pos.type == mt5.POSITION_TYPE_SELL and pos.sl <= entry_price and pos.sl > 0:
            return OrderResult(True, ticket=ticket, message="SL already at/below entry (SHORT)")

        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   symbol,
            "sl":       entry_price,
            "tp":       pos.tp,
            "position": pos.ticket,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else "None"
            return OrderResult(False, message=f"Modify SL failed: retcode={code}  {_mt5_err()}")

        log.info("SL moved to breakeven: ticket=%d  entry=%.2f", ticket, entry_price)
        return OrderResult(True, ticket=ticket,
                           message=f"SL set to breakeven={entry_price:.2f}")

    return OrderResult(False, message=f"Position ticket={ticket} not found")


# --- 4. Monitor open trades --------------------------------------------------

def monitor_open_trades(
    symbol: str = SYMBOL,
    magic:  int = MAGIC_NUMBER,
    be_trigger_pct: float = BREAKEVEN_TRIGGER,
) -> list[TradeInfo]:
    """
    Scan all open positions, calculate progress to TP, and
    automatically move SL to breakeven when the trigger is reached.

    Parameters
    ----------
    be_trigger_pct : float
        When current price has moved this fraction of the way from
        entry to TP, SL is moved to entry (breakeven).
        Defaults to BREAKEVEN_TRIGGER from config.py.

    Returns
    -------
    List of TradeInfo for every open position on this symbol.
    """
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        log.debug("No open positions for %s", symbol)
        return []

    tick = mt5.symbol_info_tick(symbol)
    current_price = (tick.bid + tick.ask) / 2 if tick else 0.0

    results: list[TradeInfo] = []

    for pos in positions:
        if pos.magic != magic:
            continue

        direction = "LONG" if pos.type == mt5.POSITION_TYPE_BUY else "SHORT"
        entry     = pos.price_open
        sl        = pos.sl
        tp        = pos.tp
        pnl       = pos.profit

        # Progress toward TP (0.0 = at entry, 1.0 = at TP)
        tp_distance   = abs(tp - entry)   if tp   else 0
        move_distance = abs(current_price - entry)
        pnl_pct       = (move_distance / tp_distance) if tp_distance > 0 else 0.0

        # Is it moving in the right direction?
        if direction == "LONG":
            in_profit = current_price > entry
        else:
            in_profit = current_price < entry

        if not in_profit:
            pnl_pct = 0.0

        info = TradeInfo(
            ticket        = pos.ticket,
            symbol        = pos.symbol,
            direction     = direction,
            entry         = entry,
            sl            = sl,
            tp            = tp,
            lot           = pos.volume,
            open_time     = datetime.fromtimestamp(pos.time, tz=UTC),
            current_price = current_price,
            current_pnl   = pnl,
            pnl_pct_to_tp = round(min(pnl_pct, 1.0), 3),
        )
        results.append(info)

        # Auto breakeven
        sl_already_be = (
            (direction == "LONG"  and sl >= entry) or
            (direction == "SHORT" and sl <= entry and sl > 0)
        )
        if pnl_pct >= be_trigger_pct and not sl_already_be:
            log.info(
                "BE trigger: ticket=%d %s %.0f%% to TP -- moving SL to entry %.2f",
                pos.ticket, direction, pnl_pct * 100, entry,
            )
            modify_sl_to_breakeven(pos.ticket, entry, symbol)

    return results


# --- 5. Get trade result from history ----------------------------------------

def get_trade_result(
    ticket: int,
    symbol: str = SYMBOL,
) -> Optional[ClosedTradeResult]:
    """
    Fetch a closed trade's data from MT5 deal history.

    Searches the last 90 days for deals matching the ticket.
    Returns ClosedTradeResult or None if not found.
    """
    if PAPER_TRADE_MODE:
        log.info("PAPER TRADE: get_trade_result called for ticket=%d", ticket)
        return ClosedTradeResult(
            ticket=ticket, entry=0.0, exit=0.0,
            pnl=0.0, duration="--", result="PAPER",
            lot=0.01,
        )

    from_dt = datetime.now(UTC) - timedelta(days=90)
    to_dt   = datetime.now(UTC)

    deals = mt5.history_deals_get(from_dt, to_dt)
    if deals is None:
        log.error("history_deals_get failed: %s", _mt5_err())
        return None

    # Filter deals for this symbol that belong to this order chain
    matching = [d for d in deals if d.symbol == symbol and d.order == ticket]
    if not matching:
        # Try matching by position id (netting)
        matching = [d for d in deals if d.symbol == symbol and d.position_id == ticket]

    if not matching:
        log.warning("No deals found for ticket=%d", ticket)
        return None

    entry_deal = next((d for d in matching if d.entry == mt5.DEAL_ENTRY_IN),  None)
    exit_deal  = next((d for d in matching if d.entry == mt5.DEAL_ENTRY_OUT), None)

    if not entry_deal:
        return None

    entry_price = entry_deal.price
    exit_price  = exit_deal.price if exit_deal else 0.0
    pnl         = exit_deal.profit if exit_deal else 0.0

    open_time  = datetime.fromtimestamp(entry_deal.time, tz=UTC) if entry_deal else None
    close_time = datetime.fromtimestamp(exit_deal.time,  tz=UTC) if exit_deal  else None

    if open_time and close_time:
        diff    = close_time - open_time
        minutes = int(diff.total_seconds() / 60)
        hours   = minutes // 60
        mins    = minutes % 60
        duration = f"{hours}h {mins}m" if hours else f"{mins} min"
    else:
        duration = "--"

    result = "OPEN"
    if exit_deal:
        result = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BE")

    return ClosedTradeResult(
        ticket     = ticket,
        entry      = entry_price,
        exit       = exit_price,
        pnl        = pnl,
        duration   = duration,
        result     = result,
        lot        = entry_deal.volume,
        open_time  = open_time,
        close_time = close_time,
    )


# --- Pretty printer ----------------------------------------------------------

def _print_trade_info(t: TradeInfo) -> None:
    bar = int(t.pnl_pct_to_tp * 20)
    progress = "[" + "#" * bar + "." * (20 - bar) + "]"
    pnl_sign = "+" if t.current_pnl >= 0 else ""
    print(
        f"  ticket={t.ticket}  {t.direction}  entry={t.entry:.2f}"
        f"  SL={t.sl:.2f}  TP={t.tp:.2f}  lot={t.lot}"
        f"\n  price={t.current_price:.2f}  P&L={pnl_sign}{t.current_pnl:.2f}"
        f"  TP progress {progress} {t.pnl_pct_to_tp*100:.0f}%"
    )


# --- Entry Point -------------------------------------------------------------

if __name__ == "__main__":
    from mt5_connector import connect, disconnect

    print("=" * 60)
    print(f"  Trade Executor - XAUUSD Bot")
    print(f"  Mode: {'PAPER TRADE (simulation)' if PAPER_TRADE_MODE else 'LIVE EXECUTION'}")
    print("=" * 60)

    if not connect():
        raise SystemExit("Could not connect to MT5.")

    try:
        tick = mt5.symbol_info_tick(SYMBOL)
        ask  = tick.ask
        bid  = tick.bid
        print(f"\nCurrent price  Ask={ask:.2f}  Bid={bid:.2f}")

        # Simulate a realistic setup: entry below ask for a long
        test_entry = round(ask - 5.00, 2)   # 5 pips below current ask
        test_sl    = round(test_entry - 15.00, 2)
        test_tp    = round(test_entry + 50.00, 2)
        test_lot   = 0.02

        print(f"\n[1] LONG limit order (PAPER)")
        print(f"    Entry={test_entry}  SL={test_sl}  TP={test_tp}  Lot={test_lot}")
        r = place_limit_order(
            symbol    = SYMBOL,
            direction = "LONG",
            entry     = test_entry,
            sl        = test_sl,
            tp        = test_tp,
            lot_size  = test_lot,
        )
        print(f"    Result: {r}")
        fake_ticket = r.ticket

        print(f"\n[2] SHORT limit order (PAPER)")
        s_entry = round(bid + 5.00, 2)
        s_sl    = round(s_entry + 15.00, 2)
        s_tp    = round(s_entry - 50.00, 2)
        r2 = place_limit_order(
            symbol    = SYMBOL,
            direction = "SHORT",
            entry     = s_entry,
            sl        = s_sl,
            tp        = s_tp,
            lot_size  = test_lot,
        )
        print(f"    Result: {r2}")

        print(f"\n[3] modify_sl_to_breakeven (PAPER)")
        r3 = modify_sl_to_breakeven(fake_ticket, test_entry)
        print(f"    Result: {r3}")

        print(f"\n[4] close_trade (PAPER)")
        r4 = close_trade(fake_ticket)
        print(f"    Result: {r4}")

        print(f"\n[5] monitor_open_trades (live MT5 positions)")
        trades = monitor_open_trades()
        if trades:
            for t in trades:
                _print_trade_info(t)
        else:
            print("    No open positions on this account.")

        print(f"\n[6] get_trade_result for fake ticket (PAPER mode)")
        cr = get_trade_result(fake_ticket)
        print(f"    Result: {cr}")

        print(f"\n[7] Invalid order validation checks")
        bad_cases = [
            ("LONG",  test_entry, test_entry,      test_tp,    "SL == entry"),
            ("LONG",  test_entry, test_entry + 1,  test_tp,    "SL above entry (LONG)"),
            ("SHORT", s_entry,    s_entry,          s_tp,       "SL == entry"),
            ("LONG",  0,          test_sl,          test_tp,    "zero entry price"),
        ]
        for direction, entry, sl, tp, label in bad_cases:
            rv = place_limit_order(SYMBOL, direction, entry, sl, tp, 0.01)
            icon = "[ OK ]" if not rv.success else "[PASS]"
            print(f"    {icon}  {label:<30}  rejected={not rv.success}  msg={rv.message[:60]}")

    finally:
        disconnect()

    print("\n  All PAPER TRADE tests complete.")
    print("  Set PAPER_TRADE_MODE = False in config.py only after live testing.")
    print("=" * 60)
