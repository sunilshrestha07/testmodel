import os
import sys
import logging
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from config import SYMBOL, DXY_SYMBOL, CANDLE_COUNT, LOG_LEVEL

load_dotenv()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _get_credentials() -> tuple[int, str, str]:
    login_raw = os.getenv("MT5_LOGIN", "")
    password  = os.getenv("MT5_PASSWORD", "")
    server    = os.getenv("MT5_SERVER", "")

    if not login_raw or not password or not server:
        raise EnvironmentError(
            "MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER must be set in .env"
        )

    try:
        login = int(login_raw)
    except ValueError:
        raise ValueError(f"MT5_LOGIN must be a number, got: {login_raw!r}")

    return login, password, server


def connect() -> bool:
    login, password, server = _get_credentials()

    if not mt5.initialize():
        log.error("mt5.initialize() failed: %s", mt5.last_error())
        return False

    authorized = mt5.login(login, password=password, server=server)
    if not authorized:
        log.error("mt5.login() failed for account %d on %s: %s", login, server, mt5.last_error())
        mt5.shutdown()
        return False

    log.info("Connected to MT5  |  account: %d  |  server: %s", login, server)
    return True


def disconnect() -> None:
    mt5.shutdown()
    log.info("MT5 connection closed.")


def fetch_candles(symbol: str, timeframe=mt5.TIMEFRAME_H4, count: int = CANDLE_COUNT) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        log.warning("No candles returned for %s  -  error: %s", symbol, mt5.last_error())
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df[["open", "high", "low", "close", "tick_volume", "spread"]]


def fetch_account_info() -> dict:
    info = mt5.account_info()
    if info is None:
        log.error("Failed to fetch account info: %s", mt5.last_error())
        return {}

    return {
        "login":   info.login,
        "name":    info.name,
        "server":  info.server,
        "balance": info.balance,
        "equity":  info.equity,
        "margin":  info.margin,
        "free_margin": info.margin_free,
        "margin_level": info.margin_level,
        "currency": info.currency,
        "leverage": info.leverage,
    }


def _print_candles(df: pd.DataFrame, label: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {label}   -   last {len(df)} H4 candles")
    print(f"{'═'*60}")
    if df.empty:
        print("  [no data]")
    else:
        pd.set_option("display.float_format", "{:.5f}".format)
        pd.set_option("display.max_rows", 10)
        print(df.tail(10).to_string())
        print(f"  ... ({len(df)} rows total)")


def _print_account(info: dict) -> None:
    print(f"\n{'═'*60}")
    print("  Account Info")
    print(f"{'═'*60}")
    if not info:
        print("  [no data]")
        return
    for key, val in info.items():
        print(f"  {key:<16}: {val}")


def run() -> bool:
    if not connect():
        return False

    try:
        xau_df = fetch_candles(SYMBOL)
        _print_candles(xau_df, f"{SYMBOL} (H4)")

        dxy_df = fetch_candles(DXY_SYMBOL)
        if dxy_df.empty:
            log.warning(
                "Could not fetch candles for %s. "
                "Try changing DXY_SYMBOL to 'DX' in config.py.",
                DXY_SYMBOL,
            )
        _print_candles(dxy_df, f"{DXY_SYMBOL} (H4)")

        account = fetch_account_info()
        _print_account(account)

    finally:
        disconnect()

    return True
