# ─── Trading Symbol Configuration ───────────────────────────────────────────
SYMBOL      = "XAUUSD"
DXY_SYMBOL  = "USDX"   # Try "DX" if your broker uses a different ticker

# ─── Order Configuration ─────────────────────────────────────────────────────
MAGIC_NUMBER = 234567
DEVIATION    = 10       # Max price deviation in points

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"

# ─── Candle History ──────────────────────────────────────────────────────────
CANDLE_COUNT = 100

# ─── Session Times (UTC) ─────────────────────────────────────────────────────
# All times are UTC — convert to NPT by adding +5:45
SESSIONS = {
    "sydney":   {"open": "22:00", "close": "07:00"},  # UTC (prev day open)
    "tokyo":    {"open": "00:00", "close": "09:00"},  # UTC
    "london":   {"open": "08:00", "close": "16:00"},  # UTC
    "new_york": {"open": "13:00", "close": "21:00"},  # UTC
}

# ─── High-Impact News Blackout (minutes before/after) ────────────────────────
NEWS_BLACKOUT_MINUTES = 30

# --- Execution Mode ----------------------------------------------------------
# PAPER_TRADE_MODE = True  -> log orders only, nothing sent to MT5
# PAPER_TRADE_MODE = False -> live execution (only after full testing)
PAPER_TRADE_MODE = True
