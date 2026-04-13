"""
session_manager.py
------------------
All time logic for the XAUUSD bot.
Nepal Time (NPT) = UTC + 5:45

Trading Windows (all times NPT):
  ASIAN_RANGE          05:45 - 12:45  (range marking only, no trades)
  ASIAN_SWEEP_WINDOW   12:45 - 13:45  (Strategy 1)
  NY_AM_WINDOW         19:45 - 20:45  (Strategy 2 - Silver Bullet)
  NY_PM_WINDOW         23:45 - 00:45  (Strategy 3 - Silver Bullet, crosses midnight)
"""

from __future__ import annotations
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

# --- Timezone -----------------------------------------------------------------
NPT = timezone(timedelta(hours=5, minutes=45))   # Nepal has no ZoneInfo entry

# --- Window Definitions (start, end) as (hour, minute) tuples in NPT ---------
WINDOWS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "ASIAN_RANGE":          ((5, 45),  (12, 45)),
    "ASIAN_SWEEP_WINDOW":   ((12, 45), (13, 45)),
    "NY_AM_SILVER_BULLET":  ((19, 45), (20, 45)),
    "NY_PM_SILVER_BULLET":  ((23, 45), (0, 45)),   # crosses midnight
}

# Windows where actual trading happens (not just range marking)
TRADING_WINDOWS = {"ASIAN_SWEEP_WINDOW", "NY_AM_SILVER_BULLET", "NY_PM_SILVER_BULLET"}

# Days considered lower quality (0=Monday, 4=Friday)
HIGH_RISK_WEEKDAYS = {0, 4}


# --- Helpers ------------------------------------------------------------------

def _to_time(h: int, m: int) -> time:
    return time(h, m)


def _minutes_between(now: datetime, target_h: int, target_m: int) -> int:
    """
    Returns positive minutes from `now` (NPT-aware) until the next occurrence
    of (target_h:target_m) NPT, always looking forward (wraps over midnight).
    """
    today = now.date()
    candidate = datetime(today.year, today.month, today.day,
                         target_h, target_m, tzinfo=NPT)
    if candidate <= now:
        candidate += timedelta(days=1)
    delta = candidate - now
    return int(delta.total_seconds() // 60)


def _window_contains(h_start: int, m_start: int,
                     h_end: int, m_end: int,
                     now_time: time) -> bool:
    """
    Check if `now_time` falls inside [start, end).
    Handles windows that cross midnight (end < start).
    """
    start = _to_time(h_start, m_start)
    end   = _to_time(h_end, m_end)

    if start < end:                         # normal window (no midnight crossing)
        return start <= now_time < end
    else:                                   # crosses midnight
        return now_time >= start or now_time < end


def _minutes_to_window_end(h_start: int, m_start: int,
                            h_end: int, m_end: int,
                            now: datetime) -> int:
    """Minutes remaining until this window closes (assuming now is inside it)."""
    today = now.date()
    end_dt = datetime(today.year, today.month, today.day,
                      h_end, m_end, tzinfo=NPT)

    # If the window crosses midnight and end is "tomorrow" relative to start
    start = _to_time(h_start, m_start)
    end   = _to_time(h_end, m_end)
    if start > end and now.time() >= start:   # we're in the pre-midnight half
        end_dt += timedelta(days=1)

    delta = end_dt - now
    return max(0, int(delta.total_seconds() // 60))


# --- Public API ---------------------------------------------------------------

def get_current_npt() -> datetime:
    """Return the current datetime in Nepal Time."""
    return datetime.now(tz=NPT)


def get_active_session() -> str:
    """
    Return the name of the currently active window, or 'NONE'.
    If multiple windows overlap (shouldn't happen by design), returns the first match.
    """
    now = get_current_npt()
    t   = now.time()
    for name, ((hs, ms), (he, me)) in WINDOWS.items():
        if _window_contains(hs, ms, he, me, t):
            return name
    return "NONE"


def is_window_active(window_name: str) -> bool:
    """Return True if the named window is currently active."""
    if window_name not in WINDOWS:
        raise ValueError(f"Unknown window: {window_name!r}. "
                         f"Valid options: {list(WINDOWS)}")
    return get_active_session() == window_name


def get_minutes_until_next_window() -> tuple[str, int]:
    """
    Return (window_name, minutes) for the next trading window to open.
    Skips windows that are currently active.
    Active means it opens now or in the past (but not yet closed).
    Returns the soonest upcoming window start.
    """
    now    = get_current_npt()
    active = get_active_session()

    candidates: list[tuple[int, str]] = []
    for name, ((hs, ms), _) in WINDOWS.items():
        if name not in TRADING_WINDOWS:
            continue
        if name == active:
            continue
        mins = _minutes_between(now, hs, ms)
        candidates.append((mins, name))

    candidates.sort()
    return candidates[0][1], candidates[0][0]


def is_high_risk_day() -> bool:
    """Return True if today (NPT) is Monday or Friday (lower-quality trading days)."""
    return get_current_npt().weekday() in HIGH_RISK_WEEKDAYS


# --- Status Printer -----------------------------------------------------------

def print_status() -> None:
    now    = get_current_npt()
    active = get_active_session()
    risk   = is_high_risk_day()
    day    = now.strftime("%A")

    npt_str = now.strftime("%I:%M %p")   # e.g.  07:52 PM

    if active != "NONE":
        (hs, ms), (he, me) = WINDOWS[active]
        mins_left = _minutes_to_window_end(hs, ms, he, me, now)
        trading   = active in TRADING_WINDOWS
        kind      = "TRADE" if trading else "RANGE MARKING"
        print(
            f"Current NPT: {npt_str}  |  "
            f"Active Window: {active}  |  "
            f"Type: {kind}  |  "
            f"Minutes to close: {mins_left}"
        )
    else:
        next_win, mins_away = get_minutes_until_next_window()
        h_next = mins_away // 60
        m_next = mins_away % 60
        print(
            f"Current NPT: {npt_str}  |  "
            f"Active Window: NONE  |  "
            f"Next: {next_win} in {h_next}h {m_next}m"
        )

    risk_label = f"  [!] HIGH-RISK DAY ({day})" if risk else f"  [ ] Normal day ({day})"
    print(risk_label)
    print()

    # Full session overview
    print("  Window overview (NPT):")
    for name, ((hs, ms), (he, me)) in WINDOWS.items():
        marker = " <-- ACTIVE" if name == active else ""
        tag    = "[TRADE]" if name in TRADING_WINDOWS else "[RANGE]"
        print(f"    {tag}  {name:<26}  "
              f"{hs:02d}:{ms:02d} - {he:02d}:{me:02d}{marker}")


# --- Entry Point --------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  Session Manager  -  XAUUSD Bot")
    print("=" * 65)
    print_status()
