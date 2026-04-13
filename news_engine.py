"""
news_engine.py
--------------
Fetches economic calendar events and breaking news to determine
the fundamental bias for gold (XAUUSD).

Data sources:
  1. ForexFactory calendar via faireconomy.media JSON mirror
     (ForexFactory blocks direct scraping; this feed is the same data)
  2. NewsAPI (free tier) for breaking headline sentiment

All event times are compared against UTC; display is in EST (UTC-4 during DST).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from dotenv import load_dotenv

from config import LOG_LEVEL

load_dotenv()
log = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- Constants ----------------------------------------------------------------
FF_CALENDAR_URL  = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
NEWSAPI_KEY      = os.getenv("NEWS_API_KEY", "")

EST = timezone(timedelta(hours=-4))   # EDT (UTC-4); use -5 in winter
UTC = timezone.utc

NEWS_KEYWORDS = ["gold", "XAUUSD", "federal reserve", "inflation", "USD"]

BULLISH_GOLD_WORDS = [
    "rate cut", "weak dollar", "inflation rise", "geopolitical",
    "safe haven", "war", "crisis", "rate cuts", "dovish",
    "recession", "tariff", "trade war", "debt ceiling",
]
BEARISH_GOLD_WORDS = [
    "rate hike", "strong dollar", "hawkish", "nfp beat",
    "dollar surge", "rate hikes", "tightening", "jobs beat",
]


# --- Data class ---------------------------------------------------------------

@dataclass
class EconomicEvent:
    time_utc:  datetime
    name:      str
    currency:  str
    impact:    str          # "High", "Medium", "Low"
    forecast:  str
    previous:  str
    actual:    str = ""

    @property
    def time_est(self) -> str:
        return self.time_utc.astimezone(EST).strftime("%I:%M %p EST")

    @property
    def time_npt(self) -> str:
        npt = timezone(timedelta(hours=5, minutes=45))
        return self.time_utc.astimezone(npt).strftime("%I:%M %p NPT")

    def __str__(self) -> str:
        parts = [f"[{self.impact.upper()}] {self.name} @ {self.time_est}"]
        if self.forecast:
            parts.append(f"Forecast: {self.forecast}")
        if self.previous:
            parts.append(f"Prev: {self.previous}")
        if self.actual:
            parts.append(f"Actual: {self.actual}")
        return "  ".join(parts)


# --- 1. Fetch today's high-impact events -------------------------------------

def _fetch_ff_calendar() -> list[dict]:
    """
    Fetch the ForexFactory weekly calendar from the faireconomy.media mirror.
    Returns raw event dicts on success, empty list on failure.
    """
    try:
        r = requests.get(
            FF_CALENDAR_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        log.warning("ForexFactory calendar fetch failed: %s", exc)
        return []


def get_todays_events(impact_filter: str = "High") -> list[EconomicEvent]:
    """
    Return today's HIGH impact USD events from the ForexFactory calendar.

    Each event has: time (UTC), name, currency, impact, forecast, previous.
    Only events where currency == "USD" and impact matches filter are returned.
    """
    raw = _fetch_ff_calendar()
    if not raw:
        log.warning("No calendar data returned. Returning empty event list.")
        return []

    now_utc   = datetime.now(UTC)
    today_utc = now_utc.date()
    events: list[EconomicEvent] = []

    for item in raw:
        # Skip non-USD or low-impact
        if item.get("country", "") != "USD":
            continue
        if item.get("impact", "") != impact_filter:
            continue

        # Parse ISO-8601 date string (may include timezone offset)
        raw_date = item.get("date", "")
        try:
            event_dt = datetime.fromisoformat(raw_date)
            if event_dt.tzinfo is None:
                event_dt = event_dt.replace(tzinfo=UTC)
            event_dt = event_dt.astimezone(UTC)
        except (ValueError, TypeError):
            log.debug("Could not parse event date: %s", raw_date)
            continue

        if event_dt.date() != today_utc:
            continue

        events.append(EconomicEvent(
            time_utc  = event_dt,
            name      = item.get("title", "Unknown"),
            currency  = item.get("country", ""),
            impact    = item.get("impact", ""),
            forecast  = item.get("forecast", "") or "",
            previous  = item.get("previous", "") or "",
            actual    = item.get("actual",   "") or "",
        ))

    events.sort(key=lambda e: e.time_utc)
    log.info("Found %d high-impact USD events today.", len(events))
    return events


# --- 2. News proximity check -------------------------------------------------

def is_news_in_next_minutes(
    minutes: int = 45,
    events: Optional[list[EconomicEvent]] = None,
) -> tuple[bool, Optional[EconomicEvent]]:
    """
    Check if any high-impact event fires within the next `minutes`.

    Returns (True, event) or (False, None).
    Pass pre-fetched `events` to avoid a duplicate HTTP call.
    """
    if events is None:
        events = get_todays_events()

    now_utc  = datetime.now(UTC)
    cutoff   = now_utc + timedelta(minutes=minutes)

    for event in events:
        if now_utc <= event.time_utc <= cutoff:
            log.info(
                "High-impact news in next %d min: %s @ %s",
                minutes, event.name, event.time_est,
            )
            return True, event

    return False, None


# --- 3. Fundamental bias from a single event ---------------------------------

def get_news_bias_from_event(
    event_name: str,
    actual: str,
    forecast: str,
) -> str:
    """
    Derive gold bias from a single scheduled news event.

    Rules (USD-centric; bad USD news = bullish gold):
      CPI actual < forecast  → BULLISH  (lower inflation = less hawkish)
      CPI actual > forecast  → BEARISH  (higher inflation = more hawkish)
      NFP actual < forecast  → BULLISH  (weak jobs = dovish Fed)
      NFP actual > forecast  → BEARISH  (strong jobs = hawkish Fed)
      FOMC rate hike         → BEARISH
      FOMC rate hold or cut  → BULLISH
      PPI actual < forecast  → BULLISH
      PPI actual > forecast  → BEARISH
      GDP actual < forecast  → BULLISH
      GDP actual > forecast  → BEARISH

    Returns "BULLISH", "BEARISH", or "NEUTRAL".
    """
    name_upper = event_name.upper()

    # -- FOMC decision --------------------------------------------------------
    if "FOMC" in name_upper or "FEDERAL FUNDS" in name_upper or "RATE DECISION" in name_upper:
        actual_l   = actual.lower()
        forecast_l = forecast.lower()
        hike_words = ["hike", "raise", "increase"]
        cut_words  = ["cut", "lower", "decrease", "hold", "unchanged", "pause"]
        if any(w in actual_l for w in hike_words):
            return "BEARISH"
        if any(w in actual_l for w in cut_words):
            return "BULLISH"
        # Fall back to comparing numeric rates
        try:
            a = float(re.sub(r"[^0-9.\-]", "", actual))
            f = float(re.sub(r"[^0-9.\-]", "", forecast))
            return "BEARISH" if a > f else "BULLISH" if a < f else "NEUTRAL"
        except ValueError:
            return "NEUTRAL"

    # -- Events that use actual vs forecast numeric comparison ----------------
    BEARISH_IF_ABOVE = ("CPI", "PPI", "NFP", "NON-FARM", "PAYROLL", "RETAIL SALES",
                        "ISM", "PMI", "GDP", "ADP", "JOLTS")
    BEARISH_IF_BELOW = ()   # events where strong reading = bearish gold

    matched_type = None
    for kw in BEARISH_IF_ABOVE:
        if kw in name_upper:
            matched_type = "BEARISH_IF_ABOVE"
            break

    if matched_type is None:
        return "NEUTRAL"

    # Parse numeric values
    try:
        a = float(re.sub(r"[^0-9.\-]", "", actual))
        f = float(re.sub(r"[^0-9.\-]", "", forecast))
    except ValueError:
        log.debug("Could not parse actual/forecast for %s: %r / %r", event_name, actual, forecast)
        return "NEUTRAL"

    if matched_type == "BEARISH_IF_ABOVE":
        if a > f:
            return "BEARISH"   # stronger-than-expected USD data = bearish gold
        if a < f:
            return "BULLISH"   # weaker-than-expected USD data = bullish gold
    return "NEUTRAL"


# --- 4. Breaking news sentiment -----------------------------------------------

def _score_text(text: str) -> int:
    """Return net sentiment score (+1 per bullish word, -1 per bearish word)."""
    text_lower = text.lower()
    score = 0
    for phrase in BULLISH_GOLD_WORDS:
        if phrase in text_lower:
            score += 1
    for phrase in BEARISH_GOLD_WORDS:
        if phrase in text_lower:
            score -= 1
    return score


def get_breaking_news_sentiment(max_articles: int = 10) -> tuple[str, list[str]]:
    """
    Fetch recent gold-related headlines from NewsAPI and score sentiment.

    Returns ("BULLISH" | "BEARISH" | "NEUTRAL", [headline, ...]).
    """
    if not NEWSAPI_KEY:
        log.warning("NEWS_API_KEY not set  -  skipping breaking news sentiment.")
        return "NEUTRAL", []

    query = " OR ".join(f'"{kw}"' for kw in NEWS_KEYWORDS)

    try:
        r = requests.get(
            NEWSAPI_ENDPOINT,
            params={
                "q":        query,
                "language": "en",
                "sortBy":   "publishedAt",
                "pageSize": max_articles,
                "apiKey":   NEWSAPI_KEY,
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as exc:
        log.warning("NewsAPI request failed: %s", exc)
        return "NEUTRAL", []

    articles = data.get("articles", [])
    if not articles:
        return "NEUTRAL", []

    headlines: list[str] = []
    total_score = 0

    for article in articles:
        title       = article.get("title", "") or ""
        description = article.get("description", "") or ""
        combined    = f"{title}. {description}"
        score       = _score_text(combined)
        total_score += score
        headlines.append(title)
        log.debug("Score %+d | %s", score, title[:80])

    if total_score > 0:
        sentiment = "BULLISH"
    elif total_score < 0:
        sentiment = "BEARISH"
    else:
        sentiment = "NEUTRAL"

    log.info(
        "Breaking news sentiment: %s (score=%+d across %d articles)",
        sentiment, total_score, len(articles),
    )
    return sentiment, headlines


# --- 5. Full news report ------------------------------------------------------

def get_full_news_report() -> dict:
    """
    Combine scheduled calendar events + breaking news into a single bias.

    Precedence:
      1. If a high-impact event fired today with actual vs forecast → use that
      2. Otherwise use breaking news sentiment
      3. If both signals agree → high confidence
      4. If they conflict → NEUTRAL (wait for clarity)

    Prints a human-readable block and returns a result dict.
    """
    print("=" * 70)
    print("  News Engine Report  -  XAUUSD Bot")
    print("=" * 70)

    # -- Scheduled events -----------------------------------------------------
    events = get_todays_events()
    if events:
        print(f"\nToday's High-Impact USD Events ({len(events)} found):")
        for ev in events:
            print(f"  {ev}")
    else:
        print("\nToday's High-Impact USD Events: None scheduled / market holiday")

    # -- Derive bias from events that have already released (actual available) -
    event_bias = "NEUTRAL"
    event_reason = ""
    for ev in events:
        if ev.actual:
            bias = get_news_bias_from_event(ev.name, ev.actual, ev.forecast)
            if bias != "NEUTRAL":
                event_bias   = bias
                event_reason = f"{ev.name} (actual={ev.actual}, forecast={ev.forecast})"
                break   # use the most recent released event

    print(f"\nScheduled Event Bias : {event_bias}"
          + (f"  -  {event_reason}" if event_reason else ""))

    # -- Breaking news ---------------------------------------------------------
    news_sentiment, headlines = get_breaking_news_sentiment()
    print(f"Breaking News Bias   : {news_sentiment}")
    if headlines:
        print("  Recent headlines:")
        for hl in headlines[:5]:
            print(f"    • {hl[:90]}")

    # -- Proximity warning -----------------------------------------------------
    upcoming, upcoming_event = is_news_in_next_minutes(45, events)
    if upcoming and upcoming_event:
        print(f"\n  [!] HIGH-IMPACT NEWS IN NEXT 45 MIN: "
              f"{upcoming_event.name} @ {upcoming_event.time_est}")

    # -- Final bias ------------------------------------------------------------
    if event_bias != "NEUTRAL":
        final_bias = event_bias
    elif news_sentiment != "NEUTRAL":
        final_bias = news_sentiment
    else:
        final_bias = "NEUTRAL"

    # Conflicting signals → downgrade to NEUTRAL
    if (event_bias != "NEUTRAL" and news_sentiment != "NEUTRAL"
            and event_bias != news_sentiment):
        final_bias = "NEUTRAL"
        print("\nConflicting signals between calendar and headlines  -  bias: NEUTRAL")

    print(f"\nFinal News Bias      : {final_bias}")
    print("=" * 70)

    return {
        "events":           [str(e) for e in events],
        "event_bias":       event_bias,
        "news_sentiment":   news_sentiment,
        "final_bias":       final_bias,
        "news_in_45min":    upcoming,
        "upcoming_event":   str(upcoming_event) if upcoming_event else None,
    }


# --- Entry Point --------------------------------------------------------------

if __name__ == "__main__":
    get_full_news_report()
