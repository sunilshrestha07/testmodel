# -*- coding: utf-8 -*-
"""
obsidian_logger.py
------------------
Creates and updates Obsidian markdown trade journals.

Vault path is read from OBSIDIAN_VAULT_PATH in .env.

Note layout inside the vault:
  trades/
  |-- YYYY-MM/
  |   |-- trade_NNN.md
  |-- weekly/
      |-- week_NN_YYYY.md
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from config import LOG_LEVEL

load_dotenv()
log = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- Config -------------------------------------------------------------------
_raw_vault = os.getenv("OBSIDIAN_VAULT_PATH", "")
if not _raw_vault:
    raise EnvironmentError(
        "OBSIDIAN_VAULT_PATH is not set in .env. "
        "Point it at your Obsidian vault folder."
    )
VAULT_PATH   = Path(_raw_vault)
TRADES_DIR   = VAULT_PATH / "trades"
WEEKLY_DIR   = VAULT_PATH / "trades" / "weekly"
SCREENSHOTS  = Path(__file__).parent / "trades"   # local screenshot root

NPT = timezone(timedelta(hours=5, minutes=45))


# --- Emoji helpers ------------------------------------------------------------

def _result_emoji(result: str) -> str:
    r = result.upper()
    if r in ("WIN", "TP", "PROFIT"):
        return "[WIN]"
    if r in ("LOSS", "SL", "STOPPED"):
        return "[LOSS]"
    if r in ("BE", "BREAKEVEN"):
        return "[BE]"
    return "[OPEN]"


def _bias_emoji(bias: str) -> str:
    b = bias.upper()
    if b == "BULLISH":
        return "(BULL)"
    if b == "BEARISH":
        return "(BEAR)"
    return "(NEUT)"


# --- Markdown template --------------------------------------------------------

_TRADE_TEMPLATE = """\
---
id: "{trade_id}"
date: "{date}"
symbol: XAUUSD
direction: {direction}
result: {result}
pnl: {pnl}
tags: {tags_yaml}
---

# Trade #{trade_id} -- XAUUSD {direction} {result_emoji}

**Date:** {date}
**Session:** {session}
**NPT Time:** {npt_time}

---

## Confluence

| Factor      | Bias     | Score  |
|-------------|----------|--------|
| Gold H4     | {gold_bias} {gold_bias_emoji} | {gold_score:+.2f} |
| News        | {news_bias} {news_bias_emoji} | {news_score:+.2f} |
| DXY         | {dxy_bias} {dxy_bias_emoji}  | {dxy_score:+.2f} |
| Divergence  | {divergence} | -- |
| **Final**   | **{signal_strength}** | **{final_score:+.2f}** |

---

## Setup

- **Strategy:** {strategy}
- **Sweep Level:** {sweep_level}
- **FVG Zone:** {fvg_top} -- {fvg_bottom}
- **Entry:** {entry} | **SL:** {sl} | **TP:** {tp}
- **R:R:** 1:{rr} | **Lot Size:** {lot_size}

---

## Outcome

- **Result:** {result} {result_emoji}
- **P&L:** {pnl}
- **Duration:** {duration}
- **Exit Price:** {exit_price}

---

## Charts

### Pre-Entry (H4 Bias)
![[{pre_entry_img}]]

### Entry Setup (M5)
![[{entry_img}]]

### Outcome (M5)
![[{outcome_img}]]

---

## Review

<!-- Fill this manually after the session -->

### What went well?


### What could be improved?


### Lessons learned?


---

## Tags

{tags_inline}
"""

_OUTCOME_BLOCK = """\
## Outcome

- **Result:** {result} {result_emoji}
- **P&L:** {pnl}
- **Duration:** {duration}
- **Exit Price:** {exit_price}
"""

_WEEKLY_TEMPLATE = """\
---
week: {week_number}
year: {year}
period: "{week_start} to {week_end}"
---

# Week {week_number} -- {year} Trade Summary

**Period:** {week_start} to {week_end}

---

## Statistics

| Metric          | Value          |
|-----------------|----------------|
| Total Trades    | {total_trades} |
| Wins            | {wins}         |
| Losses          | {losses}       |
| Breakevens      | {breakevens}   |
| Win Rate        | {win_rate:.1f}% |
| Avg R:R         | {avg_rr:.2f}  |
| Total P&L       | {total_pnl}   |

---

## Trade Log

{trade_log}

---

## Best Trade

{best_trade}

## Worst Trade

{worst_trade}

---

## Lessons Learned

<!-- Fill this section manually at the end of the week -->

1.
2.
3.

---

## Tags

#weekly-review #xauusd #{year}-W{week_number:02d}
"""


# --- Path helpers -------------------------------------------------------------

def _note_path(trade_id: str, dt: Optional[datetime] = None) -> Path:
    if dt is None:
        dt = datetime.now(tz=NPT)
    folder = TRADES_DIR / dt.strftime("%Y-%m")
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"trade_{trade_id}.md"


def _weekly_path(week_number: int, year: int) -> Path:
    WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
    return WEEKLY_DIR / f"week_{week_number:02d}_{year}.md"


def _screenshot_obsidian_name(trade_id: str, stype: str, annotated: bool = True) -> str:
    """
    Return the bare filename that Obsidian uses for ![[embeds]].
    Obsidian resolves files by name anywhere in the vault.
    """
    suffix = "_annotated" if annotated else ""
    return f"{stype}{suffix}.png"


# --- 1. Create trade note -----------------------------------------------------

def create_trade_note(
    trade_data:      dict,
    screenshot_paths: dict,
) -> Path:
    """
    Write a new trade journal note to the Obsidian vault.

    Parameters
    ----------
    trade_data : dict
        Required keys:
          trade_id, direction, session, strategy,
          gold_bias, gold_score, news_bias, news_score,
          dxy_bias, dxy_score, divergence, final_score, signal_strength,
          sweep_level, fvg_top, fvg_bottom,
          entry, sl, tp, rr, lot_size
        Optional keys:
          result (default "OPEN"), pnl (default "--"),
          duration (default "--"), exit_price (default "--"),
          tags (default [])

    screenshot_paths : dict
        Output of screenshot_manager.get_screenshot_paths().
        Structure: {"pre_entry": {"annotated": Path, ...}, ...}

    Returns
    -------
    Path to the created .md file.
    """
    now   = datetime.now(tz=NPT)
    td    = trade_data

    trade_id    = str(td["trade_id"])
    direction   = td.get("direction", "LONG").upper()
    result      = td.get("result", "OPEN").upper()
    pnl         = td.get("pnl", "--")
    duration    = td.get("duration", "--")
    exit_price  = td.get("exit_price", "--")
    tags        = td.get("tags", [])
    if not isinstance(tags, list):
        tags = [tags]
    base_tags   = ["xauusd", "trading", f"#{direction.lower()}"]
    all_tags    = list(dict.fromkeys(base_tags + tags))  # dedup, preserve order

    # Screenshot embed names
    pre_img     = _screenshot_obsidian_name(trade_id, "pre_entry")
    entry_img   = _screenshot_obsidian_name(trade_id, "entry")
    outcome_img = _screenshot_obsidian_name(trade_id, "outcome")

    content = _TRADE_TEMPLATE.format(
        trade_id       = trade_id,
        date           = now.strftime("%Y-%m-%d"),
        npt_time       = now.strftime("%I:%M %p NPT"),
        direction      = direction,
        result         = result,
        result_emoji   = _result_emoji(result),
        pnl            = pnl,
        session        = td.get("session", "--"),
        gold_bias      = td.get("gold_bias", "--"),
        gold_bias_emoji= _bias_emoji(td.get("gold_bias", "")),
        gold_score     = float(td.get("gold_score", 0)),
        news_bias      = td.get("news_bias", "--"),
        news_bias_emoji= _bias_emoji(td.get("news_bias", "")),
        news_score     = float(td.get("news_score", 0)),
        dxy_bias       = td.get("dxy_bias", "--"),
        dxy_bias_emoji = _bias_emoji(td.get("dxy_bias", "")),
        dxy_score      = float(td.get("dxy_score", 0)),
        divergence     = td.get("divergence", "--"),
        final_score    = float(td.get("final_score", 0)),
        signal_strength= td.get("signal_strength", "--"),
        strategy       = td.get("strategy", "--"),
        sweep_level    = td.get("sweep_level", "--"),
        fvg_top        = td.get("fvg_top", "--"),
        fvg_bottom     = td.get("fvg_bottom", "--"),
        entry          = td.get("entry", "--"),
        sl             = td.get("sl", "--"),
        tp             = td.get("tp", "--"),
        rr             = td.get("rr", "--"),
        lot_size       = td.get("lot_size", "--"),
        duration       = duration,
        exit_price     = exit_price,
        pre_entry_img  = pre_img,
        entry_img      = entry_img,
        outcome_img    = outcome_img,
        tags_yaml      = "[" + ", ".join(all_tags) + "]",
        tags_inline    = "  ".join(f"#{t.lstrip('#')}" for t in all_tags),
    )

    out = _note_path(trade_id, now)
    out.write_text(content, encoding="utf-8")
    log.info("Trade note created: %s", out)
    return out


# --- 2. Update trade outcome --------------------------------------------------

def update_trade_outcome(
    trade_id:     str,
    outcome_data: dict,
) -> Path:
    """
    Patch the Outcome section of an existing trade note.

    outcome_data keys:
      result, pnl, duration, exit_price

    Also copies the outcome screenshot path so the embed resolves correctly.
    """
    # Find the note (search by month folders, newest first)
    note_path: Optional[Path] = None
    for folder in sorted(TRADES_DIR.iterdir(), reverse=True):
        if not folder.is_dir():
            continue
        candidate = folder / f"trade_{trade_id}.md"
        if candidate.exists():
            note_path = candidate
            break

    if note_path is None:
        raise FileNotFoundError(
            f"No note found for trade_id={trade_id!r} under {TRADES_DIR}"
        )

    content = note_path.read_text(encoding="utf-8")

    result      = outcome_data.get("result", "OPEN").upper()
    pnl         = outcome_data.get("pnl", "--")
    duration    = outcome_data.get("duration", "--")
    exit_price  = outcome_data.get("exit_price", "--")

    new_outcome = _OUTCOME_BLOCK.format(
        result       = result,
        result_emoji = _result_emoji(result),
        pnl          = pnl,
        duration     = duration,
        exit_price   = exit_price,
    )

    # Replace the entire Outcome section (between ## Outcome and the next --- ## Charts)
    # Template uses blank lines around the --- separator, so match loosely
    pattern = r"## Outcome\n.*?(?=\n---\n\n## Charts)"
    replacement = new_outcome.rstrip("\n")
    updated = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Update the YAML frontmatter result / pnl fields
    updated = re.sub(r'^result: .*$', f'result: {result}', updated, flags=re.MULTILINE)
    updated = re.sub(r'^pnl: .*$',    f'pnl: {pnl}',       updated, flags=re.MULTILINE)

    # Update the h1 headline emoji  (e.g. [OPEN] -> [WIN])
    updated = re.sub(
        r'(# Trade #\S+ -- XAUUSD \w+) \[.*?\]',
        rf'\1 {_result_emoji(result)}',
        updated,
    )

    note_path.write_text(updated, encoding="utf-8")
    log.info("Trade outcome updated: %s  result=%s  pnl=%s", note_path.name, result, pnl)
    return note_path


# --- 3. Weekly summary --------------------------------------------------------

def _parse_frontmatter_field(text: str, field: str) -> str:
    m = re.search(rf'^{field}:\s*(.+)$', text, re.MULTILINE)
    return m.group(1).strip().strip('"') if m else "--"


def _parse_rr(text: str) -> float:
    # Matches the number after "1:" on a line containing R:R
    # e.g.  - **R:R:** 1:23.1 | **Lot Size:** 0.05
    m = re.search(r'R:R.*?1:([\d.]+)', text)
    return float(m.group(1)) if m else 0.0


def create_weekly_summary(week_number: int, year: Optional[int] = None) -> Path:
    """
    Scan all trade notes that fall in the given ISO week and generate
    a summary note at trades/weekly/week_NN_YYYY.md.

    Parameters
    ----------
    week_number : int   ISO week number (1-53)
    year        : int   defaults to current year

    Returns
    -------
    Path to the weekly summary note.
    """
    if year is None:
        year = datetime.now(tz=NPT).year

    # Determine the Monday and Sunday of the week
    week_start_dt = datetime.fromisocalendar(year, week_number, 1)  # Monday
    week_end_dt   = week_start_dt + timedelta(days=6)               # Sunday
    week_start    = week_start_dt.strftime("%Y-%m-%d")
    week_end      = week_end_dt.strftime("%Y-%m-%d")

    # Collect all trade notes whose date falls in this week
    trade_notes: list[dict] = []
    if TRADES_DIR.exists():
        for folder in sorted(TRADES_DIR.iterdir()):
            if not folder.is_dir() or folder.name == "weekly":
                continue
            for note in sorted(folder.glob("trade_*.md")):
                text = note.read_text(encoding="utf-8")
                date_str = _parse_frontmatter_field(text, "date")
                try:
                    note_dt = datetime.strptime(date_str, "%Y-%m-%d")
                    if not (week_start_dt <= note_dt <= week_end_dt):
                        continue
                except ValueError:
                    continue

                trade_notes.append({
                    "file":      note.name,
                    "trade_id":  _parse_frontmatter_field(text, "id"),
                    "direction": _parse_frontmatter_field(text, "direction"),
                    "result":    _parse_frontmatter_field(text, "result"),
                    "pnl":       _parse_frontmatter_field(text, "pnl"),
                    "rr":        _parse_rr(text),
                    "date":      date_str,
                })

    # Calculate stats
    total   = len(trade_notes)
    wins    = sum(1 for t in trade_notes if t["result"] in ("WIN", "TP", "PROFIT"))
    losses  = sum(1 for t in trade_notes if t["result"] in ("LOSS", "SL", "STOPPED"))
    bes     = sum(1 for t in trade_notes if t["result"] in ("BE", "BREAKEVEN"))
    win_rate = (wins / total * 100) if total else 0.0

    rr_values = [t["rr"] for t in trade_notes if t["rr"] > 0]
    avg_rr    = sum(rr_values) / len(rr_values) if rr_values else 0.0

    # Try to parse numeric P&L for best/worst
    def _to_float(pnl_str: str) -> float:
        try:
            return float(re.sub(r"[^0-9.\-]", "", pnl_str))
        except ValueError:
            return 0.0

    sorted_by_pnl = sorted(trade_notes, key=lambda t: _to_float(t["pnl"]), reverse=True)
    best  = sorted_by_pnl[0]  if sorted_by_pnl else None
    worst = sorted_by_pnl[-1] if sorted_by_pnl else None

    # Build trade log table
    if trade_notes:
        rows = ["| Trade | Date | Dir | Result | P&L | R:R |",
                "|-------|------|-----|--------|-----|-----|"]
        for t in trade_notes:
            rows.append(
                f"| [[{t['file']}\\|#{t['trade_id']}]] "
                f"| {t['date']} | {t['direction']} "
                f"| {t['result']} | {t['pnl']} | 1:{t['rr']:.1f} |"
            )
        trade_log = "\n".join(rows)
    else:
        trade_log = "_No trades recorded this week._"

    # P&L totals (numeric where possible)
    pnl_floats = [_to_float(t["pnl"]) for t in trade_notes]
    total_pnl  = f"${sum(pnl_floats):+.2f}" if pnl_floats else "--"

    def _fmt_trade(t: Optional[dict]) -> str:
        if t is None:
            return "_None_"
        return (f"Trade #{t['trade_id']} ({t['date']})  "
                f"{t['direction']}  Result: {t['result']}  "
                f"P&L: {t['pnl']}  R:R: 1:{t['rr']:.1f}")

    content = _WEEKLY_TEMPLATE.format(
        week_number = week_number,
        year        = year,
        week_start  = week_start,
        week_end    = week_end,
        total_trades= total,
        wins        = wins,
        losses      = losses,
        breakevens  = bes,
        win_rate    = win_rate,
        avg_rr      = avg_rr,
        total_pnl   = total_pnl,
        trade_log   = trade_log,
        best_trade  = _fmt_trade(best),
        worst_trade = _fmt_trade(worst),
    )

    out = _weekly_path(week_number, year)
    out.write_text(content, encoding="utf-8")
    log.info("Weekly summary created: %s  (%d trades)", out, total)
    return out


# --- Entry Point --------------------------------------------------------------

if __name__ == "__main__":
    from screenshot_manager import capture_chart, get_screenshot_paths

    DEMO_ID = "001"

    print("=" * 60)
    print("  Obsidian Logger - XAUUSD Bot")
    print("=" * 60)
    print(f"\nVault : {VAULT_PATH}")

    # --- 1. Build dummy trade data -------------------------------------------
    dummy_trade = {
        "trade_id":       DEMO_ID,
        "direction":      "LONG",
        "session":        "NY_AM_SILVER_BULLET",
        "strategy":       "ICT Silver Bullet (NY AM)",
        "gold_bias":      "BULLISH",
        "gold_score":     0.40,
        "news_bias":      "BULLISH",
        "news_score":     0.35,
        "dxy_bias":       "BEARISH",
        "dxy_score":      0.25,
        "divergence":     "ALIGNED",
        "final_score":    1.00,
        "signal_strength":"STRONG_LONG",
        "sweep_level":    4698.62,
        "fvg_top":        4712.80,
        "fvg_bottom":     4709.40,
        "entry":          4711.10,
        "sl":             4707.20,
        "tp":             4801.22,
        "rr":             23.1,
        "lot_size":       0.05,
        "result":         "OPEN",
        "pnl":            "--",
        "duration":       "--",
        "exit_price":     "--",
        "tags":           ["silver-bullet", "ny-am", "ict"],
    }

    # --- 2. Take a screenshot for the pre_entry frame ------------------------
    print("\n[1] Capturing pre_entry screenshot...")
    try:
        raw = capture_chart(DEMO_ID, "pre_entry")
        print(f"    Saved: {raw}")
    except Exception as exc:
        print(f"    WARNING: Screenshot failed ({exc}) -- continuing without it")
        raw = None

    paths = get_screenshot_paths(DEMO_ID)

    # --- 3. Create the trade note --------------------------------------------
    print("\n[2] Creating trade note in Obsidian vault...")
    note = create_trade_note(dummy_trade, paths)
    print(f"    Note : {note}")

    # --- 4. Simulate trade outcome -------------------------------------------
    print("\n[3] Updating with trade outcome (WIN)...")
    outcome = {
        "result":      "WIN",
        "pnl":         "+$182.50",
        "duration":    "47 min",
        "exit_price":  "4801.22",
    }
    update_trade_outcome(DEMO_ID, outcome)
    print("    Outcome written.")

    # --- 5. Weekly summary ---------------------------------------------------
    now_npt     = datetime.now(tz=NPT)
    week_num    = now_npt.isocalendar().week
    print(f"\n[4] Creating weekly summary (Week {week_num})...")
    summary = create_weekly_summary(week_num)
    print(f"    Summary : {summary}")

    print("\n    Open Obsidian and navigate to trades/ to verify.")
    print("=" * 60)
