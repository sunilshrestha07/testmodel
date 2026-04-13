# -*- coding: utf-8 -*-
"""
screenshot_manager.py
---------------------
Captures and annotates MT5 chart screenshots for trade journaling.

Requirements: mss, pywin32, Pillow

Folder layout created automatically:
  trades/
  |-- YYYY-MM/
  |   |-- trade_NNN/
  |   |   |-- pre_entry.png
  |   |   |-- pre_entry_annotated.png
  |   |   |-- entry.png
  |   |   |-- entry_annotated.png
  |   |   |-- outcome.png
  |   |   `-- outcome_annotated.png
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import ctypes
import win32con
import win32gui
import win32ui
from PIL import Image, ImageDraw, ImageFont

# ctypes handle to user32 for PrintWindow
_user32 = ctypes.windll.user32
PW_RENDERFULLCONTENT = 0x00000002

from config import LOG_LEVEL

log = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- Config -------------------------------------------------------------------
TRADES_ROOT   = Path(__file__).parent / "trades"
VALID_TYPES   = ("pre_entry", "entry", "outcome")

# Keywords used to locate the MT5 window
MT5_TITLE_KEYWORDS = ("metatrader", "metaquot", "mt5", "mt4")

# Annotation style
ANNOTATION_COLORS = {
    "entry": (0,   200, 0),    # green
    "sl":    (220, 50,  50),   # red
    "tp":    (50,  150, 220),  # blue
    "level": (200, 200, 0),    # yellow (generic)
}
LINE_WIDTH    = 2
FONT_SIZE     = 16
LABEL_PADDING = 6


# --- 1. Find MT5 window -------------------------------------------------------

def _is_mt5_title(title: str) -> bool:
    tl = title.lower()
    return any(kw in tl for kw in MT5_TITLE_KEYWORDS)


def _enum_callback(hwnd: int, results: list) -> None:
    if win32gui.IsWindowVisible(hwnd):
        try:
            title = win32gui.GetWindowText(hwnd)
            if title and _is_mt5_title(title):
                results.append((hwnd, title))
        except Exception:
            pass


def find_mt5_window() -> tuple[int, int, int, int]:
    """
    Locate the MT5 terminal window on the desktop.

    If the window is minimised it is restored and brought to the
    foreground before the coordinates are returned.

    Returns
    -------
    (x, y, width, height) in screen pixels.

    Raises
    ------
    RuntimeError if MT5 is not running.
    """
    matches: list[tuple[int, str]] = []
    win32gui.EnumWindows(_enum_callback, matches)

    if not matches:
        raise RuntimeError(
            "MetaTrader 5 window not found. "
            "Make sure MT5 is running before taking screenshots."
        )

    # Prefer the window that is NOT minimised; fall back to the first match
    hwnd = None
    for h, title in matches:
        placement = win32gui.GetWindowPlacement(h)
        if placement[1] != win32con.SW_SHOWMINIMIZED:
            hwnd = h
            log.debug("Using MT5 window (visible): %r  hwnd=%d", title[:60], h)
            break

    if hwnd is None:
        hwnd = matches[0][0]
        log.info("MT5 is minimised -- restoring window (hwnd=%d)", hwnd)
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.8)   # let the window render

    # Bring to front so mss captures the correct content
    try:
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass
    time.sleep(0.2)

    rect = win32gui.GetWindowRect(hwnd)       # (left, top, right, bottom)
    x, y, right, bottom = rect
    width  = right  - x
    height = bottom - y

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"MT5 window has invalid dimensions: {rect}. "
            "Try manually restoring the window."
        )

    log.info("MT5 window: x=%d y=%d w=%d h=%d", x, y, width, height)
    return x, y, width, height


# --- Window-handle capture (works regardless of z-order) ---------------------

def _capture_hwnd(hwnd: int) -> Image.Image:
    """
    Capture a window's content using PrintWindow + GDI BitBlt.
    This works even when the window is partially obscured or behind other apps.
    """
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    w = right  - left
    h = bottom - top

    hwnd_dc  = win32gui.GetWindowDC(hwnd)
    mfc_dc   = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc  = mfc_dc.CreateCompatibleDC()
    bmp      = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(mfc_dc, w, h)
    save_dc.SelectObject(bmp)

    # PrintWindow with PW_RENDERFULLCONTENT captures DWM-composited content
    result = _user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)
    if not result:
        # Fallback: BitBlt from the window DC
        save_dc.BitBlt((0, 0), (w, h), mfc_dc, (0, 0), win32con.SRCCOPY)

    bmp_info = bmp.GetInfo()
    bmp_str  = bmp.GetBitmapBits(True)
    img = Image.frombuffer(
        "RGB",
        (bmp_info["bmWidth"], bmp_info["bmHeight"]),
        bmp_str, "raw", "BGRX", 0, 1,
    )

    win32gui.DeleteObject(bmp.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)
    return img


def _find_mt5_hwnd() -> int:
    """Return the hwnd of the MT5 window (restoring it if minimised)."""
    matches: list[tuple[int, str]] = []
    win32gui.EnumWindows(_enum_callback, matches)
    if not matches:
        raise RuntimeError(
            "MetaTrader 5 window not found. Make sure MT5 is running."
        )
    # Prefer non-minimised window
    for h, _ in matches:
        placement = win32gui.GetWindowPlacement(h)
        if placement[1] != win32con.SW_SHOWMINIMIZED:
            return h
    # All minimised -- restore the first one
    hwnd = matches[0][0]
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(0.8)
    return hwnd


# --- 2. Capture chart ---------------------------------------------------------

def _trade_folder(trade_id: str) -> Path:
    month_dir = TRADES_ROOT / datetime.now().strftime("%Y-%m")
    folder    = month_dir / f"trade_{trade_id}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def capture_chart(
    trade_id:        str,
    screenshot_type: str,
) -> Path:
    """
    Capture the MT5 window and save it as a PNG.

    Parameters
    ----------
    trade_id : str
        Unique identifier for the trade (e.g. "001", "042").
    screenshot_type : str
        One of "pre_entry", "entry", "outcome".

    Returns
    -------
    Path to the saved PNG file.
    """
    if screenshot_type not in VALID_TYPES:
        raise ValueError(
            f"screenshot_type must be one of {VALID_TYPES}, got {screenshot_type!r}"
        )

    hwnd   = _find_mt5_hwnd()
    folder = _trade_folder(trade_id)
    out_path = folder / f"{screenshot_type}.png"

    img = _capture_hwnd(hwnd)
    img.save(str(out_path), "PNG")

    log.info("Screenshot saved: %s  (%dx%d)", out_path, img.width, img.height)
    return out_path


# --- 3. Annotate screenshot ---------------------------------------------------

def _load_font(size: int) -> ImageFont.ImageFont:
    """Try to load a crisp system font; fall back to default if unavailable."""
    candidates = [
        "C:/Windows/Fonts/consola.ttf",   # Consolas
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _price_to_y(
    price: float,
    img_height: int,
    price_high: float,
    price_low: float,
) -> int:
    """
    Map a price to a Y pixel coordinate.

    price_high maps to y=0 (top), price_low maps to y=img_height (bottom).
    Returns -1 if the price is outside the visible range.
    """
    if price_high <= price_low:
        return -1
    ratio = (price_high - price) / (price_high - price_low)
    return int(ratio * img_height)


def annotate_screenshot(
    image_path:  Path,
    annotations: dict[str, float],
    price_high:  Optional[float] = None,
    price_low:   Optional[float] = None,
) -> Path:
    """
    Draw horizontal price lines and labels onto a screenshot.

    Parameters
    ----------
    image_path : Path
        Source PNG file (typically the raw capture).
    annotations : dict
        e.g. {"entry": 3322.60, "sl": 3318.50, "tp": 3342.50}
        Any key whose name is in ANNOTATION_COLORS gets a matching colour;
        unknown keys receive the "level" (yellow) colour.
    price_high, price_low : float, optional
        The visible price range of the chart.  When provided the lines are
        placed at the correct vertical position.  When omitted, lines are
        spaced evenly from top to bottom based on the sorted price order.

    Returns
    -------
    Path to the annotated PNG (saved alongside the original).
    """
    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(FONT_SIZE)
    W, H = img.size

    sorted_items = sorted(annotations.items(), key=lambda kv: kv[1], reverse=True)

    for idx, (label, price) in enumerate(sorted_items):
        color = ANNOTATION_COLORS.get(label.lower(), ANNOTATION_COLORS["level"])

        # Y position
        if price_high is not None and price_low is not None:
            py = _price_to_y(price, H, price_high, price_low)
        else:
            # Evenly distribute when no range given
            py = int(H * 0.15 + idx * (H * 0.70 / max(len(sorted_items) - 1, 1)))

        if py < 0 or py > H:
            log.debug("Price %.2f is outside chart range, skipping annotation", price)
            continue

        # Horizontal line across the full width
        draw.line([(0, py), (W, py)], fill=color, width=LINE_WIDTH)

        # Label background + text on the right side
        label_text = f"  {label.upper()}: {price:.2f}  "
        bbox       = draw.textbbox((0, 0), label_text, font=font)
        tw         = bbox[2] - bbox[0]
        th         = bbox[3] - bbox[1]
        tx         = W - tw - LABEL_PADDING
        ty         = py - th - LABEL_PADDING
        ty         = max(LABEL_PADDING, min(ty, H - th - LABEL_PADDING))

        # Semi-transparent-style background (solid dark box)
        draw.rectangle(
            [tx - LABEL_PADDING, ty - LABEL_PADDING,
             tx + tw + LABEL_PADDING, ty + th + LABEL_PADDING],
            fill=(20, 20, 20),
        )
        draw.text((tx, ty), label_text, fill=color, font=font)

    # Save annotated version
    stem         = image_path.stem                        # e.g. "pre_entry"
    out_path     = image_path.parent / f"{stem}_annotated.png"
    img.save(out_path, "PNG")
    log.info("Annotated screenshot saved: %s", out_path)
    return out_path


# --- 4. Get screenshot paths --------------------------------------------------

def get_screenshot_paths(trade_id: str) -> dict[str, dict[str, Path]]:
    """
    Return a dict of all expected screenshot paths for a trade.
    Paths are returned whether or not the files exist yet.

    Returns
    -------
    {
        "pre_entry": {"raw": Path, "annotated": Path},
        "entry":     {"raw": Path, "annotated": Path},
        "outcome":   {"raw": Path, "annotated": Path},
    }
    """
    folder = _trade_folder(trade_id)
    return {
        stype: {
            "raw":        folder / f"{stype}.png",
            "annotated":  folder / f"{stype}_annotated.png",
        }
        for stype in VALID_TYPES
    }


# --- Entry Point --------------------------------------------------------------

if __name__ == "__main__":
    import sys

    DEMO_TRADE_ID = "test_001"

    print("=" * 55)
    print("  Screenshot Manager - XAUUSD Bot")
    print("=" * 55)

    # --- Locate MT5 -----------------------------------------------------------
    print("\n[1] Locating MT5 window...")
    try:
        x, y, w, h = find_mt5_window()
        print(f"    Found at  x={x}  y={y}  w={w}  h={h}")
    except RuntimeError as exc:
        print(f"    ERROR: {exc}")
        sys.exit(1)

    # --- Capture pre_entry ----------------------------------------------------
    print("\n[2] Capturing pre_entry screenshot...")
    raw_path = capture_chart(DEMO_TRADE_ID, "pre_entry")
    print(f"    Saved : {raw_path}")

    # --- Annotate with dummy levels -------------------------------------------
    print("\n[3] Annotating with dummy levels...")
    dummy_annotations = {
        "entry": 4762.50,
        "sl":    4748.00,
        "tp":    4801.22,
    }
    ann_path = annotate_screenshot(raw_path, dummy_annotations)
    print(f"    Saved : {ann_path}")

    # --- Path registry --------------------------------------------------------
    print("\n[4] Screenshot path registry for trade:")
    paths = get_screenshot_paths(DEMO_TRADE_ID)
    for stype, p in paths.items():
        exists_raw = "[exists]" if p["raw"].exists()       else "[pending]"
        exists_ann = "[exists]" if p["annotated"].exists() else "[pending]"
        print(f"    {stype:<12}  raw={exists_raw}  annotated={exists_ann}")

    print(f"\n    Folder: {raw_path.parent}")
    print("\n    Open the annotated file to verify the overlay.")
    print("=" * 55)
