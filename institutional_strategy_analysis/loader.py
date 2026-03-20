# -*- coding: utf-8 -*-
"""
institutional_strategy_analysis/loader.py
──────────────────────────────────────────
Downloads the Google Sheet and splits raw rows into two clean DataFrames:
    df_yearly  — rows where סוג == "Year"
    df_monthly — rows where סוג == "Month"

Key principle: the "סוג" / "סוג התאריך" column is the authoritative
discriminator.  We never guess frequency from date format.

Public API
──────────
    load_raw_blocks(sheet_url) -> (df_yearly, df_monthly, debug_info, errors)
"""
from __future__ import annotations

import io
import re
import logging
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Sheet URL helpers ─────────────────────────────────────────────────────────

def _extract_sheet_id(url: str) -> str:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if not m:
        raise ValueError(f"Cannot extract sheet ID from URL: {url}")
    return m.group(1)

def _xlsx_url(sheet_id: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"

# ── Unicode cleaning (Google Sheets injects RTL marks into Hebrew CSVs) ──────

_INVIS = re.compile(
    r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff\u00a0\u00ad]"
)

def _c(v: object) -> str:
    """Strip invisible Unicode chars and whitespace."""
    return _INVIS.sub("", str(v)).strip()

def _norm(v: object) -> str:
    return _c(v).lower().replace("_", " ").replace("-", " ")

def _blank(v: object) -> bool:
    return _norm(v) in {"", "nan", "none", "nat"}

# ── Frequency column detection ────────────────────────────────────────────────

_FREQ_KEYWORDS = {"סוג", "סוג התאריך", "type", "frequency", "freq", "kind"}
_YEAR_VALUES   = {"year", "שנתי", "שנה"}
_MONTH_VALUES  = {"month", "חודשי", "חודש", "monthly"}

def _find_freq_col(columns: list[str]) -> Optional[str]:
    for c in columns:
        if _norm(c) in _FREQ_KEYWORDS:
            return c
        if any(kw in _norm(c) for kw in _FREQ_KEYWORDS):
            return c
    return None

def _is_year_val(v: object) -> bool:
    return _c(v).lower() in _YEAR_VALUES

def _is_month_val(v: object) -> bool:
    return _c(v).lower() in _MONTH_VALUES

# ── Smart header detection ────────────────────────────────────────────────────

_DATE_KW   = {"תאריך", "date", "חודש", "month", "time", "period"}
_SKIP_KW   = {"unnamed", "index", "מספר", "id"}

def _row_header_score(row: pd.Series) -> int:
    """Score a row on how likely it is to be the header row."""
    cells = [_c(v).lower() for v in row.values if not _blank(v)]
    if not cells:
        return -999
    score = 0
    for cell in cells:
        if any(kw in cell for kw in _DATE_KW):
            score += 5
        if any(kw in cell for kw in _FREQ_KEYWORDS):
            score += 4
        if any(kw in cell for kw in _SKIP_KW):
            score -= 1
        # non-numeric text = good header indicator
        try:
            float(cell.replace("%", ""))
        except ValueError:
            score += 1
    # penalise rows that are mostly numeric
    num_cells = sum(1 for c in cells if _try_float(c) is not None)
    if num_cells > len(cells) * 0.6:
        score -= 5
    return score

def _try_float(s: str) -> Optional[float]:
    try:
        return float(s.replace("%", "").replace(",", "."))
    except ValueError:
        return None

def _find_header_row_idx(df: pd.DataFrame, max_scan: int = 20) -> int:
    best_idx, best_score = 0, -999
    for i in range(min(max_scan, len(df))):
        score = _row_header_score(df.iloc[i])
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx

# ── Sheet name → (manager, track) ────────────────────────────────────────────

_SHEET_META: dict[str, dict] = {
    # add explicit mappings here as more sheets are added
}

_MANAGER_HINTS = [
    "הראל", "מגדל", "כלל", "מנורה", "הפניקס", "אנליסט", "מיטב",
    "ילין", "פסגות", "אלטשולר", "ברקת", "אלומות",
]
_TRACK_HINTS = {
    "כלל": "כללי", "כללי": "כללי",
    "מנייתי": "מנייתי", "מניות": "מנייתי",
    "אגח": 'אג"ח', 'אג"ח': 'אג"ח',
}

def _infer_meta(sheet_name: str) -> dict:
    s = _c(sheet_name)
    for key, meta in _SHEET_META.items():
        if _c(key) in s:
            return meta
    manager = next((m for m in _MANAGER_HINTS if m in s), s)
    track = "כללי"
    for pat, val in _TRACK_HINTS.items():
        if pat in s.lower():
            track = val
            break
    return {"manager": manager, "track": track}

# ── Parse a single cleaned sheet-DataFrame ────────────────────────────────────

def _parse_sheet(raw: pd.DataFrame, sheet_name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Given a raw DataFrame (all rows, all columns as strings),
    return (df_yearly, df_monthly, debug_info).

    Strategy:
      1. Find the header row by scoring.
      2. Rebuild DataFrame with that row as columns.
      3. Clean column names (strip invisible chars).
      4. Identify the frequency column ("סוג" / "סוג התאריך").
      5. Split rows into Yearly / Monthly.
      6. Parse each set independently.
    """
    from institutional_strategy_analysis.normalizer import normalise_block

    debug: dict = {"sheet": sheet_name}

    if raw is None or raw.empty:
        debug["error"] = "empty sheet"
        return pd.DataFrame(), pd.DataFrame(), debug

    # ── 1. Find and set header ────────────────────────────────────────────
    hdr_idx = _find_header_row_idx(raw)
    cols = [_c(v) for v in raw.iloc[hdr_idx].values]
    data = raw.iloc[hdr_idx + 1:].copy()
    data.columns = cols
    data = data.loc[~data.apply(lambda r: all(_blank(v) for v in r), axis=1)]
    data = data.reset_index(drop=True)
    debug["header_row"] = hdr_idx
    debug["columns"] = cols[:10]

    # ── 2. Find frequency column ──────────────────────────────────────────
    freq_col = _find_freq_col(cols)
    debug["freq_col"] = freq_col

    if freq_col is None:
        # No frequency column — treat all rows as yearly (annual snapshots)
        debug["split_method"] = "no_freq_col_all_yearly"
        df_y = normalise_block(data, sheet_name, "yearly")
        debug.update({"yearly_rows": len(df_y), "monthly_rows": 0})
        return df_y, pd.DataFrame(), debug

    # ── 3. Split by frequency column ─────────────────────────────────────
    freq_vals = data[freq_col].astype(str).map(_c)
    year_mask  = freq_vals.map(_is_year_val)
    month_mask = freq_vals.map(_is_month_val)
    debug["split_method"] = "freq_col"
    debug["year_rows_raw"]  = int(year_mask.sum())
    debug["month_rows_raw"] = int(month_mask.sum())

    df_y = normalise_block(data[year_mask].copy(),  sheet_name, "yearly")
    df_m = normalise_block(data[month_mask].copy(), sheet_name, "monthly")

    debug["yearly_rows"]  = len(df_y)
    debug["monthly_rows"] = len(df_m)
    if not df_y.empty:
        debug["yearly_range"]  = f"{df_y['date'].min().year} – {df_y['date'].max().year}"
    if not df_m.empty:
        debug["monthly_range"] = f"{df_m['date'].min().strftime('%Y-%m')} – {df_m['date'].max().strftime('%Y-%m')}"

    return df_y, df_m, debug

# ── XLSX transport ────────────────────────────────────────────────────────────

def _download_xlsx(sheet_id: str) -> tuple[Optional[bytes], Optional[str]]:
    url = _xlsx_url(sheet_id)
    try:
        r = requests.get(url, timeout=30, allow_redirects=True)
        ct = r.headers.get("Content-Type", "").lower()
        if r.status_code in (401, 403):
            return None, f"🔒 גישה לגיליון נדחתה (HTTP {r.status_code}). ודא שהגיליון פתוח לצפייה."
        if r.status_code != 200:
            return None, f"⚠️ שגיאת HTTP {r.status_code} בהורדת הגיליון."
        if "html" in ct or r.content[:4] == b"<!DO":
            return None, "⚠️ הגיליון החזיר דף HTML במקום XLSX. בדוק הרשאות."
        return r.content, None
    except Exception as e:
        return None, f"⚠️ שגיאת רשת: {e}"

# ── Main public API ───────────────────────────────────────────────────────────

def load_raw_blocks(
    sheet_url: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[str]]:
    """
    Download the Google Sheet and return:
        df_yearly   – all yearly rows, normalised
        df_monthly  – all monthly rows, normalised
        debug_info  – list of per-sheet debug dicts
        errors      – list of error/warning strings

    Uses XLSX transport (loads all sheets in one request).
    Falls back to per-sheet CSV if XLSX fails.
    """
    errors: list[str] = []
    debug_info: list[dict] = []

    try:
        sheet_id = _extract_sheet_id(sheet_url)
    except ValueError as e:
        return pd.DataFrame(), pd.DataFrame(), [], [str(e)]

    # ── Try XLSX first ────────────────────────────────────────────────────
    xlsx_bytes, err = _download_xlsx(sheet_id)
    if err:
        errors.append(err)
        return pd.DataFrame(), pd.DataFrame(), debug_info, errors

    try:
        xls = pd.ExcelFile(io.BytesIO(xlsx_bytes), engine="openpyxl")
    except Exception as e:
        errors.append(f"⚠️ שגיאה בפתיחת XLSX: {e}")
        return pd.DataFrame(), pd.DataFrame(), debug_info, errors

    yearly_frames:  list[pd.DataFrame] = []
    monthly_frames: list[pd.DataFrame] = []

    for sheet_name in xls.sheet_names:
        try:
            raw = pd.read_excel(xls, sheet_name=sheet_name, header=None, dtype=str)
            raw = raw.fillna("").astype(str)
        except Exception as e:
            errors.append(f"⚠️ גליון '{sheet_name}': שגיאת קריאה – {e}")
            continue

        try:
            df_y, df_m, dbg = _parse_sheet(raw, sheet_name)
            debug_info.append(dbg)
            if not df_y.empty:
                yearly_frames.append(df_y)
            if not df_m.empty:
                monthly_frames.append(df_m)
        except Exception as e:
            errors.append(f"⚠️ גליון '{sheet_name}': שגיאת parsing – {e}")
            logger.exception(f"parse_sheet failed for {sheet_name}")

    df_yearly  = pd.concat(yearly_frames,  ignore_index=True) if yearly_frames  else pd.DataFrame()
    df_monthly = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()

    if df_yearly.empty and df_monthly.empty:
        errors.append("לא נטענו נתונים מאף גליון.")

    return df_yearly, df_monthly, debug_info, errors
