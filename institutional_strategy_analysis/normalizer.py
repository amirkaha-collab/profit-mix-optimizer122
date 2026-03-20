# -*- coding: utf-8 -*-
"""
institutional_strategy_analysis/normalizer.py
──────────────────────────────────────────────
Converts a raw block DataFrame (either yearly or monthly) into the
canonical normalised schema:

    manager         : str
    track           : str
    date            : datetime64[ns]  (first of month / Jan-1 for yearly)
    frequency       : "yearly" | "monthly"
    allocation_name : str
    allocation_value: float  (percent, 0-100 scale)
    source_sheet    : str

Public API
──────────
    normalise_block(raw_df, sheet_name, frequency) -> pd.DataFrame
    normalize_allocation_name(name) -> str
"""
from __future__ import annotations

import re
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Re-use the _c / _blank helpers from loader via direct definition ──────────
_INVIS = re.compile(
    r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff\u00a0\u00ad]"
)

def _c(v: object) -> str:
    return _INVIS.sub("", str(v)).strip()

def _blank(v: object) -> bool:
    return _c(v).lower() in {"", "nan", "none", "nat"}

# ── Hebrew month names ────────────────────────────────────────────────────────
_HEB_MONTHS = {
    "ינואר": 1, "פברואר": 2, "מרץ": 3, "מרס": 3,
    "אפריל": 4, "מאי": 5, "יוני": 6,
    "יולי": 7, "אוגוסט": 8, "ספטמבר": 9,
    "אוקטובר": 10, "נובמבר": 11, "דצמבר": 12,
}
_EN_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# ── Date parsing ──────────────────────────────────────────────────────────────

def _parse_date(val: object, frequency: str) -> Optional[datetime]:
    """
    Parse a date value to a datetime.
    - yearly  → Jan 1 of that year
    - monthly → first day of that month
    """
    if val is None:
        return None
    s = _c(val)
    if not s or s.lower() in {"nan", "none", ""}:
        return None

    # Already a datetime
    if isinstance(val, (datetime, pd.Timestamp)):
        dt = pd.Timestamp(val)
        return dt.replace(day=1).to_pydatetime()

    # Excel serial number
    try:
        f = float(s)
        if 20000 < f < 80000:
            dt = pd.Timestamp("1899-12-30") + pd.Timedelta(days=f)
            return dt.replace(day=1).to_pydatetime()
    except ValueError:
        pass

    # Hebrew month name + year: "ינואר 2025"
    for heb, mn in _HEB_MONTHS.items():
        if heb in s:
            y = re.search(r"(19\d{2}|20\d{2})", s)
            if y:
                return datetime(int(y.group(1)), mn, 1)

    # Pure 4-digit year: "2014"
    if re.fullmatch(r"(19|20)\d{2}", s):
        return datetime(int(s), 1, 1)

    # YYYY-MM or YYYY/MM
    m = re.match(r"^(20\d{2})[/-](\d{1,2})$", s)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), 1)

    # MM/YYYY or MM-YYYY
    m = re.match(r"^(\d{1,2})[/-](20\d{2})$", s)
    if m and 1 <= int(m.group(1)) <= 12:
        return datetime(int(m.group(2)), int(m.group(1)), 1)

    # English month name + year
    sl = s.lower()
    for name, mn in _EN_MONTHS.items():
        if re.search(rf"\b{name}\b", sl):
            y = re.search(r"(19\d{2}|20\d{2})", sl)
            if y:
                return datetime(int(y.group(1)), mn, 1)

    # Try pandas as last resort
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%m/%Y"):
        try:
            return datetime.strptime(s, fmt).replace(day=1)
        except ValueError:
            pass
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.notna(dt):
            return dt.replace(day=1).to_pydatetime()
    except Exception:
        pass

    return None

# ── Percent parsing ───────────────────────────────────────────────────────────

def _parse_percent(val: object) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if isinstance(val, float) and np.isnan(val):
            return None
        f = float(val)
        return round(f * 100 if abs(f) <= 1.5 else f, 4)
    s = _c(val).replace(",", ".").replace("−", "-").replace("%", "").strip()
    if not s:
        return None
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return None
    f = float(s)
    return round(f * 100 if abs(f) <= 1.5 else f, 4)

# ── Allocation name normalisation ─────────────────────────────────────────────

_NAME_MAP = {
    # common variations → canonical form
    'חול': 'חו"ל',
    'חו"ל': 'חו"ל',
    "חו'ל": 'חו"ל',
    "חוחל": 'חו"ל',
    "מטח": 'מט"ח',
    'מט"ח': 'מט"ח',
    "לא סחיר": "לא סחיר",
    "לאסחיר": "לא סחיר",
    "מניות": "מניות",
    "אגח": 'אג"ח',
    'אג"ח': 'אג"ח',
}

def normalize_allocation_name(name: str) -> str:
    """Return canonical allocation component name."""
    s = _c(name)
    cleaned = s.lower().replace(" ", "").replace('"', "").replace("'", "")
    for key, canonical in _NAME_MAP.items():
        if key.lower().replace(" ", "").replace('"', "") == cleaned:
            return canonical
    return s  # return as-is if no mapping found

# ── Column role detection ─────────────────────────────────────────────────────

_DATE_EXACT  = {"תאריך", "date", "חודש", "month", "time"}
_FREQ_NAMES  = {"סוג", "סוג התאריך", "type", "frequency", "freq", "kind"}
_SKIP_NAMES  = {"unnamed", "index"}

def _find_date_col(columns: list[str]) -> Optional[str]:
    cl = {c: _c(c).lower() for c in columns}
    # Exact
    for c, s in cl.items():
        if s in _DATE_EXACT and not any(fk in s for fk in {"סוג", "type"}):
            return c
    # Ends-with
    for c, s in cl.items():
        if any(s.endswith(kw) for kw in _DATE_EXACT) and "סוג" not in s:
            return c
    # Contains
    for c, s in cl.items():
        if any(kw in s for kw in _DATE_EXACT) and "סוג" not in s:
            return c
    return None

def _find_alloc_cols(columns: list[str], skip: set[str]) -> list[str]:
    result = []
    for c in columns:
        cs = _c(c).lower()
        if c in skip:
            continue
        if any(cs.startswith(s) for s in _SKIP_NAMES):
            continue
        if _blank(c):
            continue
        result.append(c)
    return result

# ── Sheet name → manager + track ─────────────────────────────────────────────

_MANAGER_HINTS = [
    "הראל", "מגדל", "כלל", "מנורה", "הפניקס", "אנליסט", "מיטב",
    "ילין", "פסגות", "אלטשולר", "ברקת", "אלומות",
]
_TRACK_HINTS = {
    "כלל": "כללי", "כללי": "כללי",
    "מנייתי": "מנייתי", "מניות": "מנייתי",
}

def _infer_meta(sheet_name: str) -> dict:
    s = _c(sheet_name)
    manager = next((m for m in _MANAGER_HINTS if m in s), s)
    track = "כללי"
    for pat, val in _TRACK_HINTS.items():
        if pat in s:
            track = val
            break
    return {"manager": manager, "track": track}

# ── Main normaliser ───────────────────────────────────────────────────────────

def normalise_block(
    raw: pd.DataFrame,
    sheet_name: str,
    frequency: str,  # "yearly" | "monthly"
) -> pd.DataFrame:
    """
    Convert a raw block (already split by frequency) into the canonical schema.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    meta = _infer_meta(sheet_name)
    raw = raw.copy()
    # Clean column names
    raw.columns = [_c(c) for c in raw.columns]

    date_col = _find_date_col(list(raw.columns))
    if date_col is None:
        logger.warning(f"No date column found in sheet '{sheet_name}' ({frequency})")
        return pd.DataFrame()

    # Determine columns to skip
    freq_col_candidates = [c for c in raw.columns if _c(c).lower() in
                           {"סוג", "סוג התאריך", "type", "frequency", "freq", "kind"}]
    skip = {date_col} | set(freq_col_candidates)
    alloc_cols = _find_alloc_cols(list(raw.columns), skip)

    if not alloc_cols:
        logger.warning(f"No allocation columns in sheet '{sheet_name}' ({frequency})")
        return pd.DataFrame()

    rows = []
    for _, row in raw.iterrows():
        dt = _parse_date(row.get(date_col), frequency)
        if dt is None:
            continue
        for col in alloc_cols:
            val = _parse_percent(row.get(col))
            if val is None:
                continue
            rows.append({
                "manager":          meta["manager"],
                "track":            meta["track"],
                "date":             pd.Timestamp(dt),
                "frequency":        frequency,
                "allocation_name":  normalize_allocation_name(col),
                "allocation_value": val,
                "source_sheet":     sheet_name,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df
