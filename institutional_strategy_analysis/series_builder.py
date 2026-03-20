# -*- coding: utf-8 -*-
"""
institutional_strategy_analysis/series_builder.py
──────────────────────────────────────────────────
Merges yearly and monthly DataFrames into a single display series with
correct priority rules:

    Monthly always wins when it covers a period.
    Yearly is used only to fill periods NOT covered by monthly.

Range priority rules
────────────────────
    YTD  → monthly only
    1Y   → monthly only
    3Y   → monthly where available, yearly for earlier years
    5Y   → monthly where available, yearly for earlier years
    All  → monthly where available, yearly for earlier years

Public API
──────────
    build_display_series(df_yearly, df_monthly, selected_range,
                         custom_start=None) -> pd.DataFrame

    get_time_bounds(df_yearly, df_monthly) -> (min_date, max_date)
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd

# ── Range label → (months_back, monthly_only) ─────────────────────────────────

_RANGE_CONFIG: dict[str, dict] = {
    "הכל":          {"months_back": None, "monthly_only": False},
    "YTD":          {"months_back": None, "monthly_only": True,  "ytd": True},
    "1Y":           {"months_back": 12,   "monthly_only": True},
    "3Y":           {"months_back": 36,   "monthly_only": False},
    "5Y":           {"months_back": 60,   "monthly_only": False},
    "מותאם אישית":  {"months_back": None, "monthly_only": False, "custom": True},
}


def get_time_bounds(
    df_yearly: pd.DataFrame,
    df_monthly: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (min_date, max_date) across both DataFrames."""
    dates = []
    for df in (df_yearly, df_monthly):
        if not df.empty and "date" in df.columns:
            dates.extend([df["date"].min(), df["date"].max()])
    if not dates:
        today = pd.Timestamp.today().normalize()
        return today, today
    return min(dates), max(dates)


def build_display_series(
    df_yearly: pd.DataFrame,
    df_monthly: pd.DataFrame,
    selected_range: str,
    custom_start: Optional[date] = None,
    filters: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build a unified, chronological, de-duplicated display series.

    Parameters
    ----------
    df_yearly, df_monthly  : normalised DataFrames from the loader
    selected_range         : one of the _RANGE_CONFIG keys
    custom_start           : used when selected_range == "מותאם אישית"
    filters                : dict with optional keys:
                             managers, tracks, allocation_names

    Returns
    -------
    pd.DataFrame with columns:
        manager, track, date, frequency, allocation_name, allocation_value,
        source_sheet
    Sorted by (manager, track, allocation_name, date).
    """
    cfg = _RANGE_CONFIG.get(selected_range, _RANGE_CONFIG["הכל"])

    # ── Apply entity filters ──────────────────────────────────────────────
    def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or not filters:
            return df
        for col, key in [("manager", "managers"), ("track", "tracks"),
                          ("allocation_name", "allocation_names")]:
            vals = filters.get(key)
            if vals and col in df.columns:
                df = df[df[col].isin(vals)]
        return df

    dy = _apply_filters(df_yearly.copy() if not df_yearly.empty else pd.DataFrame())
    dm = _apply_filters(df_monthly.copy() if not df_monthly.empty else pd.DataFrame())

    # ── Determine reference date (most recent data point) ────────────────
    max_date = pd.Timestamp.today().normalize()
    for df in (dy, dm):
        if not df.empty and "date" in df.columns:
            max_date = max(max_date, df["date"].max())

    # ── Apply time range ──────────────────────────────────────────────────
    start_cut: Optional[pd.Timestamp] = None

    if cfg.get("ytd"):
        start_cut = pd.Timestamp(max_date.year, 1, 1)
    elif cfg.get("custom") and custom_start:
        start_cut = pd.Timestamp(custom_start)
    elif cfg.get("months_back"):
        start_cut = max_date - pd.DateOffset(months=cfg["months_back"])

    if cfg.get("monthly_only"):
        # Only monthly data, optionally windowed
        if dm.empty:
            return pd.DataFrame()
        if start_cut is not None:
            dm = dm[dm["date"] >= start_cut]
        return _sort(dm)

    # ── Merge: monthly + yearly for years not covered by monthly ─────────
    if dm.empty and dy.empty:
        return pd.DataFrame()

    if dm.empty:
        result = dy
        if start_cut is not None:
            result = result[result["date"] >= start_cut]
        return _sort(result)

    if dy.empty:
        if start_cut is not None:
            dm = dm[dm["date"] >= start_cut]
        return _sort(dm)

    # Determine the earliest month covered by monthly data
    earliest_monthly = dm["date"].min()

    # Yearly data: keep only years BEFORE the earliest monthly year
    # (monthly covers that year's data better)
    cutoff_year = earliest_monthly.year
    dy_prior = dy[dy["date"].dt.year < cutoff_year].copy()

    combined = pd.concat([dy_prior, dm], ignore_index=True)

    if start_cut is not None:
        combined = combined[combined["date"] >= start_cut]

    return _sort(combined)


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sort_cols = [c for c in ["manager", "track", "allocation_name", "date"] if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)


def get_available_options(
    df_yearly: pd.DataFrame,
    df_monthly: pd.DataFrame,
) -> dict:
    """
    Return dicts of unique values for filter dropdowns.
    Deduplicates across yearly and monthly.
    """
    def _uniq(col: str) -> list:
        vals = set()
        for df in (df_yearly, df_monthly):
            if not df.empty and col in df.columns:
                vals.update(df[col].dropna().unique())
        return sorted(vals)

    return {
        "managers":         _uniq("manager"),
        "tracks":           _uniq("track"),
        "allocation_names": _uniq("allocation_name"),
    }
