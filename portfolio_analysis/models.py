# -*- coding: utf-8 -*-
"""
portfolio_analysis/models.py
─────────────────────────────
Data model and session-state helpers for the unified portfolio module.

All session-state keys are prefixed  "pf_"  to avoid collisions.
"""
from __future__ import annotations

import uuid
from typing import Optional

import numpy as np
import pandas as pd

# ── Schema ────────────────────────────────────────────────────────────────────
#
# Each portfolio holding is a dict with these keys:
#
#   uid             : str   – stable unique id (uuid4 hex)
#   product_type    : str   – "קרנות השתלמות" | "פוליסות חיסכון" | etc.
#   provider        : str   – manager / institution name
#   product_name    : str   – fund / product name
#   track           : str   – track name within the fund
#   amount          : float – current balance (₪)
#   weight          : float – computed: amount / total  (0–100 %)
#   equity_pct      : float – stocks exposure %     (nan = unknown)
#   foreign_pct     : float – foreign exposure %    (nan = unknown)
#   fx_pct          : float – FX exposure %         (nan = unknown)
#   illiquid_pct    : float – illiquid exposure %   (nan = unknown)
#   sharpe          : float – Sharpe ratio           (nan = unknown)
#   notes           : str
#   source_type     : "imported" | "manual"
#   allocation_source: "imported" | "auto_filled" | "manual" | "missing"
#   locked          : bool  – lock in what-if (do not optimise weight)
#   excluded        : bool  – exclude from what-if run

ALLOC_COLS = ["equity_pct", "foreign_pct", "fx_pct", "illiquid_pct"]
ALLOC_LABELS = {
    "equity_pct":   "מניות (%)",
    "foreign_pct":  'חו"ל (%)',
    "fx_pct":       'מט"ח (%)',
    "illiquid_pct": "לא סחיר (%)",
}

# ── Session-state helpers ─────────────────────────────────────────────────────

STATE_KEY = "pf_holdings"   # list[dict]


def _init(st) -> None:
    """Ensure all pf_ keys exist in session state."""
    st.session_state.setdefault(STATE_KEY, [])


def get_holdings(st) -> list[dict]:
    _init(st)
    return st.session_state[STATE_KEY]


def set_holdings(st, holdings: list[dict]) -> None:
    st.session_state[STATE_KEY] = holdings


def _new_uid() -> str:
    return uuid.uuid4().hex[:12]


# ── DataFrame conversion ──────────────────────────────────────────────────────

def holdings_to_df(holdings: list[dict]) -> pd.DataFrame:
    """Convert list-of-dicts to a display DataFrame with computed weight column."""
    if not holdings:
        return pd.DataFrame()

    df = pd.DataFrame(holdings)
    total = df["amount"].sum()
    df["weight"] = (df["amount"] / total * 100).round(2) if total > 0 else 0.0
    return df


# ── Weighted portfolio summary ────────────────────────────────────────────────

def compute_portfolio_summary(df: pd.DataFrame) -> dict:
    """
    Compute weighted average allocations across all holdings that have data.
    Returns dict with keys: total_amount, n_total, n_complete, n_missing,
    n_manual, equity_pct, foreign_pct, fx_pct, illiquid_pct, sharpe.
    """
    if df.empty:
        return {}

    total = df["amount"].sum()
    n_total = len(df)
    n_complete = int(df[ALLOC_COLS].notna().all(axis=1).sum())
    n_missing  = n_total - n_complete
    n_manual   = int((df["source_type"] == "manual").sum()) if "source_type" in df.columns else 0

    # Weighted average (weight by amount, only rows with data)
    result = {
        "total_amount": total,
        "n_total":      n_total,
        "n_complete":   n_complete,
        "n_missing":    n_missing,
        "n_manual":     n_manual,
    }

    for col in ALLOC_COLS + ["sharpe"]:
        sub = df[df[col].notna()].copy() if col in df.columns else pd.DataFrame()
        if sub.empty:
            result[col] = float("nan")
        else:
            sub_total = sub["amount"].sum()
            result[col] = float((sub[col] * sub["amount"]).sum() / sub_total) if sub_total > 0 else float("nan")

    return result


# ── Auto-fill from df_long ─────────────────────────────────────────────────────

def try_autofill(holding: dict, df_long: pd.DataFrame) -> dict:
    """
    Try to fill missing allocation data from df_long (app's main data).
    Returns a *copy* of the holding with filled values and updated allocation_source.
    """
    h = dict(holding)
    if not any(np.isnan(h.get(c, float("nan"))) for c in ALLOC_COLS):
        return h  # already complete

    # column mapping: df_long uses stocks/foreign/fx/illiquid
    COL_MAP = {
        "equity_pct":   "stocks",
        "foreign_pct":  "foreign",
        "fx_pct":       "fx",
        "illiquid_pct": "illiquid",
    }

    # Lookup strategies: exact fund → exact manager → fuzzy manager
    match = pd.DataFrame()
    fund_name = str(h.get("product_name", "")).strip().lower()
    mgr_name  = str(h.get("provider", "")).strip().lower()

    if fund_name and not df_long.empty:
        m = df_long[df_long["fund"].str.lower().str.strip() == fund_name]
        if not m.empty:
            match = m.head(1)

    if match.empty and mgr_name and not df_long.empty:
        m = df_long[df_long["manager"].str.lower().str.strip() == mgr_name]
        if not m.empty:
            # prefer same track if available
            track = str(h.get("track", "")).strip().lower()
            if track:
                mt = m[m["track"].str.lower().str.strip() == track]
                match = mt.head(1) if not mt.empty else m.head(1)
            else:
                match = m.head(1)

    if match.empty and mgr_name:
        for word in mgr_name.split():
            if len(word) > 2:
                m = df_long[df_long["manager"].str.lower().str.contains(word, na=False)]
                if not m.empty:
                    match = m.head(1)
                    break

    if match.empty:
        return h  # nothing found

    row = match.iloc[0]
    filled = False
    for pf_col, app_col in COL_MAP.items():
        if np.isnan(h.get(pf_col, float("nan"))) and app_col in row.index:
            val = row[app_col]
            if not (isinstance(val, float) and np.isnan(val)):
                h[pf_col] = float(val)
                filled = True

    if filled and h.get("allocation_source") == "missing":
        h["allocation_source"] = "auto_filled"

    if "sharpe" in row.index and np.isnan(h.get("sharpe", float("nan"))):
        sv = row["sharpe"]
        if not (isinstance(sv, float) and np.isnan(sv)):
            h["sharpe"] = float(sv)

    return h


# ── Import from existing portfolio_holdings session-state ─────────────────────

def import_from_session(st, df_long: pd.DataFrame, product_type: str) -> int:
    """
    Pull holdings from st.session_state["portfolio_holdings"] (set by the
    existing import-from-clearinghouse feature) and merge into pf_holdings.
    Returns number of new records added.
    """
    _init(st)
    raw = st.session_state.get("portfolio_holdings") or []
    if not raw:
        return 0

    existing_uids = {h["uid"] for h in get_holdings(st)}
    # Deduplicate by fund+manager+track
    existing_keys = {
        (h["provider"].lower(), h["product_name"].lower(), h["track"].lower())
        for h in get_holdings(st)
    }

    added = 0
    for r in raw:
        key = (
            str(r.get("manager", "")).lower(),
            str(r.get("fund", "")).lower(),
            str(r.get("track", "")).lower(),
        )
        if key in existing_keys:
            continue

        h = {
            "uid":              _new_uid(),
            "product_type":     product_type,
            "provider":         str(r.get("manager", "")),
            "product_name":     str(r.get("fund", "")),
            "track":            str(r.get("track", "")),
            "amount":           float(r.get("amount", 0)),
            "weight":           0.0,
            "equity_pct":       float("nan"),
            "foreign_pct":      float("nan"),
            "fx_pct":           float("nan"),
            "illiquid_pct":     float("nan"),
            "sharpe":           float("nan"),
            "notes":            "",
            "source_type":      "imported",
            "allocation_source": "missing",
            "locked":           False,
            "excluded":         False,
        }
        h = try_autofill(h, df_long)
        get_holdings(st).append(h)
        existing_keys.add(key)
        added += 1

    return added


def make_manual_holding(
    product_type: str, provider: str, product_name: str, track: str,
    amount: float, equity: float, foreign: float, fx: float,
    illiquid: float, sharpe: float, notes: str,
) -> dict:
    """Create a new manually-entered holding dict."""
    def _f(v): return float(v) if v not in (None, "") else float("nan")
    return {
        "uid":              _new_uid(),
        "product_type":     product_type,
        "provider":         provider,
        "product_name":     product_name,
        "track":            track,
        "amount":           float(amount) if amount else 0.0,
        "weight":           0.0,
        "equity_pct":       _f(equity),
        "foreign_pct":      _f(foreign),
        "fx_pct":           _f(fx),
        "illiquid_pct":     _f(illiquid),
        "sharpe":           _f(sharpe),
        "notes":            notes or "",
        "source_type":      "manual",
        "allocation_source": "manual" if not any(
            (v in (None, "")) for v in [equity, foreign, fx, illiquid]
        ) else "missing",
        "locked":           False,
        "excluded":         False,
    }


def build_whatif_baseline(holdings: list[dict]) -> dict:
    """
    Build the baseline dict expected by the existing optimizer:
      {foreign, stocks, fx, illiquid, sharpe, service}
    Uses only non-excluded holdings.
    """
    active = [h for h in holdings if not h.get("excluded", False)]
    if not active:
        return {}

    total = sum(h["amount"] for h in active)
    if total <= 0:
        return {}

    result = {"foreign": 0.0, "stocks": 0.0, "fx": 0.0,
              "illiquid": 0.0, "sharpe": 0.0, "service": 0.0}

    for h in active:
        w = h["amount"] / total
        _safe_add = lambda key, val: result.__setitem__(key, result[key] + val * w) \
                    if not np.isnan(val) else None
        _safe_add("stocks",   h.get("equity_pct",   float("nan")))
        _safe_add("foreign",  h.get("foreign_pct",  float("nan")))
        _safe_add("fx",       h.get("fx_pct",        float("nan")))
        _safe_add("illiquid", h.get("illiquid_pct",  float("nan")))
        _safe_add("sharpe",   h.get("sharpe",        float("nan")))

    return result
