# -*- coding: utf-8 -*-
"""
portfolio_analysis/ui.py
─────────────────────────
Full portfolio-analysis UI rendered as an st.expander at the app's bottom.

Entry point (one line in streamlit_app.py):
    from portfolio_analysis.ui import render_portfolio_analysis
    render_portfolio_analysis(df_long, product_type)

All session-state keys are prefixed  "pf_"  to avoid any collision.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from portfolio_analysis.models import (
    ALLOC_COLS, ALLOC_LABELS,
    get_holdings, set_holdings, holdings_to_df,
    compute_portfolio_summary, try_autofill,
    import_from_session, make_manual_holding, build_whatif_baseline,
)

# ── Constants ─────────────────────────────────────────────────────────────────

PRODUCT_TYPES = [
    "קרנות השתלמות", "פוליסות חיסכון", "קרנות פנסיה",
    "קופות גמל", "גמל להשקעה", "אחר",
]

_SRC_BADGE = {
    "imported":          ("🔵", "יובא"),
    "manual":            ("✏️", "ידני"),
    "auto_filled":       ("🟢", "מולא אוטו׳"),
    "missing":           ("🔴", "חסר"),
}

# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
<style>
.pf-table { width:100%; border-collapse:collapse; font-size:13px; direction:rtl; }
.pf-table th {
  background:#1F3A5F; color:#fff; padding:7px 10px;
  text-align:right; font-weight:700; white-space:nowrap;
}
.pf-table td { padding:6px 10px; border-bottom:1px solid #E5E7EB; vertical-align:middle; }
.pf-table tr:hover td { background:#F0F4FF; }
.pf-table tr.summary-row td {
  background:#EFF6FF; font-weight:800; border-top:2px solid #3A7AFE;
}
.pf-table tr.excluded td { opacity:0.45; text-decoration:line-through; }
.pf-badge { display:inline-block; border-radius:4px; padding:2px 7px;
            font-size:11px; font-weight:700; }
.pf-badge-missing  { background:#FEE2E2; color:#B91C1C; }
.pf-badge-imported { background:#DBEAFE; color:#1D4ED8; }
.pf-badge-manual   { background:#FEF3C7; color:#92400E; }
.pf-badge-auto     { background:#D1FAE5; color:#065F46; }
.pf-locked  { font-size:14px; cursor:default; }
</style>
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _nan_str(v, fmt="{:.1f}"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    try:
        return fmt.format(float(v)) + "%"
    except Exception:
        return str(v)


def _fmt_amount(v: float) -> str:
    if math.isnan(v) or v == 0:
        return "—"
    if v >= 1_000_000:
        return f"₪{v/1_000_000:.2f}M"
    return f"₪{v:,.0f}"


def _source_badge(alloc_src: str) -> str:
    icon, label = _SRC_BADGE.get(alloc_src, ("❓", alloc_src))
    css_cls = {
        "imported":    "pf-badge-imported",
        "manual":      "pf-badge-manual",
        "auto_filled": "pf-badge-auto",
        "missing":     "pf-badge-missing",
    }.get(alloc_src, "pf-badge-missing")
    return f"<span class='pf-badge {css_cls}'>{icon} {label}</span>"


# ── Summary metrics row ───────────────────────────────────────────────────────

def _render_summary_metrics(summary: dict) -> None:
    if not summary:
        return
    cols = st.columns(8)
    vals = [
        ("סך נכסים",    _fmt_amount(summary.get("total_amount", 0))),
        ("מוצרים",       str(summary.get("n_total", 0))),
        ("מולאו",         str(summary.get("n_complete", 0))),
        ("חסרים",        str(summary.get("n_missing", 0))),
        ("מניות (משוקלל)", _nan_str(summary.get("equity_pct"),   "{:.1f}")),
        ('חו"ל (משוקלל)',  _nan_str(summary.get("foreign_pct"),  "{:.1f}")),
        ('מט"ח (משוקלל)',  _nan_str(summary.get("fx_pct"),       "{:.1f}")),
        ("לא סחיר",       _nan_str(summary.get("illiquid_pct"),  "{:.1f}")),
    ]
    for col, (label, value) in zip(cols, vals):
        col.metric(label, value)


# ── Portfolio HTML table ──────────────────────────────────────────────────────

def _render_portfolio_table(holdings: list[dict], df_long: pd.DataFrame) -> None:
    if not holdings:
        st.info("הפורטפוליו ריק. ייבא מסלקה או הוסף מוצר ידנית.")
        return

    df = holdings_to_df(holdings)

    rows_html = ""
    for _, h in df.iterrows():
        uid         = h["uid"]
        excluded    = h.get("excluded", False)
        locked      = h.get("locked", False)
        alloc_src   = h.get("allocation_source", "missing")
        src_type    = h.get("source_type", "imported")
        row_cls     = "excluded" if excluded else ""

        lock_icon = "🔒" if locked else "🔓"
        excl_icon = "🚫" if excluded else ""

        rows_html += f"""
<tr class='{row_cls}'>
  <td>{excl_icon} {lock_icon} <span style='font-size:10px;color:#6B7280'>{uid[:6]}</span></td>
  <td><b>{h.get('provider','')}</b></td>
  <td>{h.get('product_name','')}</td>
  <td>{h.get('track','')}</td>
  <td style='text-align:center'><span style='font-size:10px;background:#F3F4F6;padding:2px 6px;border-radius:4px'>{h.get('product_type','')}</span></td>
  <td style='text-align:left'>{_fmt_amount(h['amount'])}</td>
  <td style='text-align:left'>{h['weight']:.1f}%</td>
  <td style='text-align:left'>{_nan_str(h.get('equity_pct'))}</td>
  <td style='text-align:left'>{_nan_str(h.get('foreign_pct'))}</td>
  <td style='text-align:left'>{_nan_str(h.get('fx_pct'))}</td>
  <td style='text-align:left'>{_nan_str(h.get('illiquid_pct'))}</td>
  <td style='text-align:left'>{_nan_str(h.get('sharpe'), '{:.2f}').replace('%','')}</td>
  <td>{_source_badge(alloc_src)}</td>
  <td style='font-size:11px;color:#9CA3AF'>{h.get('notes','')}</td>
</tr>"""

    # Summary row
    df_vis = df[~df["excluded"].astype(bool)] if "excluded" in df.columns else df
    total  = df_vis["amount"].sum()

    def _wsum(col):
        sub = df_vis[df_vis[col].notna()]
        if sub.empty:
            return "—"
        st_ = sub["amount"].sum()
        v   = (sub[col] * sub["amount"]).sum() / st_ if st_ > 0 else float("nan")
        return f"{v:.1f}%"

    rows_html += f"""
<tr class='summary-row'>
  <td colspan='5'>📊 סיכום (לא כולל מוצרים מוחרגים)</td>
  <td style='text-align:left'>{_fmt_amount(total)}</td>
  <td style='text-align:left'>100%</td>
  <td style='text-align:left'>{_wsum('equity_pct')}</td>
  <td style='text-align:left'>{_wsum('foreign_pct')}</td>
  <td style='text-align:left'>{_wsum('fx_pct')}</td>
  <td style='text-align:left'>{_wsum('illiquid_pct')}</td>
  <td colspan='3'></td>
</tr>"""

    table_html = f"""
{_CSS}
<div style='overflow-x:auto;direction:rtl'>
<table class='pf-table'>
<thead><tr>
  <th>⚙️</th><th>גוף</th><th>מוצר</th><th>מסלול</th><th>סוג</th>
  <th>סכום</th><th>משקל</th><th>מניות</th><th>חו"ל</th><th>מט"ח</th>
  <th>לא סחיר</th><th>שארפ</th><th>מקור</th><th>הערות</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>"""
    st.markdown(table_html, unsafe_allow_html=True)


# ── Edit controls ─────────────────────────────────────────────────────────────

def _render_edit_controls(holdings: list[dict], df_long: pd.DataFrame) -> bool:
    """
    Shows per-holding edit controls: lock, exclude, complete allocation, delete.
    Returns True if any change was made (triggers rerun).
    """
    if not holdings:
        return False

    changed = False
    st.markdown("#### 🛠️ עריכת מוצרים")

    for i, h in enumerate(holdings):
        uid        = h["uid"]
        alloc_src  = h.get("allocation_source", "missing")
        is_missing = alloc_src == "missing"
        is_excluded = h.get("excluded", False)
        is_locked   = h.get("locked", False)

        label = f"{h.get('provider','')} | {h.get('product_name','')} | {h.get('track','')}"
        with st.expander(
            f"{'🚫 ' if is_excluded else ''}{'🔒 ' if is_locked else ''}"
            f"{'🔴 ' if is_missing else ''}{label}",
            expanded=False,
        ):
            cols = st.columns([1, 1, 1, 1])
            with cols[0]:
                new_locked = st.checkbox("🔒 נעול (what-if)", value=is_locked, key=f"pf_lock_{uid}")
                if new_locked != is_locked:
                    holdings[i]["locked"] = new_locked
                    changed = True
            with cols[1]:
                new_excl = st.checkbox("🚫 החרג", value=is_excluded, key=f"pf_excl_{uid}")
                if new_excl != is_excluded:
                    holdings[i]["excluded"] = new_excl
                    changed = True
            with cols[2]:
                if st.button("🗑️ מחק", key=f"pf_del_{uid}"):
                    holdings.pop(i)
                    return True  # immediate rerun
            with cols[3]:
                # Auto-fill button
                if st.button("🔄 מלא אוטו׳", key=f"pf_auto_{uid}",
                             help="חפש נתונים מהגיליון הראשי ומלא אוטומטית"):
                    filled = try_autofill(h, df_long)
                    if filled.get("allocation_source") != alloc_src:
                        holdings[i] = filled
                        changed = True
                        st.toast("✅ מולא!")
                    else:
                        st.toast("לא נמצאו נתונים להשלמה")

            # Manual allocation completion
            if is_missing or st.session_state.get(f"pf_edit_alloc_{uid}", False):
                st.markdown("**השלם תמהיל ידנית:**")
                ac1, ac2, ac3, ac4 = st.columns(4)
                eq  = ac1.number_input("מניות %",   0.0, 100.0, float(h.get("equity_pct")   or 0), key=f"pf_eq_{uid}")
                fo  = ac2.number_input('חו"ל %',    0.0, 100.0, float(h.get("foreign_pct")  or 0), key=f"pf_fo_{uid}")
                fx  = ac3.number_input('מט"ח %',    0.0, 100.0, float(h.get("fx_pct")       or 0), key=f"pf_fx_{uid}")
                ill = ac4.number_input("לא סחיר %", 0.0, 100.0, float(h.get("illiquid_pct") or 0), key=f"pf_ill_{uid}")
                if st.button("💾 שמור תמהיל", key=f"pf_save_alloc_{uid}"):
                    holdings[i].update({
                        "equity_pct":   eq,
                        "foreign_pct":  fo,
                        "fx_pct":       fx,
                        "illiquid_pct": ill,
                        "allocation_source": "manual",
                    })
                    st.session_state[f"pf_edit_alloc_{uid}"] = False
                    changed = True
            elif alloc_src != "missing":
                if st.button("✏️ ערוך תמהיל", key=f"pf_edit_btn_{uid}"):
                    st.session_state[f"pf_edit_alloc_{uid}"] = True
                    changed = True

    return changed


# ── Add manual product form ───────────────────────────────────────────────────

def _render_add_form(holdings: list[dict], df_long: pd.DataFrame) -> bool:
    """
    Form for adding a new product manually.
    If provider+track match something in df_long, auto-fills allocation.
    Returns True if a product was added.
    """
    with st.expander("➕ הוסף מוצר ידנית", expanded=False):
        st.markdown("#### פרטי המוצר")
        r1c1, r1c2, r1c3 = st.columns(3)
        product_type = r1c1.selectbox("סוג מוצר", PRODUCT_TYPES, key="pf_add_type")
        provider     = r1c2.text_input("גוף מנהל",    key="pf_add_provider",
                                        placeholder="הראל, מגדל...")
        product_name = r1c3.text_input("שם מוצר/קרן", key="pf_add_name",
                                        placeholder="שם הקרן...")
        r2c1, r2c2, r2c3 = st.columns(3)
        track  = r2c1.text_input("מסלול",     key="pf_add_track",  placeholder="כללי, מנייתי...")
        amount = r2c2.number_input("סכום (₪)", min_value=0.0, step=1000.0, key="pf_add_amount")
        notes  = r2c3.text_input("הערות", key="pf_add_notes")

        # Smart auto-fill preview
        preview_h = None
        if provider or product_name:
            tmp = make_manual_holding(
                product_type, provider, product_name or provider, track,
                amount, None, None, None, None, None, notes
            )
            preview_h = try_autofill(tmp, df_long)
            if preview_h.get("allocation_source") == "auto_filled":
                st.success(
                    f"💡 נמצאו נתונים אוטומטיים: "
                    f"מניות {_nan_str(preview_h.get('equity_pct'))} | "
                    f"חו\"ל {_nan_str(preview_h.get('foreign_pct'))} | "
                    f"מט\"ח {_nan_str(preview_h.get('fx_pct'))} | "
                    f"לא סחיר {_nan_str(preview_h.get('illiquid_pct'))}"
                )

        # Manual allocation (shown if no auto-fill)
        use_manual_alloc = (
            preview_h is None or
            preview_h.get("allocation_source") != "auto_filled"
        )
        eq = fo = fx = ill = sh = None
        if use_manual_alloc:
            st.markdown("**תמהיל (אופציונלי):**")
            ac1, ac2, ac3, ac4, ac5 = st.columns(5)
            eq  = ac1.number_input("מניות %",    0.0, 100.0, 0.0, key="pf_add_eq")
            fo  = ac2.number_input('חו"ל %',     0.0, 100.0, 0.0, key="pf_add_fo")
            fx  = ac3.number_input('מט"ח %',     0.0, 100.0, 0.0, key="pf_add_fx")
            ill = ac4.number_input("לא סחיר %",  0.0, 100.0, 0.0, key="pf_add_ill")
            sh  = ac5.number_input("שארפ",        0.0,  5.0,  0.0, key="pf_add_sh",
                                   step=0.01, format="%.2f")

        if st.button("➕ הוסף לפורטפוליו", key="pf_add_submit", type="primary"):
            if not provider and not product_name:
                st.error("יש למלא לפחות שם גוף או שם מוצר.")
                return False
            if amount <= 0:
                st.error("יש להזין סכום חיובי.")
                return False

            if preview_h and preview_h.get("allocation_source") == "auto_filled":
                new_h = preview_h
                new_h["amount"] = float(amount)
                new_h["notes"]  = notes or ""
            else:
                new_h = make_manual_holding(
                    product_type, provider, product_name or provider,
                    track, amount, eq, fo, fx, ill, sh, notes
                )
            holdings.append(new_h)
            st.success(f"✅ {new_h['product_name']} נוסף לפורטפוליו.")
            return True
    return False


# ── Import from clearinghouse shortcut ───────────────────────────────────────

def _render_import_bar(holdings: list[dict], df_long: pd.DataFrame, product_type: str) -> bool:
    """Import button that pulls from the existing clearinghouse session state."""
    raw = st.session_state.get("portfolio_holdings") or []
    if not raw:
        return False

    existing_keys = {
        (h["provider"].lower(), h["product_name"].lower())
        for h in holdings
    }
    new_count = sum(
        1 for r in raw
        if (str(r.get("manager","")).lower(), str(r.get("fund","")).lower())
           not in existing_keys
    )

    if new_count == 0:
        st.caption(f"✅ {len(raw)} רשומות מהמסלקה כבר מיובאות.")
        return False

    import streamlit as _st
    if st.button(
        f"📥 ייבא {new_count} מוצרים מדו\"ח המסלקה לפורטפוליו",
        key="pf_import_btn",
    ):
        import streamlit as st2
        added = import_from_session(st2, df_long, product_type)
        if added:
            st.toast(f"✅ {added} מוצרים יובאו בהצלחה")
            return True
    return False


# ── What-if integration ───────────────────────────────────────────────────────

def _render_whatif(holdings: list[dict]) -> None:
    """
    Build a what-if baseline from portfolio and inject it into the optimizer.
    Respects locked / excluded flags.
    """
    if not holdings:
        return

    active = [h for h in holdings if not h.get("excluded", False)]
    if not active:
        st.info("כל המוצרים מוחרגים — אין בסיס לניתוח what-if.")
        return

    locked = [h for h in active if h.get("locked",  False)]
    free   = [h for h in active if not h.get("locked", False)]

    st.markdown("---")
    st.markdown("### 🔀 What-If — שימוש בפורטפוליו כבסיס לאופטימיזציה")

    lc1, lc2, lc3 = st.columns(3)
    lc1.metric("מוצרים פעילים", len(active))
    lc2.metric("נעולים (לא ישתנו)", len(locked))
    lc3.metric("פנויים לאופטימיזציה", len(free))

    if locked:
        st.caption(
            "**מוצרים נעולים** יישמרו בתיק ללא שינוי. "
            "האופטימיזציה תחפש הרכב מיטבי רק עבור המוצרים הפנויים."
        )
        locked_df = pd.DataFrame([{
            "גוף":    h["provider"],
            "מוצר":   h["product_name"],
            "מסלול":  h["track"],
            "סכום":   _fmt_amount(h["amount"]),
            "מניות":  _nan_str(h.get("equity_pct")),
            'חו"ל':   _nan_str(h.get("foreign_pct")),
        } for h in locked])
        st.dataframe(locked_df, use_container_width=True, hide_index=True)

    if not any(
        not math.isnan(h.get("equity_pct", float("nan")))
        for h in active
    ):
        st.warning(
            "⚠️ לא ניתן לחשב בסיס — אין נתוני אלוקציה. "
            "מלא תמהיל לפחות לחלק מהמוצרים."
        )
        return

    if st.button(
        "🚀 שלח לאופטימיזציה כפורטפוליו בסיס",
        key="pf_whatif_submit",
        type="primary",
        help="מחשב תמהיל משוקלל ושומר כ-baseline לאופטימיזציה הראשית",
    ):
        baseline = build_whatif_baseline(active)
        st.session_state["portfolio_baseline"] = baseline
        st.session_state["portfolio_total"] = sum(h["amount"] for h in active)
        st.session_state["portfolio_managers"] = list({h["provider"] for h in active})

        # Pre-fill optimizer targets from baseline
        if "targets" in st.session_state:
            if baseline.get("stocks"):
                st.session_state["targets"]["stocks"]   = round(baseline["stocks"], 1)
            if baseline.get("foreign"):
                st.session_state["targets"]["foreign"]  = round(baseline["foreign"], 1)
            if baseline.get("fx"):
                st.session_state["targets"]["fx"]       = round(baseline["fx"], 1)
            if baseline.get("illiquid"):
                st.session_state["targets"]["illiquid"] = round(baseline["illiquid"], 1)

        st.success(
            f"✅ הבסיס נשמר: "
            f"מניות {baseline.get('stocks',0):.1f}% | "
            f"חו\"ל {baseline.get('foreign',0):.1f}% | "
            f"מט\"ח {baseline.get('fx',0):.1f}% | "
            f"לא סחיר {baseline.get('illiquid',0):.1f}%\n\n"
            "גלול למעלה לאופטימיזציה."
        )


# ── Main entry point ──────────────────────────────────────────────────────────

def render_portfolio_analysis(df_long: pd.DataFrame, product_type: str) -> None:
    """
    Render the full portfolio analysis module as an st.expander.
    Call once at the bottom of streamlit_app.py, after render_history().
    """
    with st.expander("💼 ניתוח פורטפוליו נוכחי", expanded=False):

        holdings = get_holdings(st)

        # ── Top bar: import from clearinghouse ────────────────────────────
        if _render_import_bar(holdings, df_long, product_type):
            set_holdings(st, get_holdings(st))
            st.rerun()

        # ── Summary metrics ───────────────────────────────────────────────
        df = holdings_to_df(holdings)
        if not df.empty:
            summary = compute_portfolio_summary(df)
            _render_summary_metrics(summary)

        st.markdown("---")

        # ── Main tabs ─────────────────────────────────────────────────────
        t_table, t_edit, t_add, t_whatif = st.tabs([
            "📋 טבלת פורטפוליו",
            "✏️ עריכה וניהול",
            "➕ הוסף מוצר",
            "🔀 What-If",
        ])

        with t_table:
            _render_portfolio_table(holdings, df_long)
            if not df.empty:
                # Export
                exp_df = df.rename(columns={
                    "provider": "גוף", "product_name": "מוצר", "track": "מסלול",
                    "product_type": "סוג", "amount": "סכום", "weight": "משקל %",
                    "equity_pct": "מניות %", "foreign_pct": 'חו"ל %',
                    "fx_pct": 'מט"ח %', "illiquid_pct": "לא סחיר %",
                    "sharpe": "שארפ", "notes": "הערות", "source_type": "מקור",
                })
                st.download_button(
                    "⬇️ ייצוא CSV",
                    data=exp_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                    file_name="portfolio.csv", mime="text/csv",
                    key="pf_dl",
                )

        with t_edit:
            if _render_edit_controls(holdings, df_long):
                set_holdings(st, holdings)
                st.rerun()

            # Auto-fill all missing
            missing = [h for h in holdings if h.get("allocation_source") == "missing"]
            if missing:
                st.markdown("---")
                if st.button(
                    f"🔄 מלא אוטומטית את כל {len(missing)} המוצרים החסרים",
                    key="pf_autofill_all",
                ):
                    for i, h in enumerate(holdings):
                        if h.get("allocation_source") == "missing":
                            holdings[i] = try_autofill(h, df_long)
                    set_holdings(st, holdings)
                    st.rerun()

            # Clear all
            st.markdown("---")
            if holdings and st.button("🗑️ נקה פורטפוליו כולו", key="pf_clear_all"):
                set_holdings(st, [])
                st.rerun()

        with t_add:
            if _render_add_form(holdings, df_long):
                set_holdings(st, holdings)
                st.rerun()

        with t_whatif:
            _render_whatif(holdings)
