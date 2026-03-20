# -*- coding: utf-8 -*-
"""
client_portfolio/ui.py
───────────────────────
Standalone Streamlit UI for "ניתוח תיק לקוח".
Rendered as an st.expander — zero interference with the rest of the app.

Entry point (one line in streamlit_app.py):
    from client_portfolio.ui import render_client_portfolio
    render_client_portfolio(df_long, product_type)

All session-state keys are prefixed  "cp_"  to avoid any collision.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd
import streamlit as st

# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_plotly(fig, key: str) -> None:
    try:
        st.plotly_chart(fig, use_container_width=True, key=key)
    except TypeError:
        st.plotly_chart(fig)


def _fmt(v, fmt="{:.1f}%"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    try:
        return fmt.format(float(v))
    except Exception:
        return str(v)


def _ils(v: float) -> str:
    if not v or math.isnan(v):
        return "—"
    if v >= 1_000_000:
        return f"₪{v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"₪{v/1_000:.0f}K"
    return f"₪{v:.0f}"


# ── Pull holdings from portfolio_analysis session state ───────────────────────

def _get_pf_holdings() -> list[dict]:
    """Pull holdings from the portfolio_analysis module's session state."""
    return st.session_state.get("pf_holdings", [])


# ── Enrich with cost column ───────────────────────────────────────────────────

def _enrich_costs(holdings: list[dict]) -> list[dict]:
    """
    Add annual_cost_pct from cp_costs session state (user-entered per product).
    """
    costs = st.session_state.get("cp_costs", {})
    enriched = []
    for h in holdings:
        hc = dict(h)
        hc["annual_cost_pct"] = costs.get(h["uid"], None)
        enriched.append(hc)
    return enriched


# ── Cost input UI ─────────────────────────────────────────────────────────────

def _render_cost_inputs(holdings: list[dict]) -> None:
    if not holdings:
        return
    st.markdown("#### 8. דמי ניהול שנתיים (אופציונלי)")
    st.caption("הזן דמי ניהול לכל מוצר (%) לחישוב עלות משוקללת. שדה ריק = לא ידוע.")
    costs = st.session_state.get("cp_costs", {})
    changed = False
    cols = st.columns(3)
    for i, h in enumerate(holdings):
        uid   = h["uid"]
        label = f"{h.get('provider','')} | {h.get('product_name','')}"
        val   = costs.get(uid, 0.0) or 0.0
        with cols[i % 3]:
            new_val = st.number_input(label, 0.0, 5.0, float(val), step=0.01,
                                      format="%.2f", key=f"cp_cost_{uid}")
            if new_val != val:
                costs[uid] = new_val
                changed = True
    if changed:
        st.session_state["cp_costs"] = costs


# ── Main entry point ──────────────────────────────────────────────────────────

def render_client_portfolio(df_long: pd.DataFrame, product_type: str) -> None:
    """
    Render the client portfolio analysis module as a top-level expander.
    Reads holdings from portfolio_analysis module (pf_holdings).
    """
    with st.expander("📊 ניתוח תיק לקוח", expanded=False):

        holdings_raw = _get_pf_holdings()
        if not holdings_raw:
            st.info(
                "💡 **כדי להפעיל ניתוח זה**, ייבא פורטפוליו בחלק "
                "**💼 ניתוח פורטפוליו נוכחי** שמתחת.",
                icon="📂",
            )
            return

        holdings = _enrich_costs(holdings_raw)
        import pandas as pd
        df = pd.DataFrame(holdings)
        for col in ["amount", "equity_pct", "foreign_pct", "fx_pct",
                    "illiquid_pct", "sharpe"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "excluded" not in df.columns:
            df["excluded"] = False

        from client_portfolio.charts import compute_totals
        totals = compute_totals(df)

        # ── Client name input ─────────────────────────────────────────────
        client_name = st.text_input(
            "שם הלקוח (לכותרת הדוח)",
            value=st.session_state.get("cp_client_name", ""),
            key="cp_client_name_input",
            placeholder="ישראל ישראלי",
        )
        st.session_state["cp_client_name"] = client_name

        # ── KPI strip ──────────────────────────────────────────────────────
        k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
        k1.metric("סך נכסים",       _ils(totals.get("total", 0)))
        k2.metric("מוצרים",          str(totals.get("n_products", 0)))
        k3.metric("מנהלים",          str(totals.get("n_managers", 0)))
        k4.metric("מניות (משוקלל)", _fmt(totals.get("equity")))
        k5.metric('חו"ל (משוקלל)',   _fmt(totals.get("foreign")))
        k6.metric('מט"ח (משוקלל)',   _fmt(totals.get("fx")))
        k7.metric("לא סחיר",         _fmt(totals.get("illiquid")))

        st.markdown("---")

        # ── Tabs ───────────────────────────────────────────────────────────
        tabs = st.tabs([
            "📊 גרפים",
            "💰 עלויות",
            "📋 טבלה מלאה",
            "📥 הורדת דוחות",
        ])

        with tabs[0]:
            _render_charts(df, totals)

        with tabs[1]:
            _render_cost_inputs(holdings_raw)
            # Re-enrich after cost input
            holdings = _enrich_costs(holdings_raw)
            df_cost = pd.DataFrame(holdings)
            for col in ["amount", "annual_cost_pct"]:
                if col in df_cost.columns:
                    df_cost[col] = pd.to_numeric(df_cost[col], errors="coerce")
            if "excluded" not in df_cost.columns:
                df_cost["excluded"] = False
            from client_portfolio.charts import chart_costs
            fig_cost = chart_costs(df_cost)
            if fig_cost.data:
                _safe_plotly(fig_cost, key="cp_costs_chart")
            else:
                st.info("הזן דמי ניהול בשדות למעלה לצפייה בגרף עלויות.")

        with tabs[2]:
            _render_full_table(df, totals)

        with tabs[3]:
            _render_downloads(df, totals, client_name, holdings_raw)


def _render_charts(df: pd.DataFrame, totals: dict) -> None:
    """Render portfolio analysis charts."""
    from client_portfolio.charts import (
        chart_by_manager, chart_stocks_bonds, chart_foreign_domestic,
        chart_fx_ils, chart_asset_breakdown, chart_annuity_capital,
    )

    # Row 1: manager donut + stocks/bonds/illiquid
    c1, c2 = st.columns(2)
    with c1:
        _safe_plotly(chart_by_manager(df), "cp_mgr")
    with c2:
        _safe_plotly(chart_stocks_bonds(df), "cp_sb")

    # Row 2: foreign/domestic + FX/ILS
    c3, c4 = st.columns(2)
    with c3:
        _safe_plotly(chart_foreign_domestic(df), "cp_fd")
    with c4:
        _safe_plotly(chart_fx_ils(df), "cp_fx")

    # Row 3: product-type donut + annuity vs capital
    st.markdown("---")
    c5, c6 = st.columns(2)
    with c5:
        _safe_plotly(chart_asset_breakdown(df), "cp_ab")
    with c6:
        # Annuity vs capital: auto from product_type, or manual slider
        has_product_type = ("product_type" in df.columns and
                            df["product_type"].notna().any() and
                            df["product_type"].ne("").any())
        if has_product_type:
            _safe_plotly(chart_annuity_capital(df), "cp_ac")
            ann_pct = None
        else:
            st.caption("ℹ️ לא זוהה סוג מוצר — הזן אחוז קצבה ידנית:")
            ann_pct = st.slider("% קצבה בתיק", 0, 100,
                                int(st.session_state.get("cp_ann_pct", 30)),
                                key="cp_ann_slider")
            st.session_state["cp_ann_pct"] = ann_pct
            _safe_plotly(chart_annuity_capital(df, manual_annuity_pct=ann_pct), "cp_ac")


def _render_full_table(df: pd.DataFrame, totals: dict) -> None:
    active = df[~df.get("excluded", pd.Series([False]*len(df))).astype(bool)]
    if active.empty:
        st.info("אין מוצרים להצגה.")
        return

    total_amt = active["amount"].sum()
    disp = active.copy()
    disp["משקל %"] = (disp["amount"] / total_amt * 100).round(1) if total_amt > 0 else 0
    disp = disp.rename(columns={
        "provider": "גוף", "product_name": "מוצר", "track": "מסלול",
        "product_type": "סוג", "amount": "סכום",
        "equity_pct": "מניות %", "foreign_pct": 'חו"ל %',
        "fx_pct": 'מט"ח %', "illiquid_pct": "לא סחיר %",
        "sharpe": "שארפ",
    })
    show_cols = [c for c in ["גוף", "מוצר", "מסלול", "סוג", "סכום", "משקל %",
                              "מניות %", 'חו"ל %', 'מט"ח %', "לא סחיר %", "שארפ"]
                 if c in disp.columns]
    st.dataframe(disp[show_cols].reset_index(drop=True),
                 use_container_width=True, hide_index=True)


def _render_downloads(df: pd.DataFrame, totals: dict,
                      client_name: str, holdings_raw: list[dict]) -> None:
    from client_portfolio.report_builder import build_html_report, build_notebook

    st.markdown("#### הורדת דוחות")

    # Enrich costs
    holdings_cost = _enrich_costs(holdings_raw)
    df_full = pd.DataFrame(holdings_cost)
    for col in ["amount","equity_pct","foreign_pct","fx_pct","illiquid_pct","sharpe","annual_cost_pct"]:
        if col in df_full.columns:
            df_full[col] = pd.to_numeric(df_full[col], errors="coerce")
    if "excluded" not in df_full.columns:
        df_full["excluded"] = False

    from client_portfolio.charts import compute_totals
    totals_full = compute_totals(df_full)

    dc1, dc2, dc3 = st.columns(3)

    # HTML report
    with dc1:
        html_bytes = build_html_report(df_full, client_name, totals_full)
        st.download_button(
            "📄 דוח HTML מעוצב",
            data=html_bytes,
            file_name=f"portfolio_report_{client_name or 'client'}.html",
            mime="text/html",
            key="cp_dl_html",
            help="דוח מעוצב שניתן להדפיס או לפתוח בדפדפן",
        )

    # Jupyter notebook
    with dc2:
        nb_bytes = build_notebook(df_full, client_name, totals_full)
        st.download_button(
            "📓 Jupyter Notebook",
            data=nb_bytes,
            file_name=f"portfolio_analysis_{client_name or 'client'}.ipynb",
            mime="application/json",
            key="cp_dl_nb",
            help="נוטבוק מוכן להפעלה — פתח ב-Jupyter / Google Colab ולחץ Run All",
        )

    # CSV
    with dc3:
        csv = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "⬇️ CSV",
            data=csv,
            file_name="portfolio_data.csv",
            mime="text/csv",
            key="cp_dl_csv",
        )

    st.markdown("---")
    st.markdown("""
**🚀 כיצד להשתמש בנוטבוק:**
1. הורד את הקובץ `.ipynb`
2. פתח ב-[Jupyter Lab](https://jupyter.org) / [VS Code](https://code.visualstudio.com) / [Google Colab](https://colab.research.google.com)
3. `Run All` — כל הגרפים ייוצרו אוטומטית
4. ייצא ל-PDF / HTML לצורך מצגת ללקוח

**📎 העתקה ל-Google Colab:**
- [Colab](https://colab.research.google.com) → `File → Upload notebook` → בחר את הקובץ
""")


# ── Full-page mode (called when product_type == "תיק לקוח") ──────────────────

def render_client_portfolio_page(df_long) -> None:
    """
    Full-page client portfolio UI — replaces the optimizer entirely.
    Called from streamlit_app.py when product_type == "תיק לקוח".
    """
    import streamlit as st
    # Import helpers from portfolio_analysis module
    from portfolio_analysis.ui import (
        _render_add_form, _render_edit_controls, _render_whatif,
    )
    from portfolio_analysis.models import import_from_session, set_holdings

    st.markdown("""
<div style='background:linear-gradient(135deg,#1F3A5F 0%,#3A7AFE 100%);
     border-radius:14px;padding:20px 28px;margin-bottom:18px;color:#fff'>
  <div style='font-size:22px;font-weight:900'>📊 ניתוח תיק לקוח</div>
  <div style='font-size:13px;opacity:0.8;margin-top:4px'>
    העלה דוח מסלקה · הוסף מוצרים ידנית · קבל ניתוח מקיף · הפק דוח מקצועי
  </div>
</div>
""", unsafe_allow_html=True)

    holdings_raw = _get_pf_holdings()

    # ── Step 1: Import / Add holdings ────────────────────────────────────
    with st.expander(
        f"{'✅' if holdings_raw else '📂'} שלב 1 — ייבוא פורטפוליו "
        f"({'%d מוצרים' % len(holdings_raw) if holdings_raw else 'ריק'})",
        expanded=not bool(holdings_raw),
    ):
        # ── File uploader (moved here from advanced settings) ─────────────
        st.markdown("##### 📂 העלאת דוח מסלקה (XLSX)")
        uploaded = st.file_uploader(
            'דוח מסלקה (XLSX/XLS)', type=["xlsx","xls"],
            key="cppage_upload", label_visibility="collapsed",
        )
        if uploaded:
            try:
                from streamlit_app import parse_clearing_report, _compute_baseline_from_holdings
            except Exception:
                # fallback: import directly from parent scope
                import importlib, sys
                # parse_clearing_report lives in streamlit_app — access via session workaround
                parse_clearing_report = st.session_state.get("_parse_clearing_fn")

            # Direct inline parser to avoid cross-module import issues
            if uploaded:
                raw_bytes = uploaded.read()
                try:
                    import io, math, re
                    import pandas as _pd
                    import numpy as _np
                    AMOUNT_ALIASES  = ["יתרה","ערך","סכום","balance","amount","שווי"]
                    FUND_ALIASES    = ["שם הקרן","קרן","שם מוצר","fund","product","שם הקופה","שם הגוף"]
                    MANAGER_ALIASES = ["מנהל","גוף מנהל","בית השקעות","manager","provider"]
                    TRACK_ALIASES   = ["מסלול","track","שם מסלול"]

                    def _to_f(v):
                        try:
                            s = str(v).replace(",","").replace("₪","").strip()
                            return float(s)
                        except Exception:
                            return float("nan")

                    xls = _pd.ExcelFile(io.BytesIO(raw_bytes))
                    all_recs = []
                    for sheet in xls.sheet_names:
                        try:
                            df_s = _pd.read_excel(xls, sheet_name=sheet, header=None)
                        except Exception:
                            continue
                        if df_s.empty or df_s.shape[0] < 2:
                            continue
                        header_idx = None
                        for i in range(min(10, len(df_s))):
                            row_vals = [str(v).strip().lower() for v in df_s.iloc[i].tolist()]
                            matches = sum(1 for v in row_vals
                                if any(a.lower() in v for a in AMOUNT_ALIASES+FUND_ALIASES+MANAGER_ALIASES))
                            if matches >= 2:
                                header_idx = i; break
                        if header_idx is None:
                            continue
                        dc = df_s.iloc[header_idx:].copy().reset_index(drop=True)
                        dc.columns = [str(c).strip() for c in dc.iloc[0].tolist()]
                        dc = dc.iloc[1:].reset_index(drop=True)
                        def _fc(aliases):
                            for col in dc.columns:
                                if any(a.lower() in col.lower() for a in aliases):
                                    return col
                            return None
                        fund_col    = _fc(FUND_ALIASES)
                        manager_col = _fc(MANAGER_ALIASES)
                        amount_col  = _fc(AMOUNT_ALIASES)
                        track_col   = _fc(TRACK_ALIASES)
                        if not (fund_col or manager_col) or not amount_col:
                            continue
                        for _, row in dc.iterrows():
                            fn = str(row.get(fund_col,"") or "").strip() if fund_col else ""
                            mn = str(row.get(manager_col,"") or "").strip() if manager_col else ""
                            tn = str(row.get(track_col,"") or "").strip() if track_col else ""
                            av = _to_f(row.get(amount_col, _np.nan))
                            if not fn and not mn: continue
                            if math.isnan(av) or av <= 0: continue
                            all_recs.append({"fund": fn or mn, "manager": mn or fn,
                                             "track": tn, "amount": av})
                    if all_recs:
                        total = sum(r["amount"] for r in all_recs)
                        for r in all_recs:
                            r["weight_pct"] = round(r["amount"]/total*100, 2) if total>0 else 0.0
                        st.session_state["portfolio_holdings"] = all_recs
                        st.session_state["portfolio_total"]    = total
                        st.session_state["portfolio_managers"] = list({r["manager"] for r in all_recs})
                        st.success(f"✅ טעון: {len(all_recs)} קרנות, ₪{total:,.0f}")
                        holdings_raw = _get_pf_holdings()  # refresh
                    else:
                        st.error("לא נמצאו נתונים בקובץ. ודא שהקובץ הוא דוח מסלקה עם עמודות שם קרן/מנהל וסכום.")
                except Exception as _e:
                    st.error(f"שגיאה בפרסור הקובץ: {_e}")

        st.markdown("---")

        # ── Import from already-parsed session state ──────────────────────
        raw_import = st.session_state.get("portfolio_holdings") or []
        if raw_import:
            existing_keys = {(h["provider"].lower(), h["product_name"].lower()) for h in holdings_raw}
            new_ct = sum(1 for r in raw_import
                         if (str(r.get("manager","")).lower(), str(r.get("fund","")).lower())
                         not in existing_keys)
            if new_ct > 0:
                if st.button(f"📥 ייבא {new_ct} מוצרים לניתוח", key="cppage_import", type="primary"):
                    added = import_from_session(st, df_long, "קרנות השתלמות")
                    if added:
                        st.success(f"✅ {added} מוצרים יובאו")
                        st.rerun()
            else:
                st.success(f"✅ {len(raw_import)} מוצרים מיובאים לניתוח")

        st.markdown("---")
        # Manual add form
        _render_add_form(holdings_raw, df_long)

    if not holdings_raw:
        st.info("💡 העלה דוח מסלקה בהגדרות המתקדמות (⚙️) או הוסף מוצרים ידנית למעלה.")
        return

    # ── Step 2: Edit & manage ─────────────────────────────────────────────
    with st.expander("✏️ שלב 2 — ניהול ועריכת מוצרים", expanded=False):
        if _render_edit_controls(holdings_raw, df_long):
            set_holdings(st, holdings_raw)
            st.rerun()

    # ── Step 3: Analysis ──────────────────────────────────────────────────
    holdings = _enrich_costs(holdings_raw)
    df = pd.DataFrame(holdings)
    for col in ["amount","equity_pct","foreign_pct","fx_pct","illiquid_pct","sharpe"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "excluded" not in df.columns:
        df["excluded"] = False

    from client_portfolio.charts import compute_totals
    totals = compute_totals(df)

    # Client name
    col_name, col_spacer = st.columns([2, 4])
    with col_name:
        client_name = st.text_input("שם הלקוח", value=st.session_state.get("cp_client_name",""),
                                     key="cppage_client_name", placeholder="ישראל ישראלי")
        st.session_state["cp_client_name"] = client_name

    # KPIs
    k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
    k1.metric("סך נכסים",       _ils(totals.get("total",0)))
    k2.metric("מוצרים",          str(totals.get("n_products",0)))
    k3.metric("מנהלים",          str(totals.get("n_managers",0)))
    k4.metric("מניות (משוקלל)", _fmt(totals.get("equity")))
    k5.metric('חו"ל (משוקלל)',   _fmt(totals.get("foreign")))
    k6.metric('מט"ח (משוקלל)',   _fmt(totals.get("fx")))
    k7.metric("לא סחיר",         _fmt(totals.get("illiquid")))

    st.markdown("---")

    # ── 1. Summary table — first thing the user sees ─────────────────────
    st.markdown("""
<div style='display:flex;align-items:center;gap:10px;margin:4px 0 14px 0;direction:rtl'>
  <div style='width:4px;height:24px;background:linear-gradient(180deg,#1F3A5F,#3A7AFE);
       border-radius:3px;flex-shrink:0'></div>
  <div style='font-size:17px;font-weight:800;color:#1F3A5F'>📋 סיכום תיק</div>
</div>""", unsafe_allow_html=True)
    _render_full_table(df, totals)

    st.markdown("---")

    # ── 2. Charts ─────────────────────────────────────────────────────────
    st.markdown("""
<div style='display:flex;align-items:center;gap:10px;margin:4px 0 14px 0;direction:rtl'>
  <div style='width:4px;height:24px;background:linear-gradient(180deg,#1F3A5F,#3A7AFE);
       border-radius:3px;flex-shrink:0'></div>
  <div style='font-size:17px;font-weight:800;color:#1F3A5F'>📈 ניתוח גרפי</div>
</div>""", unsafe_allow_html=True)
    _render_charts(df, totals)

    # Costs chart (shown only if cost data entered)
    holdings_cost = _enrich_costs(holdings_raw)
    df_cost = pd.DataFrame(holdings_cost)
    for col in ["amount","annual_cost_pct"]:
        if col in df_cost.columns:
            df_cost[col] = pd.to_numeric(df_cost[col], errors="coerce")
    if "excluded" not in df_cost.columns:
        df_cost["excluded"] = False
    from client_portfolio.charts import chart_costs
    fc = chart_costs(df_cost)
    if fc.data:
        st.markdown("""
<div style='display:flex;align-items:center;gap:10px;margin:12px 0 14px 0;direction:rtl'>
  <div style='width:4px;height:24px;background:linear-gradient(180deg,#1F3A5F,#3A7AFE);
       border-radius:3px;flex-shrink:0'></div>
  <div style='font-size:17px;font-weight:800;color:#1F3A5F'>💰 ניתוח עלויות</div>
</div>""", unsafe_allow_html=True)
        _safe_plotly(fc, "cppage_costs_main")

    st.markdown("---")

    # ── 3. Export — prominent, at the bottom ─────────────────────────────
    st.markdown("""
<div style='background:linear-gradient(135deg,#1F3A5F 0%,#3A7AFE 100%);
border-radius:12px;padding:20px 28px;margin:8px 0 16px 0;direction:rtl'>
  <div style='color:#fff;font-size:18px;font-weight:900;margin-bottom:4px'>
    📥 הפקת דוחות ויצוא
  </div>
  <div style='color:#93c5fd;font-size:12px'>
    דוח HTML מעוצב · Jupyter Notebook · NotebookLM Package · CSV
  </div>
</div>
""", unsafe_allow_html=True)
    _render_downloads_page(df, totals, client_name, holdings_raw)

    # ── Additional tabs for editing and what-if ───────────────────────────
    st.markdown("---")
    with st.expander("✏️ עריכת מוצרים ו-What-If", expanded=False):
        t_edit, t_costs2, t_whatif = st.tabs(["✏️ עריכה", "💰 עלויות", "🔀 What-If"])
        with t_edit:
            pass  # edit controls rendered in step 2 above
        with t_costs2:
            _render_cost_inputs(holdings_raw)
        with t_whatif:
            _render_whatif(holdings_raw)


def _render_downloads_page(df, totals, client_name, holdings_raw):
    """Enhanced export tab with NotebookLM package."""
    import streamlit as st
    from client_portfolio.report_builder import (
        build_html_report, build_notebook, build_notebooklm_package
    )

    holdings = _enrich_costs(holdings_raw)
    df_full = pd.DataFrame(holdings)
    for col in ["amount","equity_pct","foreign_pct","fx_pct","illiquid_pct","sharpe","annual_cost_pct"]:
        if col in df_full.columns:
            df_full[col] = pd.to_numeric(df_full[col], errors="coerce")
    if "excluded" not in df_full.columns:
        df_full["excluded"] = False

    from client_portfolio.charts import compute_totals
    totals_full = compute_totals(df_full)

    html_bytes = build_html_report(df_full, client_name, totals_full)
    nb_bytes   = build_notebook(df_full, client_name, totals_full)
    nlm_bytes  = build_notebooklm_package(df_full, client_name, totals_full)
    csv_bytes  = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    # ── Download buttons grid ─────────────────────────────────────────────
    dc1, dc2, dc3, dc4 = st.columns(4)
    with dc1:
        st.markdown("""
<div style='background:#F0F4FF;border-radius:10px;padding:14px 16px 10px 16px;
     text-align:center;direction:rtl;margin-bottom:8px'>
  <div style='font-size:22px;margin-bottom:4px'>📄</div>
  <div style='font-size:14px;font-weight:800;color:#1F3A5F'>דוח HTML</div>
  <div style='font-size:11px;color:#64748b;margin-top:3px'>מעוצב · מוכן להדפסה</div>
</div>""", unsafe_allow_html=True)
        st.download_button("📄 הורד דוח", data=html_bytes,
                           file_name=f"portfolio_{client_name or 'client'}.html",
                           mime="text/html", key="cppage_dl_html",
                           use_container_width=True, type="primary")
    with dc2:
        st.markdown("""
<div style='background:#F0F4FF;border-radius:10px;padding:14px 16px 10px 16px;
     text-align:center;direction:rtl;margin-bottom:8px'>
  <div style='font-size:22px;margin-bottom:4px'>📓</div>
  <div style='font-size:14px;font-weight:800;color:#1F3A5F'>Jupyter Notebook</div>
  <div style='font-size:11px;color:#64748b;margin-top:3px'>Run All ב-Colab → גרפים</div>
</div>""", unsafe_allow_html=True)
        st.download_button("📓 הורד Notebook", data=nb_bytes,
                           file_name=f"portfolio_{client_name or 'client'}.ipynb",
                           mime="application/json", key="cppage_dl_nb",
                           use_container_width=True, type="primary")
    with dc3:
        st.markdown("""
<div style='background:#F0F4FF;border-radius:10px;padding:14px 16px 10px 16px;
     text-align:center;direction:rtl;margin-bottom:8px'>
  <div style='font-size:22px;margin-bottom:4px'>🔬</div>
  <div style='font-size:14px;font-weight:800;color:#1F3A5F'>NotebookLM Package</div>
  <div style='font-size:11px;color:#64748b;margin-top:3px'>נתונים + פרומפט מצגת</div>
</div>""", unsafe_allow_html=True)
        st.download_button("🔬 הורד Package", data=nlm_bytes,
                           file_name=f"notebooklm_{client_name or 'client'}.md",
                           mime="text/markdown", key="cppage_dl_nlm",
                           use_container_width=True, type="primary")
    with dc4:
        st.markdown("""
<div style='background:#F0F4FF;border-radius:10px;padding:14px 16px 10px 16px;
     text-align:center;direction:rtl;margin-bottom:8px'>
  <div style='font-size:22px;margin-bottom:4px'>⬇️</div>
  <div style='font-size:14px;font-weight:800;color:#1F3A5F'>נתונים גולמיים</div>
  <div style='font-size:11px;color:#64748b;margin-top:3px'>CSV לעיבוד עצמאי</div>
</div>""", unsafe_allow_html=True)
        st.download_button("⬇️ הורד CSV", data=csv_bytes,
                           file_name="portfolio.csv", mime="text/csv",
                           key="cppage_dl_csv", use_container_width=True)

    st.markdown("---")

    # ── Usage instructions — two clean columns ────────────────────────────
    col_nlm, col_jup = st.columns(2)

    with col_nlm:
        st.markdown("""
<div style='background:#F8FAFF;border:1px solid #CBD5E1;border-radius:10px;
     padding:16px 18px;direction:rtl'>
  <div style='font-size:14px;font-weight:800;color:#1F3A5F;margin-bottom:10px'>
    🔬 NotebookLM — איך להשתמש
  </div>
  <ol style='margin:0;padding-right:18px;color:#374151;font-size:13px;line-height:2'>
    <li>הורד קובץ <strong>NotebookLM Package</strong> <code>(.md)</code></li>
    <li>פתח <a href="https://notebooklm.google.com" target="_blank">NotebookLM</a></li>
    <li><code>Add source</code> ← <code>Upload</code> ← בחר את הקובץ</li>
    <li>NotebookLM ינתח את התיק ויענה על כל שאלה</li>
  </ol>
</div>""", unsafe_allow_html=True)

    with col_jup:
        st.markdown("""
<div style='background:#F8FAFF;border:1px solid #CBD5E1;border-radius:10px;
     padding:16px 18px;direction:rtl'>
  <div style='font-size:14px;font-weight:800;color:#1F3A5F;margin-bottom:10px'>
    📓 Jupyter / Colab — איך להשתמש
  </div>
  <ol style='margin:0;padding-right:18px;color:#374151;font-size:13px;line-height:2'>
    <li>הורד <strong>Jupyter Notebook</strong> <code>(.ipynb)</code></li>
    <li>פתח <a href="https://colab.research.google.com" target="_blank">Google Colab</a>
        ← <code>File → Upload notebook</code></li>
    <li><code>Runtime → Run all</code> — כל הגרפים ייוצרו אוטומטית</li>
  </ol>
</div>""", unsafe_allow_html=True)
