# -*- coding: utf-8 -*-
"""
institutional_strategy_analysis/ui.py
───────────────────────────────────────
Self-contained Streamlit UI for "ניתוח אסטרטגיות מוסדיים".
Renders as an st.expander at the bottom of the main app.

Entry point (one line in streamlit_app.py):
    from institutional_strategy_analysis.ui import render_institutional_analysis
    render_institutional_analysis()

All session-state keys are prefixed "isa_" to avoid any collision.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

# ── Sheet URL ─────────────────────────────────────────────────────────────────
# ▼▼▼  Set your Google Sheets URL here  ▼▼▼
ISA_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1e9zjj1OWMYqUYoK6YFYvYwOnN7qbydYDyArHbn8l9pE/edit"
)
# ▲▲▲─────────────────────────────────────────────────────────────────────────

# ── Lazy imports (never execute at import time) ───────────────────────────────

def _load_data():
    from institutional_strategy_analysis.loader     import load_raw_blocks
    from institutional_strategy_analysis.series_builder import get_time_bounds
    import streamlit as st

    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached(url: str):
        return load_raw_blocks(url)

    return _cached(ISA_SHEET_URL)


def _build_series(df_y, df_m, rng, custom_start, filters):
    from institutional_strategy_analysis.series_builder import build_display_series
    return build_display_series(df_y, df_m, rng, custom_start, filters)


def _options(df_y, df_m):
    from institutional_strategy_analysis.series_builder import get_available_options
    return get_available_options(df_y, df_m)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_plotly(fig, key=None):
    try:
        st.plotly_chart(fig, use_container_width=True, key=key)
    except TypeError:
        st.plotly_chart(fig)


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def _clamp(val: date, lo: date, hi: date) -> date:
    return max(lo, min(hi, val))


# ── Debug panel ───────────────────────────────────────────────────────────────

def _render_debug(df_yearly, df_monthly, debug_info, errors):
    with st.expander("🛠️ מידע אבחון (debug)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("גליונות שנטענו", len(debug_info))
            st.metric("שורות שנתי", len(df_yearly))
            st.metric("שורות חודשי", len(df_monthly))
        with col2:
            if not df_yearly.empty:
                yr = df_yearly["date"]
                st.metric("טווח שנתי", f"{yr.min().year} – {yr.max().year}")
            if not df_monthly.empty:
                mr = df_monthly["date"]
                st.metric("טווח חודשי",
                          f"{mr.min().strftime('%Y-%m')} – {mr.max().strftime('%Y-%m')}")

        if debug_info:
            rows = []
            for d in debug_info:
                rows.append({
                    "גליון": d.get("sheet", "?"),
                    "header row": d.get("header_row", "?"),
                    "freq col": d.get("freq_col", "—"),
                    "שורות שנתיות": d.get("yearly_rows", 0),
                    "שורות חודשיות": d.get("monthly_rows", 0),
                    "טווח שנתי": d.get("yearly_range", "—"),
                    "טווח חודשי": d.get("monthly_range", "—"),
                    "שגיאה": d.get("error", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if errors:
            for e in errors:
                st.warning(e)



# ── AI Analysis renderer ──────────────────────────────────────────────────────

_SECTION_ICONS = {
    "ניתוח לפי גוף ומסלול":     "🏢",
    "השוואה בין גופים":          "⚖️",
    "ניתוח סיכון":               "🎯",
    "קשר לתשואה היסטורית":      "📈",
    "יתרונות וחסרונות לפי גוף":  "✅",
    "תובנה אסטרטגית":            "💡",
}


def _render_ai_analysis(display_df, context: dict):
    """Render AI analysis panel below the chart. Shows only after button press."""
    if not st.session_state.get("isa_run_ai"):
        return

    st.markdown("---")
    st.markdown("### 🤖 ניתוח AI מעמיק — אסטרטגיות מוסדיים")
    if context.get("user_question"):
        st.caption(f"שאלת המשתמש: {context['user_question']}")
    if context.get("focus_manager") and context.get("compare_manager"):
        st.caption(f"השוואת יתרונות ממוקדת: {context['focus_manager']} מול {context['compare_manager']}")

    cache_key = "isa_ai_result"
    filter_sig = str(sorted(context.items()))
    if st.session_state.get("isa_ai_sig") != filter_sig:
        st.session_state.pop(cache_key, None)
        st.session_state["isa_ai_sig"] = filter_sig

    if cache_key not in st.session_state:
        with st.spinner("Claude מנתח את הנתונים... (עד 30 שניות)"):
            try:
                from institutional_strategy_analysis.ai_analyst import run_ai_analysis
                result = run_ai_analysis(display_df, context)
                st.session_state[cache_key] = result
            except Exception as e:
                st.error(f"שגיאה בניתוח AI: {e}")
                st.session_state["isa_run_ai"] = False
                return

    result = st.session_state[cache_key]

    if getattr(result, "provider", ""):
        provider_label = "Anthropic" if result.provider == "anthropic" else "OpenAI"
        st.caption(f"מנוע הניתוח הפעיל: {provider_label}")

    if result.error:
        st.error(f"⚠️ {result.error}")
        if st.button("נסה שוב", key="isa_ai_retry"):
            st.session_state.pop(cache_key, None)
            st.rerun()
        return

    if not result.sections:
        st.markdown(result.raw_text)
    else:
        for title, body in result.sections.items():
            icon = _SECTION_ICONS.get(title, "📋")
            expanded = title in ("ניתוח לפי גוף ומסלול", "תובנה אסטרטגית")
            with st.expander(f"{icon} {title}", expanded=expanded):
                st.markdown(body)

    col_a, col_b, _ = st.columns([1, 1, 4])
    with col_a:
        st.download_button(
            "⬇️ שמור ניתוח",
            data=result.raw_text.encode("utf-8"),
            file_name="isa_ai_analysis.txt",
            mime="text/plain",
            key="isa_dl_ai",
        )
    with col_b:
        if st.button("🔄 רענן", key="isa_ai_refresh",
                     help="הרץ מחדש את הניתוח עם הנתונים הנוכחיים"):
            st.session_state.pop(cache_key, None)
            st.session_state.pop("isa_ai_sig", None)
            st.rerun()




def _render_auto_track_summary(display_df, sel_mgr, sel_tracks, sel_allocs, sel_range):
    st.markdown("#### 🧠 סיכום אוטומטי למסלול נבחר")
    st.caption("בחירה של גוף ומסלול תייצר סיכום אוטומטי בעברית, עם דגש על דינמיות, שינויי כיוון, סיכון, חו\"ל/מט\"ח ותזוזות מרכזיות לאורך התקופה.")

    c1, c2 = st.columns(2)
    with c1:
        summary_manager = st.selectbox(
            "גוף לסיכום",
            options=sel_mgr,
            key="isa_auto_summary_manager",
        )
    with c2:
        summary_tracks = sorted(display_df[display_df["manager"].eq(summary_manager)]["track"].dropna().unique().tolist())
        if not summary_tracks:
            st.info("אין מסלולים זמינים לסיכום בבחירה הנוכחית.")
            return
        summary_track = st.selectbox(
            "מסלול לסיכום",
            options=summary_tracks,
            key="isa_auto_summary_track",
        )

    summary_df = display_df[(display_df["manager"] == summary_manager) & (display_df["track"] == summary_track)].copy()
    if summary_df.empty:
        st.info("אין נתונים לסיכום בבחירה הנוכחית.")
        return

    try:
        from institutional_strategy_analysis.ai_analyst import _get_api_key, run_ai_analysis
        has_key = bool(_get_api_key())
    except Exception:
        has_key = False

    if not has_key:
        st.info(
            "להפעלת הסיכום האוטומטי הוסף `ANTHROPIC_API_KEY` או `OPENAI_API_KEY` ב-Settings → Secrets",
            icon="🔑",
        )
        return

    context = {
        "managers": [summary_manager],
        "tracks": [summary_track],
        "allocation_names": sorted(summary_df["allocation_name"].dropna().unique().tolist()),
        "selected_range": sel_range,
        "date_min": summary_df["date"].min().strftime("%Y-%m") if not summary_df.empty else "",
        "date_max": summary_df["date"].max().strftime("%Y-%m") if not summary_df.empty else "",
        "user_question": (
            f"כתוב סיכום אוטומטי ומובנה עבור {summary_manager} במסלול {summary_track}. "
            "התייחס למבנה החשיפות, לדינמיות של הגוף, לשינויים חודשיים, לשינויי כיוון, "
            "למגמות בולטות, לעלייה או ירידה בסיכון, למעבר בין ישראל לחו\"ל, "
            "להתנהגות ברכיב הלא-סחיר, וליחס מט\"ח מול חו\"ל כאשר הנתונים מאפשרים זאת. "
            "סיים ב-3-5 תובנות תמציתיות וברורות שמתאימות להצגה ללקוח או ליועץ."
        ),
        "focus_manager": summary_manager,
        "compare_manager": None,
    }

    summary_sig = str(sorted(context.items()))
    cache_key = "isa_auto_summary_result"
    sig_key = "isa_auto_summary_sig"
    if st.session_state.get(sig_key) != summary_sig:
        st.session_state.pop(cache_key, None)
        st.session_state[sig_key] = summary_sig

    if cache_key not in st.session_state:
        with st.spinner("מייצר סיכום אוטומטי למסלול הנבחר..."):
            try:
                st.session_state[cache_key] = run_ai_analysis(summary_df, context)
            except Exception as e:
                st.error(f"שגיאה ביצירת הסיכום האוטומטי: {e}")
                return

    result = st.session_state[cache_key]
    if getattr(result, "provider", ""):
        provider_label = "Anthropic" if result.provider == "anthropic" else "OpenAI"
        st.caption(f"מנוע הסיכום הפעיל: {provider_label}")

    if result.error:
        st.error(f"⚠️ {result.error}")
        return

    preferred_order = [
        "תשובה לשאלת המשתמש",
        "ניתוח לפי גוף ומסלול",
        "דינמיות, שינויי כיוון וסיכון",
        "מט\"ח מול חו\"ל וגידור",
        "תובנה אסטרטגית",
    ]

    shown_any = False
    for title in preferred_order:
        body = result.sections.get(title)
        if body:
            shown_any = True
            if title == "תובנה אסטרטגית":
                st.success(body)
            else:
                with st.expander(title, expanded=(title == "תשובה לשאלת המשתמש")):
                    st.markdown(body)

    if not shown_any:
        st.markdown(result.raw_text)

    cdl, cref, _ = st.columns([1, 1, 4])
    with cdl:
        st.download_button(
            "⬇️ שמור סיכום",
            data=result.raw_text.encode("utf-8"),
            file_name=f"isa_auto_summary_{summary_manager}_{summary_track}.txt",
            mime="text/plain",
            key="isa_auto_summary_dl",
        )
    with cref:
        if st.button("🔄 רענן סיכום", key="isa_auto_summary_refresh"):
            st.session_state.pop(cache_key, None)
            st.session_state.pop(sig_key, None)
            st.rerun()

# ── Main entry point ──────────────────────────────────────────────────────────

def render_institutional_analysis():
    """Render the full "ניתוח אסטרטגיות מוסדיים" section."""

    with st.expander("📐 ניתוח אסטרטגיות מוסדיים", expanded=False):

        # ── Load data ─────────────────────────────────────────────────────
        with st.spinner("טוען נתונים..."):
            try:
                df_yearly, df_monthly, debug_info, errors = _load_data()
            except Exception as e:
                st.error(f"שגיאת טעינה: {e}")
                return

        if df_yearly.empty and df_monthly.empty:
            st.error("לא נטענו נתונים. בדוק את קישור הגיליון ואת הרשאות הגישה.")
            for e in errors:
                st.warning(e)
            return

        _render_debug(df_yearly, df_monthly, debug_info, errors)

        # ── Available options ─────────────────────────────────────────────
        opts = _options(df_yearly, df_monthly)

        # ── Filters ───────────────────────────────────────────────────────
        st.markdown("#### 🎛️ סינון")
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            sel_mgr = st.multiselect(
                "מנהל השקעות",
                options=opts["managers"],
                default=opts["managers"],
                help="בחר גוף מוסדי אחד או יותר. הנתונים מציגים את אסטרטגיית האלוקציה שלהם לאורך זמן.",
                key="isa_managers",
            )
        with fc2:
            avail_tracks = sorted({
                t for df in (df_yearly, df_monthly) if not df.empty
                for t in df[df["manager"].isin(sel_mgr)]["track"].unique()
            }) if sel_mgr else opts["tracks"]
            sel_tracks = st.multiselect(
                "מסלול",
                options=avail_tracks,
                default=avail_tracks,
                help="בחר מסלול השקעה — כגון כללי, מנייתי. מסלול כללי מאזן בין כמה נכסים.",
                key="isa_tracks",
            )
        with fc3:
            avail_allocs = sorted({
                a for df in (df_yearly, df_monthly) if not df.empty
                for a in df[
                    df["manager"].isin(sel_mgr) & df["track"].isin(sel_tracks)
                ]["allocation_name"].unique()
            }) if sel_mgr and sel_tracks else opts["allocation_names"]
            sel_allocs = st.multiselect(
                "רכיב אלוקציה",
                options=avail_allocs,
                default=avail_allocs[:5] if len(avail_allocs) > 5 else avail_allocs,
                help='בחר רכיבי חשיפה — למשל מניות, חו"ל, מט"ח, לא-סחיר.',
                key="isa_allocs",
            )

        # Time range
        rng_c, cust_c = st.columns([3, 2])
        with rng_c:
            sel_range = st.radio(
                "טווח זמן",
                options=["הכל", "YTD", "1Y", "3Y", "5Y", "מותאם אישית"],
                index=0, horizontal=True,
                label_visibility="collapsed",
                key="isa_range",
            )
            st.caption(
                "⏱️ **טווח זמן** — YTD ו-1Y משתמשים בנתונים חודשיים בלבד. "
                "3Y/5Y/הכל משלבים חודשי + שנתי."
            )
        with cust_c:
            custom_start = None
            if sel_range == "מותאם אישית":
                from institutional_strategy_analysis.series_builder import get_time_bounds
                min_d, max_d = get_time_bounds(df_yearly, df_monthly)
                custom_start = st.date_input(
                    "מתאריך", value=min_d.date(),
                    min_value=min_d.date(), max_value=max_d.date(),
                    key="isa_custom_start",
                )

        if not sel_mgr or not sel_tracks or not sel_allocs:
            st.info("יש לבחור לפחות מנהל, מסלול ורכיב אחד.")
            return

        # ── Build display series ──────────────────────────────────────────
        filters = {"managers": sel_mgr, "tracks": sel_tracks,
                   "allocation_names": sel_allocs}

        display_df = _build_series(df_yearly, df_monthly, sel_range, custom_start, filters)

        if display_df.empty:
            if sel_range in ("YTD", "1Y") and df_monthly.empty:
                st.warning(
                    "⚠️ לא נמצאו נתונים חודשיים. "
                    "YTD ו-1Y דורשים נתונים חודשיים. "
                    "נסה 'הכל' או '3Y' לקבלת נתונים שנתיים."
                )
            else:
                st.warning("אין נתונים לסינון הנוכחי.")
            return

        # Quick stats row
        n_dates  = display_df["date"].nunique()
        n_yearly = (display_df["frequency"] == "yearly").sum()  if "frequency" in display_df.columns else 0
        n_monthly = (display_df["frequency"] == "monthly").sum() if "frequency" in display_df.columns else 0
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("נקודות זמן", n_dates)
        sc2.metric("נתונים חודשיים", n_monthly // max(1, display_df["allocation_name"].nunique()))
        sc3.metric("נתונים שנתיים",  n_yearly  // max(1, display_df["allocation_name"].nunique()))

        # ── Tabs ──────────────────────────────────────────────────────────
        t_ts, t_snap, t_delta, t_heat, t_stats, t_rank = st.tabs([
            "📈 סדרת זמן",
            "📍 Snapshot",
            "🔄 שינוי / Delta",
            "🌡️ Heatmap",
            "📊 סטטיסטיקות",
            "🏆 דירוג",
        ])

        # ── Tab 1: Time series ────────────────────────────────────────────
        with t_ts:
            from institutional_strategy_analysis.charts import build_timeseries
            fig = build_timeseries(display_df)
            _safe_plotly(fig, key="isa_ts")
            st.caption(
                "קווים מלאים = נתונים חודשיים | קווים מקווקוים = נתונים שנתיים. "
                "שנים שמכוסות על ידי נתונים חודשיים לא מוצגות כשנתיות."
            )

            _render_auto_track_summary(display_df, sel_mgr, sel_tracks, sel_allocs, sel_range)

            st.markdown("#### שאלת AI חופשית")
            ai_q_col, ai_f_col, ai_c_col = st.columns([3, 1.2, 1.2])
            with ai_q_col:
                isa_user_question = st.text_area(
                    "מה לבדוק בנתונים?",
                    value=st.session_state.get("isa_user_question", ""),
                    placeholder='למשל: מי המנהל הכי דינאמי במסלול כללי, מי שינה כיוון הכי הרבה, ומי מגדר מט"ח יותר ביחס לחשיפת חו"ל?',
                    height=90,
                    key="isa_user_question",
                )
            with ai_f_col:
                focus_opts = ["—"] + sel_mgr
                focus_manager = st.selectbox(
                    "גוף נבחר",
                    options=focus_opts,
                    index=0,
                    help="אם תבחר גוף, ה-AI יבליט את היתרונות והחסרונות שלו ביחס לאחרים.",
                    key="isa_focus_manager",
                )
            with ai_c_col:
                compare_candidates = ["—"] + [m for m in sel_mgr if m != focus_manager] if focus_manager != "—" else ["—"] + sel_mgr
                compare_manager = st.selectbox(
                    "להשוות מול",
                    options=compare_candidates,
                    index=0,
                    help="גוף ייחוס להשוואה ישירה.",
                    key="isa_compare_manager",
                )

            col_dl, col_ai, _ = st.columns([1, 2, 3])
            with col_dl:
                st.download_button("⬇️ CSV", data=_csv(display_df),
                                   file_name="isa_timeseries.csv", mime="text/csv",
                                   key="isa_dl_ts")
            with col_ai:
                # Check for API key BEFORE showing the button
                try:
                    from institutional_strategy_analysis.ai_analyst import _get_api_key
                    _has_key = bool(_get_api_key())
                except Exception:
                    _has_key = False

                if _has_key:
                    if st.button("🤖 ניתוח AI מעמיק", key="isa_ai_btn",
                                 help='הניתוח יתייחס גם לסטטיסטיקות, דינמיות, שינויי כיוון, חו"ל/מט"ח והשוואה בין גופים'):
                        st.session_state["isa_run_ai"] = True
                else:
                    st.info(
                        "להפעלת ניתוח AI הוסף `ANTHROPIC_API_KEY` או `OPENAI_API_KEY` "
                        "ב-Settings → Secrets",
                        icon="🔑",
                    )
            _render_ai_analysis(display_df, {
                "managers": sel_mgr,
                "tracks": sel_tracks,
                "allocation_names": sel_allocs,
                "selected_range": sel_range,
                "date_min": display_df["date"].min().strftime("%Y-%m") if not display_df.empty else "",
                "date_max": display_df["date"].max().strftime("%Y-%m") if not display_df.empty else "",
                "user_question": isa_user_question,
                "focus_manager": None if focus_manager == "—" else focus_manager,
                "compare_manager": None if compare_manager == "—" else compare_manager,
            })

        # ── Tab 2: Snapshot ───────────────────────────────────────────────
        with t_snap:
            max_d = display_df["date"].max().date()
            min_d = display_df["date"].min().date()
            snap_date = st.date_input(
                "תאריך Snapshot",
                value=max_d, min_value=min_d, max_value=max_d,
                help="מציג את הערך האחרון הידוע עד לתאריך שנבחר.",
                key="isa_snap_date",
            )
            from institutional_strategy_analysis.charts import build_snapshot
            _safe_plotly(build_snapshot(display_df, pd.Timestamp(snap_date)), key="isa_snap")

            snap_df = display_df[display_df["date"] <= pd.Timestamp(snap_date)]
            if not snap_df.empty:
                i = snap_df.groupby(["manager", "track", "allocation_name"])["date"].idxmax()
                tbl = snap_df.loc[i][["manager", "track", "allocation_name",
                                       "allocation_value", "date"]].copy()
                tbl["date"] = tbl["date"].dt.strftime("%Y-%m")
                tbl.columns = ["מנהל", "מסלול", "רכיב", "ערך (%)", "תאריך"]
                st.dataframe(tbl.sort_values("ערך (%)", ascending=False)
                               .reset_index(drop=True),
                             use_container_width=True, hide_index=True)

        # ── Tab 3: Delta ──────────────────────────────────────────────────
        with t_delta:
            min_d = display_df["date"].min().date()
            max_d = display_df["date"].max().date()
            dc1, dc2 = st.columns(2)
            with dc1:
                date_a = st.date_input("תאריך A (מוצא)",
                                       value=_clamp(max_d - timedelta(days=365), min_d, max_d),
                                       min_value=min_d, max_value=max_d,
                                       help="תאריך ההתחלה להשוואה.",
                                       key="isa_da")
            with dc2:
                date_b = st.date_input("תאריך B (יעד)", value=max_d,
                                       min_value=min_d, max_value=max_d,
                                       help="תאריך הסיום להשוואה.",
                                       key="isa_db")
            if date_a >= date_b:
                st.warning("תאריך A חייב להיות לפני B.")
            else:
                from institutional_strategy_analysis.charts import build_delta
                fig_d, delta_tbl = build_delta(display_df,
                                                pd.Timestamp(date_a),
                                                pd.Timestamp(date_b))
                _safe_plotly(fig_d, key="isa_delta")
                if not delta_tbl.empty:
                    st.dataframe(delta_tbl.reset_index(drop=True),
                                 use_container_width=True, hide_index=True)
                    col_dl2, _ = st.columns([1, 5])
                    with col_dl2:
                        st.download_button("⬇️ CSV", data=_csv(delta_tbl),
                                           file_name="isa_delta.csv", mime="text/csv",
                                           key="isa_dl_delta")

        # ── Tab 4: Heatmap ────────────────────────────────────────────────
        with t_heat:
            from institutional_strategy_analysis.charts import build_heatmap
            heat_df = display_df.copy()
            if display_df["date"].nunique() > 48:
                cutoff = display_df["date"].max() - pd.DateOffset(months=48)
                heat_df = display_df[display_df["date"] >= cutoff]
                st.caption("מוצגים 48 חודשים אחרונים. בחר 'הכל' לצפייה מלאה.")
            _safe_plotly(build_heatmap(heat_df), key="isa_heat")

        # ── Tab 5: Summary stats ──────────────────────────────────────────
        with t_stats:
            from institutional_strategy_analysis.charts import build_summary_stats
            stats = build_summary_stats(display_df)
            if stats.empty:
                st.info("אין מספיק נתונים לסטטיסטיקה.")
            else:
                st.dataframe(stats.reset_index(drop=True),
                             use_container_width=True, hide_index=True)
                col_dl3, _ = st.columns([1, 5])
                with col_dl3:
                    st.download_button("⬇️ CSV", data=_csv(stats),
                                       file_name="isa_stats.csv", mime="text/csv",
                                       key="isa_dl_stats")

        # ── Tab 6: Ranking ────────────────────────────────────────────────
        with t_rank:
            from institutional_strategy_analysis.charts import build_ranking
            if display_df["allocation_name"].nunique() > 1:
                rank_alloc = st.selectbox(
                    "רכיב לדירוג",
                    options=sorted(display_df["allocation_name"].unique()),
                    help="בחר רכיב שלפיו יוצג הדירוג החודשי.",
                    key="isa_rank_alloc",
                )
                rank_df = display_df[display_df["allocation_name"] == rank_alloc]
            else:
                rank_df = display_df

            _safe_plotly(
                build_ranking(rank_df,
                              title=f"דירוג מנהלים — {rank_df['allocation_name'].iloc[0]}"
                              if not rank_df.empty else "דירוג"),
                key="isa_rank",
            )

            # Volatility table
            if not rank_df.empty:
                vol = []
                for (mgr, trk), g in rank_df.groupby(["manager", "track"]):
                    chg = g.sort_values("date")["allocation_value"].diff().dropna()
                    vol.append({
                        "מנהל": mgr, "מסלול": trk,
                        "תנודתיות (STD)": round(chg.std(), 3) if len(chg) > 1 else float("nan"),
                        "שינוי מקסימלי": round(chg.abs().max(), 3) if not chg.empty else float("nan"),
                    })
                if vol:
                    st.caption("תנודתיות לפי מנהל:")
                    st.dataframe(
                        pd.DataFrame(vol).sort_values("תנודתיות (STD)", ascending=False)
                          .reset_index(drop=True),
                        use_container_width=True, hide_index=True,
                    )

        # ── Raw data ──────────────────────────────────────────────────────
        with st.expander("📋 נתונים גולמיים", expanded=False):
            disp = display_df.copy()
            if "date" in disp.columns:
                disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(disp.reset_index(drop=True),
                         use_container_width=True, hide_index=True)
            st.download_button("⬇️ ייצוא כל הנתונים", data=_csv(display_df),
                               file_name="isa_all.csv", mime="text/csv",
                               key="isa_dl_all")
