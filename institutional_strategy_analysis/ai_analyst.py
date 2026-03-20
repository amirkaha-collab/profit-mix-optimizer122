# -*- coding: utf-8 -*-
"""
institutional_strategy_analysis/ai_analyst.py
──────────────────────────────────────────────
Builds rich prompts from display-series data and calls an LLM provider for
institutional strategy analysis.
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


def _read_secret(name: str) -> str:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, "")


def _get_api_key() -> str:
    return _read_secret("ANTHROPIC_API_KEY") or _read_secret("OPENAI_API_KEY")


def _get_provider() -> str:
    if _read_secret("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _read_secret("OPENAI_API_KEY"):
        return "openai"
    return ""


def _call_anthropic(prompt: str, max_tokens: int = 1800, model: str = "claude-sonnet-4-5") -> tuple[str, Optional[str]]:
    api_key = _read_secret("ANTHROPIC_API_KEY")
    if not api_key:
        return "", ""
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            for blk in data.get("content", []):
                if blk.get("type") == "text":
                    return blk["text"].strip(), None
            return "", "תגובה ריקה מהמודל."
        if resp.status_code == 401:
            return "", "מפתח Anthropic לא תקין."
        if resp.status_code == 429:
            return "", "Anthropic החזיר חריגת קצב. נסה שוב בעוד כמה שניות."
        return "", f"שגיאת Anthropic: HTTP {resp.status_code}"
    except requests.exceptions.Timeout:
        return "", "תם הזמן הקצוב לבקשה ל-Anthropic."
    except Exception as e:
        return "", f"שגיאת תקשורת מול Anthropic: {e}"


def _call_openai(prompt: str, max_tokens: int = 1800, model: str = "gpt-5-mini") -> tuple[str, Optional[str]]:
    api_key = _read_secret("OPENAI_API_KEY")
    if not api_key:
        return "", ""
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a precise investment-strategy analyst. Write in Hebrew only, stay grounded in the provided data, and do not invent values.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_completion_tokens": max_tokens,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content.strip(), None
                if isinstance(content, list):
                    parts = [c.get("text", "") for c in content if isinstance(c, dict)]
                    return "\n".join([p for p in parts if p]).strip(), None
            return "", "תגובה ריקה מהמודל."
        if resp.status_code == 401:
            return "", "מפתח OpenAI לא תקין."
        if resp.status_code == 429:
            return "", "OpenAI החזיר חריגת קצב. נסה שוב בעוד כמה שניות."
        return "", f"שגיאת OpenAI: HTTP {resp.status_code}"
    except requests.exceptions.Timeout:
        return "", "תם הזמן הקצוב לבקשה ל-OpenAI."
    except Exception as e:
        return "", f"שגיאת תקשורת מול OpenAI: {e}"


def _call_llm(prompt: str, max_tokens: int = 1800) -> tuple[str, Optional[str]]:
    provider = _get_provider()
    if not provider:
        return "", "לא הוגדר מפתח AI. אפשר להוסיף ANTHROPIC_API_KEY או OPENAI_API_KEY ב‑Settings → Secrets."
    if provider == "anthropic":
        return _call_anthropic(prompt, max_tokens=max_tokens)
    return _call_openai(prompt, max_tokens=max_tokens)


def _format_series_for_prompt(df: pd.DataFrame, alloc: str, manager: str, track: str) -> str:
    sub = df[(df["allocation_name"] == alloc) & (df["manager"] == manager) & (df["track"] == track)].sort_values("date")
    if sub.empty:
        return "  (אין נתונים)"
    lines = []
    for _, row in sub.iterrows():
        freq_tag = "(ש)" if row.get("frequency") == "yearly" else "(ח)"
        date_str = row["date"].strftime("%Y") if row.get("frequency") == "yearly" else row["date"].strftime("%m/%Y")
        lines.append(f"  {date_str}{freq_tag}: {row['allocation_value']:.1f}%")
    return "\n".join(lines)


def _compute_stats(df: pd.DataFrame, alloc: str, manager: str, track: str) -> dict:
    sub = df[(df["allocation_name"] == alloc) & (df["manager"] == manager) & (df["track"] == track)].sort_values("date")
    if sub.empty or len(sub) < 2:
        return {}

    vals = sub["allocation_value"].astype(float)
    monthly_sub = sub[sub["frequency"] == "monthly"].copy() if "frequency" in sub.columns else sub.copy()
    monthly_vals = monthly_sub["allocation_value"].astype(float)
    diffs = monthly_vals.diff().dropna()

    slope = 0.0
    if len(monthly_vals) >= 3:
        x = np.arange(len(monthly_vals))
        slope = float(np.polyfit(x, monthly_vals.values, 1)[0])

    peak = vals.cummax()
    drawdown = vals - peak

    direction_changes = 0
    if len(diffs) >= 2:
        signs = np.sign(diffs.values)
        for i in range(1, len(signs)):
            if signs[i] != 0 and signs[i - 1] != 0 and signs[i] != signs[i - 1]:
                direction_changes += 1

    one_year_ago = sub[sub["date"] <= sub["date"].max() - pd.DateOffset(months=12)]
    yr_ago_val = float(one_year_ago.iloc[-1]["allocation_value"]) if not one_year_ago.empty else np.nan

    return {
        "current": round(float(vals.iloc[-1]), 2),
        "mean": round(float(vals.mean()), 2),
        "min": round(float(vals.min()), 2),
        "max": round(float(vals.max()), 2),
        "std": round(float(vals.std()), 2),
        "range_pp": round(float(vals.max() - vals.min()), 2),
        "slope_monthly": round(float(slope), 3),
        "max_drawdown": round(float(drawdown.min()), 2),
        "mom_avg": round(float(diffs.mean()), 2) if not diffs.empty else 0.0,
        "mom_max": round(float(diffs.abs().max()), 2) if not diffs.empty else 0.0,
        "direction_changes": int(direction_changes),
        "change_12m": round(float(vals.iloc[-1] - yr_ago_val), 2) if not np.isnan(yr_ago_val) else None,
        "n_monthly": int((sub["frequency"] == "monthly").sum()) if "frequency" in sub.columns else 0,
        "n_yearly": int((sub["frequency"] == "yearly").sum()) if "frequency" in sub.columns else 0,
        "date_first": sub["date"].min().strftime("%Y-%m"),
        "date_last": sub["date"].max().strftime("%Y-%m"),
    }


def _cross_manager_snapshot(df: pd.DataFrame, alloc: str) -> str:
    sub = df[df["allocation_name"] == alloc].copy()
    if sub.empty:
        return "  (אין נתונים)"
    idx = sub.groupby(["manager", "track"])["date"].idxmax()
    snap = sub.loc[idx].sort_values("allocation_value", ascending=False)
    return "\n".join(
        f"  {row['manager']} {row['track']}: {row['allocation_value']:.1f}%"
        for _, row in snap.iterrows()
    )


def _build_stats_table(display_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (manager, track, alloc), _ in display_df.groupby(["manager", "track", "allocation_name"]):
        stats = _compute_stats(display_df, alloc, manager, track)
        if stats:
            rows.append({"manager": manager, "track": track, "allocation": alloc, **stats})
    return pd.DataFrame(rows)


def _find_alloc(stats_df: pd.DataFrame, manager: str, alloc_keywords: list[str]) -> pd.DataFrame:
    result = stats_df[stats_df["manager"].eq(manager)].copy()
    alloc_lower = result["allocation"].astype(str).str.lower()
    mask = pd.Series(True, index=result.index)
    for kw in alloc_keywords:
        mask &= alloc_lower.str.contains(kw.lower(), na=False)
    return result[mask].copy()


def _pairwise_advantages(stats_df: pd.DataFrame, focus_manager: str, compare_manager: Optional[str]) -> str:
    if not focus_manager or not compare_manager or focus_manager == compare_manager or stats_df.empty:
        return "לא הוגדרה השוואת יתרונות ממוקדת בין שני גופים."

    lines = []
    for alloc in sorted(stats_df["allocation"].unique()):
        a = stats_df[(stats_df["manager"] == focus_manager) & (stats_df["allocation"] == alloc)]
        b = stats_df[(stats_df["manager"] == compare_manager) & (stats_df["allocation"] == alloc)]
        if a.empty or b.empty:
            continue
        row_a = a.sort_values(["track"]).iloc[0]
        row_b = b.sort_values(["track"]).iloc[0]
        lines.append(
            f"{alloc}: {focus_manager} נוכחי {row_a['current']}%, סטיית תקן {row_a['std']}, שינוי חודשי מקס׳ {row_a['mom_max']} | "
            f"{compare_manager} נוכחי {row_b['current']}%, סטיית תקן {row_b['std']}, שינוי חודשי מקס׳ {row_b['mom_max']}"
        )

    focus_fx = _find_alloc(stats_df, focus_manager, ["מט"])
    focus_foreign = _find_alloc(stats_df, focus_manager, ["חו"])
    comp_fx = _find_alloc(stats_df, compare_manager, ["מט"])
    comp_foreign = _find_alloc(stats_df, compare_manager, ["חו"])
    if not focus_fx.empty and not focus_foreign.empty and not comp_fx.empty and not comp_foreign.empty:
        a_foreign = float(focus_foreign.iloc[0]["current"])
        b_foreign = float(comp_foreign.iloc[0]["current"])
        fx_ratio_a = round(float(focus_fx.iloc[0]["current"] / a_foreign), 3) if a_foreign else None
        fx_ratio_b = round(float(comp_fx.iloc[0]["current"] / b_foreign), 3) if b_foreign else None
        if fx_ratio_a is not None and fx_ratio_b is not None:
            lines.append(
                f"יחס מט\"ח/חו\"ל (proxy חלקי לרמת גידור מט\"ח): {focus_manager}={fx_ratio_a}, {compare_manager}={fx_ratio_b}"
            )

    return "\n".join(lines) if lines else "אין מספיק חפיפה בין הגופים לצורך השוואת יתרונות מדויקת."


def _build_full_prompt(display_df: pd.DataFrame, context: dict) -> str:
    managers = context.get("managers", [])
    tracks = context.get("tracks", [])
    allocation_names = context.get("allocation_names", [])
    sel_range = context.get("selected_range", "הכל")
    user_question = (context.get("user_question") or "").strip()
    focus_manager = context.get("focus_manager") or ""
    compare_manager = context.get("compare_manager") or ""

    series_blocks = []
    all_stats = []
    for mgr in managers:
        for trk in tracks:
            for alloc in allocation_names:
                stats = _compute_stats(display_df, alloc, mgr, trk)
                if not stats:
                    continue
                all_stats.append({**stats, "manager": mgr, "track": trk, "alloc": alloc})
                series_text = _format_series_for_prompt(display_df, alloc, mgr, trk)
                change_12m_txt = "n/a" if stats["change_12m"] is None else f"{stats['change_12m']:+.2f}pp"
                series_blocks.append(
                    f"[{mgr} | {trk} | {alloc}]\n"
                    f"  טווח: {stats['date_first']} – {stats['date_last']}\n"
                    f"  נוכחי: {stats['current']}% | ממוצע: {stats['mean']}% | מינ׳: {stats['min']}% | מקס׳: {stats['max']}%\n"
                    f"  סטד: {stats['std']}pp | טווח: {stats['range_pp']}pp | מגמה חודשית: {stats['slope_monthly']:+.2f}pp/חודש\n"
                    f"  שינוי חודשי ממוצע: {stats['mom_avg']:+.2f}pp | שינוי חודשי מקס׳: {stats['mom_max']:.2f}pp | שינוי 12 חו׳: {change_12m_txt}\n"
                    f"  שינויי כיוון: {stats['direction_changes']} | מקסימום ירידה מהשיא: {stats['max_drawdown']}pp\n"
                    f"  נתונים ({stats['n_monthly']} חודשי, {stats['n_yearly']} שנתי):\n{series_text}"
                )

    cross_blocks = [
        f"השוואה — {alloc} (ערך נוכחי לפי מנהל):\n{_cross_manager_snapshot(display_df, alloc)}"
        for alloc in allocation_names
    ]

    risk_lines = []
    for s in all_stats:
        alloc_name = str(s["alloc"])
        if (("סחיר" in alloc_name and "לא" in alloc_name) or "לא-סחיר" in alloc_name or "לא סחיר" in alloc_name) and s["current"] > 30:
            risk_lines.append(f"⚠️ {s['manager']} {s['track']}: חשיפה גבוהה ללא-סחיר ({s['current']}%).")
        if s["std"] > 5:
            risk_lines.append(f"⚠️ {s['manager']} {s['track']} | {alloc_name}: תנודתיות גבוהה (STD={s['std']}pp, max move={s['mom_max']}pp).")
        if s["slope_monthly"] < -0.5:
            risk_lines.append(f"📉 {s['manager']} {s['track']} | {alloc_name}: מגמת ירידה מובהקת ({s['slope_monthly']:+.2f}pp לחודש).")
        if s["slope_monthly"] > 0.5:
            risk_lines.append(f"📈 {s['manager']} {s['track']} | {alloc_name}: מגמת עלייה מובהקת ({s['slope_monthly']:+.2f}pp לחודש).")
        if s["direction_changes"] >= 3:
            risk_lines.append(f"🔁 {s['manager']} {s['track']} | {alloc_name}: מספר גבוה של שינויי כיוון ({s['direction_changes']}).")

    stats_df = _build_stats_table(display_df)
    stats_table_text = stats_df.to_csv(index=False) if not stats_df.empty else "(אין טבלת סטטיסטיקה)"
    pairwise_text = _pairwise_advantages(stats_df, focus_manager, compare_manager)

    user_request_section = (
        f"שאלת המשתמש לניתוח המותאם: {user_question}"
        if user_question
        else "שאלת משתמש ספציפית לא הוזנה. יש לתת ניתוח כללי עמוק ומבוסס-נתונים."
    )

    series_section = "\n\n".join(series_blocks) or "(אין נתונים)"
    cross_section = "\n\n".join(cross_blocks) or "(אין נתונים)"
    risk_section = "\n".join(risk_lines) if risk_lines else "לא זוהו אותות סיכון חריגים."

    return f"""אתה אנליסט השקעות ישראלי בכיר המתמחה בגופים מוסדיים ובניתוח מדיניות השקעה לאורך זמן.
לפניך נתוני חשיפות/אלוקציה היסטוריים של גופים מוסדיים. (ש) = נתון שנתי, (ח) = נתון חודשי. טווח מוצג: {sel_range}.

המטרה: לענות בצורה עמוקה על כל שאלה שהמשתמש שואל לגבי הנתונים, כולל:
- שינוי חודשי, סטיית תקן, שינויי כיוון, דינמיות של מנהל ההשקעות
- השוואה בין גופים ומסלולים
- האם גוף מגדיל/מקטין סיכון, נע בין ישראל לחו"ל, מגדיל/מקטין לא-סחיר
- האם רמת המט"ח גבוהה או נמוכה יחסית לחשיפת חו"ל, כ-proxy חלקי לרמת גידור
- יתרונות יחסיים של גוף שנבחר לעומת גוף אחר

{user_request_section}
גוף מועדף להשוואה: {focus_manager or 'לא הוגדר'}
גוף להשוואה מולו: {compare_manager or 'לא הוגדר'}

══════════════════════════════
נתוני סדרות זמן לפי מנהל/מסלול/רכיב:
══════════════════════════════
{series_section}

══════════════════════════════
טבלת סטטיסטיקה מרוכזת:
══════════════════════════════
{stats_table_text}

══════════════════════════════
השוואה בין מנהלים (ערך נוכחי):
══════════════════════════════
{cross_section}

══════════════════════════════
אותות סיכון אוטומטיים:
══════════════════════════════
{risk_section}

══════════════════════════════
השוואת יתרונות ממוקדת בין שני גופים:
══════════════════════════════
{pairwise_text}

══════════════════════════════
הנחיות כתיבה:
══════════════════════════════
1. כתוב בעברית בלבד.
2. הישען רק על הנתונים שסופקו.
3. כאשר הנתונים לא מספיקים, אמור זאת במפורש.
4. אם המשתמש שאל שאלה ספציפית — ענה עליה ראשית, ואז תן הרחבות מועילות מהנתונים.
5. כאשר יש השוואה בין שני גופים — ציין גם יתרונות של הגוף שנבחר וגם נקודות שבהן הגוף האחר עדיף.
6. התייחס במידת האפשר לסטטיסטיקות: סטיית תקן, שינוי חודשי ממוצע, שינוי חודשי מקסימלי, שינוי 12 חודשים, שינויי כיוון, טווח, מגמה.
7. כאשר יש גם חו"ל וגם מט"ח — התייחס ליחס ביניהם בזהירות כ-indication חלקי לרמת גידור מט"ח, בלי לקבוע מסקנה מוחלטת.

כתוב עם הכותרות הבאות בדיוק:
## תשובה לשאלת המשתמש
## ניתוח לפי גוף ומסלול
## השוואה בין גופים
## דינמיות, שינויי כיוון וסיכון
## מט"ח מול חו"ל וגידור
## יתרונות וחסרונות יחסיים
## תובנה אסטרטגית
"""


@dataclass
class AnalysisResult:
    raw_text: str = ""
    sections: dict = field(default_factory=dict)
    error: Optional[str] = None
    provider: str = ""

    def parse_sections(self):
        if not self.raw_text:
            return
        current_title = "כללי"
        current_lines: list[str] = []
        for line in self.raw_text.splitlines():
            if line.startswith("## "):
                if current_lines:
                    self.sections[current_title] = "\n".join(current_lines).strip()
                current_title = line[3:].strip()
                current_lines = []
            else:
                current_lines.append(line)
        if current_lines:
            self.sections[current_title] = "\n".join(current_lines).strip()


def run_ai_analysis(display_df: pd.DataFrame, context: dict) -> AnalysisResult:
    if display_df.empty:
        return AnalysisResult(error="אין נתונים לניתוח.")
    prompt = _build_full_prompt(display_df, context)
    text, err = _call_llm(prompt, max_tokens=2200)
    result = AnalysisResult(raw_text=text, error=err, provider=_get_provider())
    if text:
        result.parse_sections()
    return result
