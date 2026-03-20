# -*- coding: utf-8 -*-
"""
client_portfolio/report_builder.py
───────────────────────────────────
Generates two deliverables from the portfolio DataFrame:

1. HTML report  — styled, RTL, print-ready, can be embedded in a presentation.
2. Jupyter notebook (.ipynb) — copy-paste ready, with code cells that
   reproduce every chart and a markdown cell with a full client narrative.

Both are returned as bytes for st.download_button.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(v, fmt="{:.1f}%"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    try:
        return fmt.format(float(v))
    except Exception:
        return str(v)

def _ils(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    v = float(v)
    if v >= 1_000_000:
        return f"₪{v/1_000_000:.2f}M"
    return f"₪{v:,.0f}"

def _now():
    return datetime.now().strftime("%d/%m/%Y %H:%M")


# ── HTML Report ───────────────────────────────────────────────────────────────

_HTML_STYLE = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;700;900&display=swap');
  * { box-sizing: border-box; }
  body { direction: rtl; font-family: 'Heebo', 'Segoe UI', sans-serif;
         background: #F8FAFC; color: #1E293B; margin: 0; padding: 24px; }
  .report-header { background: linear-gradient(135deg, #1F3A5F 0%, #3A7AFE 100%);
    color: #fff; border-radius: 16px; padding: 36px 40px; margin-bottom: 32px; }
  .report-header h1 { font-size: 32px; font-weight: 900; margin: 0 0 8px; }
  .report-header p  { font-size: 14px; opacity: 0.8; margin: 0; }
  .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }
  .kpi-card { background: #fff; border-radius: 12px; padding: 20px 18px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.06); text-align: center; }
  .kpi-value { font-size: 26px; font-weight: 900; color: #1F3A5F; }
  .kpi-label { font-size: 12px; color: #64748B; margin-top: 4px; }
  .section { background: #fff; border-radius: 12px; padding: 24px;
             box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 24px; }
  .section h2 { font-size: 18px; color: #1F3A5F; border-bottom: 2px solid #EFF6FF;
                padding-bottom: 10px; margin: 0 0 16px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { background: #1F3A5F; color: #fff; padding: 9px 12px; text-align: right; font-weight: 700; }
  td { padding: 8px 12px; border-bottom: 1px solid #E2E8F0; }
  tr:hover td { background: #F0F7FF; }
  tr.total-row td { background: #EFF6FF; font-weight: 800; border-top: 2px solid #3A7AFE; }
  .badge { display: inline-block; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 700; }
  .badge-ok   { background: #D1FAE5; color: #065F46; }
  .badge-warn { background: #FEF3C7; color: #92400E; }
  .badge-miss { background: #FEE2E2; color: #B91C1C; }
  .footer { text-align: center; font-size: 11px; color: #94A3B8; margin-top: 32px; }
  @media print { body { background: #fff; padding: 0; } .section { break-inside: avoid; } }
</style>
"""

def build_html_report(df: pd.DataFrame, client_name: str = "", totals: dict = None) -> bytes:
    """Generate a styled HTML report as bytes."""
    active = df[~df.get("excluded", pd.Series([False]*len(df))).astype(bool)] if not df.empty else df
    if totals is None:
        from client_portfolio.charts import compute_totals
        totals = compute_totals(df) if not df.empty else {}

    total_amount = totals.get("total", 0)
    n_products   = totals.get("n_products", 0)
    n_managers   = totals.get("n_managers", 0)

    client_hdr = f"<div style='font-size:14px;opacity:0.85'>לקוח: {client_name}</div>" if client_name else ""

    # KPI cards
    kpis = [
        (_ils(total_amount), "סך נכסים"),
        (str(n_products), "מוצרים"),
        (str(n_managers), "מנהלים"),
        (_fmt(totals.get("equity")), "מניות (משוקלל)"),
        (_fmt(totals.get("foreign")), 'חו"ל (משוקלל)'),
        (_fmt(totals.get("fx")), 'מט"ח (משוקלל)'),
        (_fmt(totals.get("illiquid")), "לא סחיר (משוקלל)"),
        (_fmt(totals.get("cost"), "{:.2f}%") if not _nan_val(totals.get("cost")) else "—", "עלות שנתית משוקללת"),
    ]
    kpi_html = '<div class="kpi-grid">' + "".join(
        f'<div class="kpi-card"><div class="kpi-value">{v}</div><div class="kpi-label">{l}</div></div>'
        for v, l in kpis
    ) + "</div>"

    # Holdings table
    rows = ""
    if not active.empty:
        total_w = active["amount"].sum()
        for _, h in active.iterrows():
            w = h["amount"] / total_w * 100 if total_w > 0 else 0
            alloc_src = h.get("allocation_source", "missing")
            badge_cls = {"imported": "badge-ok", "auto_filled": "badge-ok",
                         "manual": "badge-warn", "missing": "badge-miss"}.get(alloc_src, "badge-miss")
            badge_txt = {"imported": "✓ יובא", "auto_filled": "✓ אוטו",
                         "manual": "✏ ידני", "missing": "⚠ חסר"}.get(alloc_src, "?")
            cost_cell = _fmt(h.get("annual_cost_pct"), "{:.2f}%") if "annual_cost_pct" in h.index else "—"
            rows += f"""<tr>
              <td><b>{h.get('provider','')}</b></td>
              <td>{h.get('product_name','')}</td>
              <td>{h.get('track','')}</td>
              <td style='text-align:center'><span style='font-size:11px;background:#F1F5F9;padding:2px 8px;border-radius:4px'>{h.get('product_type','')}</span></td>
              <td style='text-align:left'>{_ils(h['amount'])}</td>
              <td style='text-align:center'>{w:.1f}%</td>
              <td style='text-align:center'>{_fmt(h.get('equity_pct'))}</td>
              <td style='text-align:center'>{_fmt(h.get('foreign_pct'))}</td>
              <td style='text-align:center'>{_fmt(h.get('fx_pct'))}</td>
              <td style='text-align:center'>{_fmt(h.get('illiquid_pct'))}</td>
              <td style='text-align:center'>{_fmt(h.get('sharpe'), '{:.2f}').replace('%','')}</td>
              <td style='text-align:center'>{cost_cell}</td>
              <td style='text-align:center'><span class='badge {badge_cls}'>{badge_txt}</span></td>
            </tr>"""

        # Totals row
        rows += f"""<tr class='total-row'>
          <td colspan='4'>📊 סיכום משוקלל</td>
          <td style='text-align:left'>{_ils(total_w)}</td>
          <td style='text-align:center'>100%</td>
          <td style='text-align:center'>{_fmt(totals.get('equity'))}</td>
          <td style='text-align:center'>{_fmt(totals.get('foreign'))}</td>
          <td style='text-align:center'>{_fmt(totals.get('fx'))}</td>
          <td style='text-align:center'>{_fmt(totals.get('illiquid'))}</td>
          <td style='text-align:center'>—</td>
          <td style='text-align:center'>{_fmt(totals.get('cost'), '{:.2f}%') if not _nan_val(totals.get('cost')) else '—'}</td>
          <td></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ניתוח פורטפוליו — {client_name or 'לקוח'}</title>
  {_HTML_STYLE}
</head>
<body>
  <div class="report-header">
    <h1>📊 ניתוח פורטפוליו נוכחי</h1>
    {client_hdr}
    <p>הופק: {_now()} | כל הנתונים נכונים לתאריך הפקת הדוח</p>
  </div>
  {kpi_html}
  <div class="section">
    <h2>פירוט החזקות</h2>
    <div style="overflow-x:auto">
    <table>
      <thead><tr>
        <th>גוף</th><th>מוצר</th><th>מסלול</th><th>סוג</th><th>סכום</th><th>משקל</th>
        <th>מניות</th><th>חו"ל</th><th>מט"ח</th><th>לא סחיר</th><th>שארפ</th>
        <th>דמי ניהול</th><th>מקור</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    </div>
  </div>
  <div class="footer">
    נוצר על ידי Profit Mix Optimizer · {_now()}<br>
    <i>מסמך זה מיועד לשימוש פנימי בלבד ואינו מהווה ייעוץ השקעות</i>
  </div>
</body>
</html>"""

    return html.encode("utf-8")


def _nan_val(v):
    try:
        return math.isnan(float(v))
    except Exception:
        return v is None


# ── Jupyter Notebook ──────────────────────────────────────────────────────────

def build_notebook(df: pd.DataFrame, client_name: str = "", totals: dict = None) -> bytes:
    """
    Generate a Jupyter notebook (.ipynb) with:
      - Cell 0: Setup & data (the portfolio DataFrame as inline JSON)
      - Cell 1: KPI display
      - Cells 2–9: One chart per analysis layer
      - Cell 10: Markdown narrative for the presentation
    """
    if totals is None:
        from client_portfolio.charts import compute_totals
        totals = compute_totals(df) if not df.empty else {}

    # Serialise portfolio to JSON for embedding
    df_export = df.copy() if not df.empty else pd.DataFrame()
    for col in df_export.columns:
        if df_export[col].dtype == object:
            df_export[col] = df_export[col].astype(str)
    portfolio_json = df_export.to_json(orient="records", force_ascii=False)

    name_line = f"CLIENT_NAME = '{client_name}'" if client_name else "CLIENT_NAME = 'לקוח'"

    setup_code = f"""\
# ════════════════════════════════════════════════════════
# ניתוח פורטפוליו — {client_name or 'לקוח'}
# נוצר: {_now()}
# ════════════════════════════════════════════════════════

import json
import math
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from IPython.display import display, HTML

{name_line}
REPORT_DATE = '{_now()}'

# ── נתוני הפורטפוליו ──────────────────────────────────────
_PORTFOLIO_JSON = {repr(portfolio_json)}
df = pd.DataFrame(json.loads(_PORTFOLIO_JSON))
for col in ['amount','equity_pct','foreign_pct','fx_pct','illiquid_pct','sharpe']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

active = df[df.get('excluded', pd.Series([False]*len(df))).astype(bool) == False] if not df.empty else df

# ── Helper functions ──────────────────────────────────────
_PAL = ['#3A7AFE','#10B981','#F59E0B','#EF4444','#8B5CF6','#06B6D4','#EC4899','#F97316']
def _fmt(v, f='{{:.1f}}%'):
    try: return f.format(float(v)) if not math.isnan(float(v)) else '—'
    except: return '—'
def _ils(v):
    try:
        v=float(v)
        return f'₪{{v/1_000_000:.2f}}M' if v>=1_000_000 else f'₪{{v:,.0f}}'
    except: return '—'
def _wsum(col, df=active):
    sub = df[df[col].notna()].copy() if col in df.columns else pd.DataFrame()
    if sub.empty: return float('nan')
    t = sub['amount'].sum(); total = df['amount'].sum()
    return float((sub[col]*sub['amount']).sum()/t*(t/total)) if t>0 else float('nan')

print('✅ פורטפוליו טעון:', len(df), 'מוצרים | סה"כ:', _ils(df['amount'].sum()))
"""

    kpi_code = f"""\
# ── KPI Summary ───────────────────────────────────────────
total = active['amount'].sum()
kpis = {{
    'סך נכסים': _ils(total),
    'מוצרים': str(len(active)),
    'מנהלים': str(active['provider'].nunique()),
    'מניות': _fmt(_wsum('equity_pct')),
    'חו\\"ל': _fmt(_wsum('foreign_pct')),
    'מט\\"ח': _fmt(_wsum('fx_pct')),
    'לא סחיר': _fmt(_wsum('illiquid_pct')),
}}

html_kpis = '''<div style="display:flex;gap:12px;flex-wrap:wrap;direction:rtl">'''
for label, val in kpis.items():
    html_kpis += f'''<div style="background:#EFF6FF;border-radius:10px;padding:14px 20px;text-align:center;min-width:100px">
        <div style="font-size:22px;font-weight:900;color:#1F3A5F">{{val}}</div>
        <div style="font-size:11px;color:#64748B">{{label}}</div></div>'''
html_kpis += '</div>'
display(HTML(f'<h2 style="direction:rtl;font-family:sans-serif">📊 סיכום פורטפוליו — {{CLIENT_NAME}}</h2>' + html_kpis))
"""

    chart_codes = [
        # Chart 2: by manager
        """\
# ── 2. פיזור בין מנהלים ──────────────────────────────────
grp = active.groupby('provider')['amount'].sum().reset_index()
grp = grp[grp['amount']>0].sort_values('amount',ascending=False)
fig = go.Figure(go.Pie(labels=grp['provider'],values=grp['amount'],hole=0.45,
    marker=dict(colors=_PAL[:len(grp)],line=dict(color='#fff',width=2)),
    textinfo='label+percent',
    hovertemplate='<b>%{label}</b><br>%{value:,.0f} ₪<br>%{percent}<extra></extra>'))
fig.update_layout(title='2. פיזור בין מנהלים',height=380,
    font=dict(family='sans-serif'),paper_bgcolor='rgba(0,0,0,0)')
fig.add_annotation(text=f'<b>{_ils(grp["amount"].sum())}</b>',x=0.5,y=0.5,
    font=dict(size=16,color='#1F3A5F'),showarrow=False)
fig.show()
""",
        # Chart 3: stocks vs bonds
        """\
# ── 3. מניות vs אגח ──────────────────────────────────────
eq_w = _wsum('equity_pct'); ill_w = _wsum('illiquid_pct'); fx_w = _wsum('fx_pct')
bonds = max(0, 100 - (eq_w or 0) - (ill_w or 0) - (fx_w or 0))
fig = go.Figure()
for name, val, col in [('מניות', eq_w or 0, '#3A7AFE'), ('אגח/אחר', bonds, '#10B981')]:
    fig.add_trace(go.Bar(x=['תמהיל'],y=[val],name=f'{name} ({val:.1f}%)',
        marker_color=col,text=[f'{val:.1f}%'],textposition='inside'))
fig.update_layout(title='3. מניות vs אגח',barmode='stack',height=320,
    yaxis=dict(ticksuffix='%',range=[0,105]),font=dict(family='sans-serif'))
fig.show()
""",
        # Chart 4: foreign vs domestic
        """\
# ── 4. חול vs ישראל ──────────────────────────────────────
fo = _wsum('foreign_pct') or 0
fig = go.Figure(go.Pie(labels=['חו\\"ל','ישראל'],values=[fo,max(0,100-fo)],hole=0.5,
    marker=dict(colors=['#3A7AFE','#10B981'],line=dict(color='#fff',width=2)),
    textinfo='label+percent'))
fig.update_layout(title='4. חול vs ישראל',height=340,font=dict(family='sans-serif'))
fig.show()
""",
        # Chart 5: FX vs ILS
        """\
# ── 5. מטח vs שקל ────────────────────────────────────────
fx = _wsum('fx_pct') or 0
fig = go.Figure(go.Pie(labels=['מט\\"ח','שקל'],values=[fx,max(0,100-fx)],hole=0.5,
    marker=dict(colors=['#F59E0B','#64748B'],line=dict(color='#fff',width=2)),
    textinfo='label+percent'))
fig.update_layout(title='5. מטח vs שקל',height=340,font=dict(family='sans-serif'))
fig.show()
""",
        # Chart 6: asset breakdown
        """\
# ── 6. פיזור סוגי נכסים לפי מוצר ────────────────────────
labels = (active['provider'] + ' | ' + active.get('product_name', active['provider'])).tolist()
fig = go.Figure()
for col, name, color in [('equity_pct','מניות','#3A7AFE'),('foreign_pct','חול','#10B981'),
                           ('fx_pct','מטח','#F59E0B'),('illiquid_pct','לא סחיר','#EF4444')]:
    vals = active[col].fillna(0).tolist() if col in active.columns else [0]*len(active)
    fig.add_trace(go.Bar(name=name,x=labels,y=vals,marker_color=color,opacity=0.88))
fig.update_layout(title='6. פיזור סוגי נכסים לפי מוצר',barmode='group',height=400,
    xaxis=dict(tickangle=-30),yaxis=dict(ticksuffix='%'),font=dict(family='sans-serif'))
fig.show()
""",
        # Chart 7: annuity vs capital
        """\
# ── 7. קצבה vs הון ───────────────────────────────────────
if 'product_type' in active.columns:
    annuity_types = {'קרנות פנסיה'}
    capital_types = {'קרנות השתלמות','פוליסות חיסכון','קופות גמל','גמל להשקעה'}
    ann = active[active['product_type'].isin(annuity_types)]['amount'].sum()
    cap = active[active['product_type'].isin(capital_types)]['amount'].sum()
    oth = active['amount'].sum() - ann - cap
    labels = ['קצבה (פנסיה)','הון (קרנות/גמל)']
    vals   = [ann, cap]
    if oth > 0: labels.append('אחר'); vals.append(oth)
    fig = go.Figure(go.Pie(labels=labels,values=vals,hole=0.5,
        marker=dict(colors=['#8B5CF6','#06B6D4','#64748B'],line=dict(color='#fff',width=2)),
        textinfo='label+percent+value',
        hovertemplate='<b>%{label}</b><br>%{value:,.0f} ₪<br>%{percent}<extra></extra>'))
    fig.update_layout(title='7. קצבה vs הון',height=360,font=dict(family='sans-serif'))
    fig.show()
""",
        # Chart 8: costs
        """\
# ── 8. עלויות ────────────────────────────────────────────
if 'annual_cost_pct' in active.columns and active['annual_cost_pct'].notna().any():
    sub = active[active['annual_cost_pct'].notna()].copy()
    sub['cost_ils'] = sub['amount'] * sub['annual_cost_pct'] / 100
    wc = sub['cost_ils'].sum() / sub['amount'].sum() * 100
    labels = (sub['provider'] + ' | ' + sub.get('product_name', sub['provider'])).tolist()
    fig = make_subplots(rows=1,cols=2,subplot_titles=('עלות שנתית (₪)','דמי ניהול %'))
    fig.add_trace(go.Bar(x=labels,y=sub['cost_ils'].tolist(),marker_color='#EF4444',
        text=sub['cost_ils'].map(lambda v:f'₪{v:,.0f}'),textposition='outside'),row=1,col=1)
    fig.add_trace(go.Bar(x=labels,y=sub['annual_cost_pct'].tolist(),marker_color='#F59E0B',
        text=sub['annual_cost_pct'].map(lambda v:f'{v:.2f}%'),textposition='outside'),row=1,col=2)
    fig.update_layout(title=f'8. עלויות — עלות משוקללת {wc:.2f}%',height=400,showlegend=False)
    fig.show()
else:
    print('לא הוזנו דמי ניהול. הזן עמודה annual_cost_pct בנתונים.')
""",
        # Chart 9: radar
        """\
# ── 9. מפת נכסים (Radar) ─────────────────────────────────
cats = ['מניות','חול','מטח','לא סחיר']
vals = [_wsum('equity_pct') or 0, _wsum('foreign_pct') or 0,
        _wsum('fx_pct') or 0, _wsum('illiquid_pct') or 0]
cats_c = cats + [cats[0]]; vals_c = vals + [vals[0]]
fig = go.Figure(go.Scatterpolar(r=vals_c,theta=cats_c,fill='toself',
    fillcolor='rgba(58,122,254,0.15)',line=dict(color='#3A7AFE',width=2.5),
    marker=dict(size=7,color='#3A7AFE')))
fig.update_layout(title='9. מפת נכסים',polar=dict(
    radialaxis=dict(visible=True,range=[0,100],ticksuffix='%')),
    showlegend=False,height=380,font=dict(family='sans-serif'))
fig.show()
""",
    ]

    # Narrative markdown cell
    narrative = f"""\
## 📋 סיכום ניתוח פורטפוליו — {client_name or 'לקוח'}

**תאריך:** {_now()}

### עיקרי הממצאים

| פרמטר | ערך |
|--------|-----|
| סך נכסים | {_ils(totals.get('total', 0))} |
| מספר מוצרים | {totals.get('n_products', 0)} |
| מספר מנהלים | {totals.get('n_managers', 0)} |
| חשיפת מניות (משוקלל) | {_fmt(totals.get('equity'))} |
| חשיפת חו\"ל (משוקלל) | {_fmt(totals.get('foreign'))} |
| חשיפת מט\"ח (משוקלל) | {_fmt(totals.get('fx'))} |
| חשיפה לא-סחיר (משוקלל) | {_fmt(totals.get('illiquid'))} |

### נקודות לדיון עם הלקוח

1. **פיזור בין מנהלים** — האם יש ריכוז גבוה מדי במנהל אחד?
2. **חשיפת מניות** — האם מתאימה לפרופיל הסיכון של הלקוח?
3. **חשיפת חו\"ל** — האם מאוזנת ביחס לחשיפת מטבע?
4. **עלויות** — האם דמי הניהול מוצדקים לאור ביצועי המוצרים?
5. **קצבה vs הון** — האם האיזון בין מוצרי קצבה להון מתאים לצרכי הלקוח?

---
*מסמך זה נוצר אוטומטית ומיועד לשימוש פנימי בלבד. אינו מהווה ייעוץ השקעות.*
"""

    # Build notebook structure
    def _code_cell(source: str) -> dict:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source,
        }

    def _md_cell(source: str) -> dict:
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source,
        }

    cells = (
        [_code_cell(setup_code), _code_cell(kpi_code)]
        + [_code_cell(c) for c in chart_codes]
        + [_md_cell(narrative)]
    )

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }

    return json.dumps(nb, ensure_ascii=False, indent=2).encode("utf-8")



# ── NotebookLM Package ────────────────────────────────────────────────────────

# The full presentation prompt — embedded directly so every exported package
# contains complete instructions for NotebookLM / Claude / Gemini.
_PRESENTATION_PROMPT = """
## פרומפט לבניית מצגת השקעות — Family Office / Private Wealth

### משימה
בנה מצגת השקעות מקצועית, בעברית מלאה, המעוצבת ברמת Family Office / Private Wealth /
Global Investment Review עבור תיק ההשקעות שנתוניו מופיעים במסמך זה.

### עקרונות על
1. אסור להשמיט נתונים — כל נתון חייב למצוא ביטוי ברור במצגת.
2. אסור לעגל יתר על המידה — להציג גם סכום וגם אחוז.
3. המצגת אינה המלצה — זו מצגת ניתוח, מיפוי ושיקוף מבנה תיק בלבד.
4. סגנון: Family Office בינלאומי — UBS / Morgan Stanley Wealth Management style.

### עיצוב נדרש
- פלטת צבעים: כחול כהה / אפור / לבן / זהב עדין
- פונט: אלגנטי, קריא, מודרני
- שוליים רחבים, הרבה white space
- גרפים מאוזנים עם כותרות מדויקות, יחידות מידה ותוויות קריאות
- אחידות מלאה בין שקפים
- ללא 3D, ללא הצללות כבדות, ללא צבעוניות זולה

### מבנה 14+ שקפים

**שקף 1 — שער:** שם הלקוח, תאריך, כותרת "סקירת מבנה תיק ההשקעות"

**שקף 2 — Executive Summary:** 4-6 KPI cards (סך נכסים, מנהלים, מוצרים, מניות, אגח, חול, מטח, עלות)
+ טקסט 3-5 שורות תיאורי, ללא המלצות.

**שקף 3 — סך הסכום ומבנה מקורות:** מספר גדול במרכז + donut/stacked bar לפי קבוצות מוצר.

**שקף 4 — חלוקה בין מנהלים:** horizontal bar מדורג, סכום + אחוז לכל מנהל.

**שקף 5 — מניות vs אגח:** stacked bar + donut. לציין שהנתון משוקלל.

**שקף 6 — חול vs ישראל:** stacked bar / 100% stacked. להבחין מגיאוגרפיה ≠ מטבע.

**שקף 7 — מטח vs שקל:** שקף נפרד מחול/ישראל. לציין מטבעות עיקריים אם ידוע.

**שקף 8 — פיזור סוגי נכסים:** treemap / stacked bar מפורט. רמת פירוט גבוהה — מוסדי.

**שקף 9 — קצבה vs הון:** two-level donut. קצבה (פנסיה) vs הון (קרנות, גמל, פוליסות).

**שקף 10 — עלויות:** bar chart עלות לפי מוצר + bubble עלות מול גודל + שורת עלות משוקללת.
אם חסרים נתוני עלות — להציג כמה % מהתיק כוסה / לא כוסה.

**שקף 11 — חתכים משולבים:** heatmap / matrix — מנהל × סוג נכס, מוצר × מטבע, מוצר × עלות.

**שקף 12 — סיכום מבנה התיק:** 5-7 משפטים רשמיים, בלי המלצות, רק שיקוף.

**שקף 13 — הנחות ומתודולוגיה:** נתונים שהוזנו / חושבו / חסרים, הנחות סיווג, רמת כיסוי.

**שקף 14+ — נספח מלא:** כל הטבלאות, כל הסכומים, כל הפירוקים — שום נתון לא ילך לאיבוד.

### כללים לגרפים
- מניות = צבע קבוע עקבי בין שקפים
- אגח = צבע קבוע
- ישראל = צבע קבוע
- חול = צבע קבוע
- שקל / מטח = צבעים נפרדים
- מספרים: גם על הגרף וגם בטבלה
- ניסוח: "נתון משוקלל על בסיס סך הנכסים שהוזנו"

### טון נדרש
מקצועי · שקט · בטוח · מדויק · לא שיווקי · לא עמום.
דוגמה: "התרשים מציג את חלוקת הנכסים המשוקללת על פני כלל התיק."
"""


def build_notebooklm_package(
    df: pd.DataFrame,
    client_name: str = "",
    totals: dict = None,
) -> bytes:
    """
    Build a comprehensive Markdown file for NotebookLM / Claude / Gemini.

    Contains:
      1. KPI summary
      2. Full holdings table
      3. Manager concentration
      4. Product-type breakdown
      5. Raw JSON data
      6. Full presentation prompt (embedded)
      7. Ready-made questions for NotebookLM
    """
    if totals is None:
        from client_portfolio.charts import compute_totals
        totals = compute_totals(df) if not df.empty else {}

    active = df[~df.get("excluded", pd.Series([False]*len(df))).astype(bool)] if not df.empty else df
    now = _now()
    name_hdr = f"**לקוח:** {client_name}" if client_name else ""

    # ── Section 1: KPI ────────────────────────────────────────────────────
    # Fix nan → "לא הוזנה" in JSON output
    cost_val = totals.get('cost')
    cost_str = f"{float(cost_val):.3f}" if cost_val and not _nan_val(cost_val) else "null"

    kpi_md = f"""# ניתוח תיק השקעות — {client_name or 'לקוח'}
{name_hdr}
**תאריך הפקה:** {now}

---

## 1. סיכום תיק כולל (KPI)

| מדד | ערך |
|-----|-----|
| **סך נכסים** | {_ils(totals.get('total', 0))} |
| **סך נכסים (מספר)** | {totals.get('total', 0):.0f} ₪ |
| **מספר מוצרים** | {totals.get('n_products', 0)} |
| **מספר מנהלים** | {totals.get('n_managers', 0)} |
| **חשיפת מניות (משוקלל)** | {_fmt(totals.get('equity'))} |
| **חשיפת חו"ל (משוקלל)** | {_fmt(totals.get('foreign'))} |
| **חשיפת מט"ח (משוקלל)** | {_fmt(totals.get('fx'))} |
| **חשיפה לא-סחיר (משוקלל)** | {_fmt(totals.get('illiquid'))} |
| **עלות שנתית משוקללת** | {_fmt(totals.get('cost'), '{:.2f}%') if not _nan_val(totals.get('cost')) else 'לא הוזנה'} |

> כל הנתונים המשוקללים מחושבים על בסיס סך הנכסים שהוזנו ({_ils(totals.get('total', 0))}).

"""

    # ── Section 2: Holdings ───────────────────────────────────────────────
    holdings_md = "## 2. פירוט מלא של החזקות\n\n"
    if not active.empty:
        total_amt = active["amount"].sum()
        holdings_md += (
            "| גוף | מוצר | מסלול | סוג מוצר | סכום (₪) | משקל | מניות | חו\"ל | מט\"ח | "
            "לא סחיר | שארפ | דמי ניהול | מקור נתון |\n"
        )
        holdings_md += "|-----|------|-------|----------|---------|------|-------|------|------|---------|------|------------|----------|\n"
        for _, h in active.iterrows():
            w = h["amount"] / total_amt * 100 if total_amt > 0 else 0
            cost_cell = _fmt(h.get("annual_cost_pct"), "{:.2f}%") if "annual_cost_pct" in h.index and not _nan_val(h.get("annual_cost_pct")) else "—"
            holdings_md += (
                f"| {h.get('provider','')} | {h.get('product_name','')} | "
                f"{h.get('track','')} | {h.get('product_type','')} | "
                f"{h['amount']:,.0f} | {w:.1f}% | "
                f"{_fmt(h.get('equity_pct'))} | {_fmt(h.get('foreign_pct'))} | "
                f"{_fmt(h.get('fx_pct'))} | {_fmt(h.get('illiquid_pct'))} | "
                f"{_fmt(h.get('sharpe'), '{:.2f}').replace('%','')} | "
                f"{cost_cell} | {h.get('allocation_source','?')} |\n"
            )
        # Totals
        holdings_md += (
            f"| **סיכום** | | | | **{total_amt:,.0f}** | **100%** | "
            f"**{_fmt(totals.get('equity'))}** | **{_fmt(totals.get('foreign'))}** | "
            f"**{_fmt(totals.get('fx'))}** | **{_fmt(totals.get('illiquid'))}** | | | |\n\n"
        )
        # Missing allocation note
        if not active.empty:
            n_miss = int((active.get("allocation_source", pd.Series()) == "missing").sum()) if "allocation_source" in active.columns else 0
            if n_miss > 0:
                holdings_md += f"> ⚠️ {n_miss} מוצרים חסרי נתוני אלוקציה — לא נכללו בחישוב המשוקלל.\n\n"

    # ── Section 3: Manager concentration ─────────────────────────────────
    mgr_md = "## 3. ריכוז לפי מנהל\n\n"
    if not active.empty:
        total_amt = active["amount"].sum()
        mgr_grp = active.groupby("provider")["amount"].sum().reset_index()
        mgr_grp["weight"] = mgr_grp["amount"] / total_amt * 100
        mgr_grp = mgr_grp.sort_values("weight", ascending=False)
        top3_pct = mgr_grp.head(3)["weight"].sum()
        mgr_md += f"**3 מנהלים גדולים = {top3_pct:.1f}% מהתיק**\n\n"
        mgr_md += "| דירוג | מנהל | סכום (₪) | משקל | הערה |\n|-------|------|---------|------|------|\n"
        for rank, (_, row) in enumerate(mgr_grp.iterrows(), 1):
            flag = " ⚠️ ריכוז >30%" if row["weight"] > 30 else ""
            mgr_md += f"| {rank} | {row['provider']} | {row['amount']:,.0f} | {row['weight']:.1f}%{flag} | |\n"
        mgr_md += "\n"

    # ── Section 4: Product type ───────────────────────────────────────────
    _ANNUITY = {"קרנות פנסיה"}
    _CAPITAL  = {"קרנות השתלמות", "פוליסות חיסכון", "קופות גמל", "גמל להשקעה"}
    prod_md = "## 4. פיזור בין סוגי מוצרים (קצבה vs הון)\n\n"
    if not active.empty and "product_type" in active.columns:
        total_amt = active["amount"].sum()
        prod_grp = active.groupby("product_type")["amount"].sum().reset_index()
        prod_grp["weight"] = prod_grp["amount"] / total_amt * 100
        prod_grp["category"] = prod_grp["product_type"].apply(
            lambda t: "קצבה" if t in _ANNUITY else "הון" if t in _CAPITAL else "אחר"
        )
        # Subtotals by category
        cat_sum = prod_grp.groupby("category")["amount"].sum().reset_index()
        cat_sum["weight"] = cat_sum["amount"] / total_amt * 100
        for _, row in cat_sum.iterrows():
            prod_md += f"**{row['category']}:** {row['amount']:,.0f} ₪ ({row['weight']:.1f}%)\n"
        prod_md += "\n| סוג מוצר | קטגוריה | סכום (₪) | משקל |\n|---------|---------|---------|------|\n"
        for _, row in prod_grp.sort_values("amount", ascending=False).iterrows():
            prod_md += f"| {row['product_type']} | {row['category']} | {row['amount']:,.0f} | {row['weight']:.1f}% |\n"
        prod_md += "\n"

    # ── Section 5: Methodology ────────────────────────────────────────────
    n_complete = int(active[active["allocation_source"].isin(["auto_filled","imported","manual"])].shape[0]) if not active.empty and "allocation_source" in active.columns else 0
    n_missing  = len(active) - n_complete if not active.empty else 0
    total_amt  = active["amount"].sum() if not active.empty else 0
    covered_pct = n_complete / len(active) * 100 if not active.empty and len(active) > 0 else 0

    method_md = f"""## 5. מתודולוגיה ומידע חסר

| פרמטר | ערך |
|--------|-----|
| סה"כ מוצרים שהוזנו | {len(active)} |
| מוצרים עם נתוני אלוקציה | {n_complete} ({covered_pct:.0f}% מהתיק לפי סכום) |
| מוצרים ללא נתוני אלוקציה | {n_missing} |
| בסיס חישוב משוקלל | סכום הנכסים שהוזנו ({_ils(total_amt)}) |
| נתוני עלות | {'חלקיים' if not _nan_val(totals.get('cost')) else 'לא הוזנו'} |

**הנחות סיווג:**
- "קצבה" = קרנות פנסיה בלבד
- "הון" = קרנות השתלמות, פוליסות חיסכון, קופות גמל, גמל להשקעה
- "לא-סחיר" = חשיפה לנכסים לא סחירים (PE, נדלן, תשתיות)
- נתוני חו"ל מייצגים חשיפה גיאוגרפית, לא בהכרח חשיפת מטבע
- נתוני מטח מייצגים חשיפה מטבעית, שאינה זהה לחשיפת חול

"""

    # ── Section 6: Raw JSON ───────────────────────────────────────────────
    json_md = f"""## 6. נתוני גלם (JSON)

```json
{{
  "portfolio_summary": {{
    "client_name": "{client_name or ''}",
    "report_date": "{now}",
    "total_amount_ils": {totals.get('total', 0):.0f},
    "n_products": {totals.get('n_products', 0)},
    "n_managers": {totals.get('n_managers', 0)},
    "equity_pct_weighted": {totals.get('equity', 0) or 0:.2f},
    "foreign_pct_weighted": {totals.get('foreign', 0) or 0:.2f},
    "fx_pct_weighted": {totals.get('fx', 0) or 0:.2f},
    "illiquid_pct_weighted": {totals.get('illiquid', 0) or 0:.2f},
    "annual_cost_pct_weighted": {cost_str},
    "data_coverage_pct": {covered_pct:.1f}
  }}
}}
```

"""

    # ── Section 7: Full presentation prompt ──────────────────────────────
    prompt_md = f"""---

## 7. פרומפט לבניית מצגת

{_PRESENTATION_PROMPT}

---

## 8. שאלות מומלצות לשאול את NotebookLM

השאלות הבאות נועדו לשימוש **לאחר** העלאת קובץ זה כמקור ב-NotebookLM:

1. "סכם את מצב תיק ההשקעות של **{client_name or 'הלקוח'}** ב-5 משפטים מקצועיים"
2. "מהם הסיכונים העיקריים בתיק זה לפי הנתונים שהוזנו?"
3. "מהי רמת הריכוז בין מנהלים — האם יש חשיפה יתרה?"
4. "כתוב את תוכן שקף Executive Summary (שקף 2) למצגת ללקוח"
5. "כתוב את תוכן שקף ניתוח סיכון (שקף 11) למצגת ללקוח"
6. "האם החשיפה למניות ({_fmt(totals.get('equity'))}) מתאימה לפרופיל לקוח ממוצע?"
7. "כתוב שקף סיכום מבנה התיק בסגנון Family Office (שקף 12)"
8. "מה ההמלצות הטופ-3 לשיפור הפיזור בתיק, על בסיס הנתונים בלבד?"
9. "כתוב הערות מתודולוגיה ומידע חסר (שקף 13) לפי הנתונים שהוזנו"
10. "בנה את כל 14 השקפים של המצגת בסגנון wealth management לפי הפרומפט והנתונים"

---
*מסמך זה נוצר אוטומטית על ידי Profit Mix Optimizer ומיועד לשימוש פנימי בלבד. אינו מהווה ייעוץ השקעות.*
"""

    full = kpi_md + holdings_md + mgr_md + prod_md + method_md + json_md + prompt_md
    return full.encode("utf-8")
