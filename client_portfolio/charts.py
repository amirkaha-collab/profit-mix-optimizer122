# -*- coding: utf-8 -*-
"""
client_portfolio/charts.py  — fixed
"""
from __future__ import annotations
import math
from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_NAVY  = "#1F3A5F"; _BLUE  = "#3A7AFE"; _GREEN = "#10B981"
_AMBER = "#F59E0B"; _RED   = "#EF4444"; _PURP  = "#8B5CF6"
_TEAL  = "#06B6D4"; _PINK  = "#EC4899"; _SLATE = "#64748B"; _INDIGO= "#6366F1"

_PAL = [_BLUE,_GREEN,_AMBER,_RED,_PURP,_TEAL,_PINK,"#F97316","#84CC16",_INDIGO,"#14B8A6","#FB7185"]
_FONT = dict(family="Segoe UI, -apple-system, sans-serif", color="#374151", size=12)
_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,249,252,1)", font=_FONT,
    legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5,
                font=dict(size=11), bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#E5E7EB", borderwidth=1),
)

# Default margins — charts that need different margins pass their own
_M_PIE  = dict(l=20, r=20, t=55, b=110)   # pie / donut
_M_BAR  = dict(l=30, r=30, t=55, b=120)   # stacked/grouped bar
_M_COST = dict(l=30, r=30, t=55, b=90)    # cost subplots

def _title(t): return dict(text=t, font=dict(size=14, color=_NAVY, family=_FONT["family"]), x=0.5)
def _fmt_ils(v):
    if _nan(v): return "—"
    if v>=1_000_000: return f"₪{v/1_000_000:.2f}M"
    if v>=1_000: return f"₪{v/1_000:.0f}K"
    return f"₪{v:.0f}"
def _nan(v):
    try: return math.isnan(float(v))
    except: return True
def _active(df):
    return df[~df.get("excluded", pd.Series([False]*len(df))).astype(bool)]
def _wsum(act, col):
    if col not in act.columns: return 0.0
    sub = act[act[col].notna()]
    if sub.empty: return 0.0
    t = sub["amount"].sum(); total = act["amount"].sum()
    return float((sub[col]*sub["amount"]).sum()/t*(t/total)) if t>0 else 0.0

def compute_totals(df):
    act = _active(df); total = float(act["amount"].sum())
    n_mgr = act["provider"].nunique() if "provider" in act.columns else 0
    def ws(col):
        sub = act[act[col].notna()] if col in act.columns else pd.DataFrame()
        if sub.empty: return float("nan")
        t = sub["amount"].sum()
        return float((sub[col]*sub["amount"]).sum()/t) if t>0 else float("nan")
    return {"total":total,"n_products":len(act),"n_managers":n_mgr,
            "equity":ws("equity_pct"),"foreign":ws("foreign_pct"),
            "fx":ws("fx_pct"),"illiquid":ws("illiquid_pct"),
            "cost":ws("annual_cost_pct") if "annual_cost_pct" in act.columns else float("nan")}

# 1. Manager donut
def chart_by_manager(df):
    act = _active(df)
    if "provider" not in act.columns or act.empty: return go.Figure()
    grp = act.groupby("provider")["amount"].sum().reset_index()
    grp = grp[grp["amount"]>0].sort_values("amount",ascending=False)
    fig = go.Figure(go.Pie(
        labels=grp["provider"], values=grp["amount"], hole=0.48,
        marker=dict(colors=_PAL[:len(grp)], line=dict(color="#fff",width=2)),
        textinfo="label+percent", textfont=dict(size=11), insidetextorientation="auto",
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} ₪<br>%{percent}<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{_fmt_ils(float(grp['amount'].sum()))}</b>",
                       x=0.5,y=0.5,font=dict(size=15,color=_NAVY),showarrow=False)
    fig.update_layout(**_BASE, title=_title("פיזור בין מנהלים"), height=420, margin=_M_PIE)
    return fig

# 2. Stocks / bonds / illiquid stacked → 100%
def chart_stocks_bonds(df):
    act = _active(df)
    eq  = _wsum(act, "equity_pct")
    ill = _wsum(act, "illiquid_pct")
    bonds = max(0.0, 100.0 - eq - ill)
    bars = [(f"מניות ({eq:.1f}%)", eq, _BLUE),
            (f'אג"ח / אחר ({bonds:.1f}%)', bonds, _GREEN),
            (f"לא-סחיר ({ill:.1f}%)", ill, _RED)]
    fig = go.Figure()
    for name, val, color in bars:
        fig.add_trace(go.Bar(x=["תמהיל"], y=[val], name=name, marker_color=color,
                             text=[f"{val:.1f}%"], textposition="inside",
                             textfont=dict(size=13, color="#fff"),
                             hovertemplate=f"<b>{name}</b>: %{{y:.1f}}%<extra></extra>"))
    fig.update_layout(**_BASE, title=_title('מניות · אג"ח · לא-סחיר'), barmode="stack",
                      height=380, margin=_M_BAR,
                      yaxis=dict(ticksuffix="%",range=[0,105],gridcolor="#E5E7EB",automargin=True),
                      xaxis=dict(automargin=True))
    return fig

# 3. Foreign vs Israel
def chart_foreign_domestic(df):
    act = _active(df); fo = _wsum(act,"foreign_pct"); dom = max(0.0,100-fo)
    fig = go.Figure(go.Pie(labels=['חו"ל',"ישראל"],values=[fo,dom],hole=0.5,
        marker=dict(colors=[_TEAL,_NAVY],line=dict(color="#fff",width=2)),
        textinfo="label+percent",textfont=dict(size=12),insidetextorientation="auto"))
    fig.update_layout(**_BASE,title=_title('חו"ל vs ישראל'),height=400,margin=_M_PIE)
    return fig

# 4. FX vs ILS
def chart_fx_ils(df):
    act = _active(df); fx = _wsum(act,"fx_pct"); ils = max(0.0,100-fx)
    fig = go.Figure(go.Pie(labels=['מט"ח',"שקל"],values=[fx,ils],hole=0.5,
        marker=dict(colors=[_AMBER,_SLATE],line=dict(color="#fff",width=2)),
        textinfo="label+percent",textfont=dict(size=12),insidetextorientation="auto"))
    fig.update_layout(**_BASE,title=_title('מט"ח vs שקל'),height=400,margin=_M_PIE)
    return fig

# 5. Asset breakdown — DONUT by product-type (not per company)
_PT_COLORS = {
    "קרנות השתלמות":_BLUE,"קופות גמל":_GREEN,"גמל להשקעה":_TEAL,
    "קרנות פנסיה":_PURP,"פוליסות חיסכון":_AMBER,"ביטוח מנהלים":_PINK,
    "תיק מנוהל":_INDIGO,"אחר":_SLATE,
}
def chart_asset_breakdown(df):
    act = _active(df)
    if act.empty or "product_type" not in act.columns: return go.Figure()
    grp = act.groupby("product_type")["amount"].sum().reset_index()
    grp = grp[grp["amount"]>0].sort_values("amount",ascending=False)
    if grp.empty: return go.Figure()
    colors = [_PT_COLORS.get(pt,_SLATE) for pt in grp["product_type"]]
    total  = float(grp["amount"].sum())
    fig = go.Figure(go.Pie(labels=grp["product_type"],values=grp["amount"],hole=0.48,
        marker=dict(colors=colors,line=dict(color="#fff",width=2)),
        textinfo="label+percent",textfont=dict(size=11),insidetextorientation="auto",
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} ₪<br>%{percent}<extra></extra>"))
    fig.add_annotation(text=f"<b>{_fmt_ils(total)}</b>",x=0.5,y=0.5,
                       font=dict(size=15,color=_NAVY),showarrow=False)
    fig.update_layout(**_BASE,title=_title("פיזור לפי סוג מוצר"),
                      height=440,margin=_M_PIE)
    return fig

# 6. Annuity vs Capital (auto from product_type or manual override)
_ANNUITY_TYPES = {"קרנות פנסיה","ביטוח מנהלים"}
_CAPITAL_TYPES = {"קרנות השתלמות","פוליסות חיסכון","קופות גמל","גמל להשקעה"}

def chart_annuity_capital(df, manual_annuity_pct=None):
    act = _active(df)
    if act.empty: return go.Figure()
    total = float(act["amount"].sum())
    if total <= 0: return go.Figure()

    if manual_annuity_pct is not None:
        ap = float(manual_annuity_pct)
        annuity, capital, other = total*ap/100, total*(100-ap)/100, 0.0
    elif "product_type" in act.columns:
        annuity = float(act[act["product_type"].isin(_ANNUITY_TYPES)]["amount"].sum())
        capital = float(act[act["product_type"].isin(_CAPITAL_TYPES)]["amount"].sum())
        other   = total - annuity - capital
    else:
        return go.Figure()

    labels = ["קצבה","הון"]; values = [annuity,capital]; colors = [_PURP,_TEAL]
    if other > 0.5:
        labels.append("אחר"); values.append(other); colors.append(_SLATE)
    sub = " (ידני)" if manual_annuity_pct is not None else ""
    fig = go.Figure(go.Pie(labels=labels,values=values,hole=0.5,
        marker=dict(colors=colors,line=dict(color="#fff",width=2)),
        textinfo="label+percent",textfont=dict(size=12),insidetextorientation="auto",
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} ₪<br>%{percent}<extra></extra>"))
    fig.update_layout(**_BASE,title=_title(f"קצבה vs הון{sub}"),
                      height=420,margin=_M_PIE)
    return fig

# 7. Costs
def chart_costs(df):
    if "annual_cost_pct" not in df.columns: return go.Figure()
    act = _active(df)
    sub = act[act["annual_cost_pct"].notna()].copy()
    if sub.empty: return go.Figure()
    sub["cost_ils"] = sub["amount"]*sub["annual_cost_pct"]/100
    sub["lbl"] = sub["provider"].str[:18]
    sub = sub.sort_values("cost_ils",ascending=False)
    wc = sub["cost_ils"].sum()/sub["amount"].sum()*100 if sub["amount"].sum()>0 else 0
    n = len(sub); bottom = max(90, 55+n*10)
    fig = make_subplots(rows=1,cols=2,subplot_titles=("עלות שנתית (₪)","דמי ניהול (%)"),
                        horizontal_spacing=0.14)
    fig.add_trace(go.Bar(x=sub["lbl"],y=sub["cost_ils"],marker_color=_RED,name="עלות ₪",
        text=sub["cost_ils"].map(lambda v:f"₪{v:,.0f}"),textposition="outside",
        hovertemplate="<b>%{x}</b><br>₪%{y:,.0f}<extra></extra>"),row=1,col=1)
    fig.add_trace(go.Bar(x=sub["lbl"],y=sub["annual_cost_pct"],marker_color=_AMBER,name="דמי ניהול %",
        text=sub["annual_cost_pct"].map(lambda v:f"{v:.2f}%"),textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>"),row=1,col=2)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(248,249,252,1)",font=_FONT,
        title=_title(f'עלויות — משוקלל: {wc:.2f}% | שנתי: {_fmt_ils(sub["cost_ils"].sum())}'),
        height=max(360,200+n*18),margin=dict(l=30,r=30,t=55,b=bottom),showlegend=False)
    fig.update_xaxes(tickangle=-35,tickfont=dict(size=10),automargin=True)
    fig.update_yaxes(gridcolor="#E5E7EB",automargin=True)
    return fig

# Removed — kept as stubs for backward compatibility
def chart_concentration(df): return go.Figure()
def chart_sharpe_comparison(df): return go.Figure()
def chart_radar(df): return go.Figure()
