"""Visual Phase 2 showcase -- the autonomous VectorSentry Live Sentry.

The page tells one story, top to bottom:
  1. Controls: break the live database, choose how tight the budget is.
  2. What state is the system in, in plain English?
  3. Search health over time: the incident and the recovery.
  4. How VectorSentry chose: which documents it fixes, how, and why that
     beats fixing them first-come-first-served.
  5. The safety check, batch by batch.
  6. Incident report, embedding map and raw log for technical viewers.
"""

from __future__ import annotations

import re
import sys
import threading
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in ("common/src", "phase1_baseline/src", "phase2_proposed/src"):
    sys.path.insert(0, str(_REPO_ROOT / _src))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from common.config import load_config  # noqa: E402
from common.seed import set_seed  # noqa: E402
from phase2.live_sentry import (  # noqa: E402
    STATUS_ALERT,
    STATUS_NEEDS_REVIEW,
    STATUS_REMEDIATING,
    STATUS_RESOLVED,
    LiveSentryState,
    build_context,
    monitor_loop,
)
from phase2.pipeline import plan_remediation  # noqa: E402

CONFIG_PATH = "phase2_proposed/configs/default.yaml"
UI_REFRESH_SECONDS = 1.5

st.set_page_config(page_title="VectorSentry Live Sentry", page_icon="🛡️", layout="wide")

# --------------------------------------------------------------------------
# Palette. Decision colours are categorical slots 1-2 of the validated
# reference palette (blue / orange: CVD dE 24.7 light, 26.8 dark); "later"
# and the first-come-first-served baseline are deliberately neutral grey so
# the eye goes to what VectorSentry does. Status colours are reserved for
# pass / fail and always ship with an icon.
# --------------------------------------------------------------------------

_DARK = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
PAL = {
    "reembed": "#3987e5" if _DARK else "#2a78d6",
    "patch": "#d95926" if _DARK else "#eb6834",
    "later": "#898781",
    "live": "#3987e5" if _DARK else "#2a78d6",
    "reference": "#898781",
    "good": "#0ca30c",
    "critical": "#d03b3b",
    "grid": "#2c2c2a" if _DARK else "#e1e0d9",
    "axis": "#383835" if _DARK else "#c3c2b7",
    "muted": "#898781",
    "ink2": "#c3c2b7" if _DARK else "#52514e",
    "surface": "#0e1117" if _DARK else "#ffffff",
}
DECISION_LABEL = {"full_reembed": "Re-embed", "patch": "Quick patch", "defer": "Later"}
DECISION_COLOR = {"full_reembed": PAL["reembed"], "patch": PAL["patch"], "defer": PAL["later"]}
PLOT_CONFIG = {"displayModeBar": False}

# Colours use translucent fills and inherit the text colour so the page reads
# correctly in both Streamlit themes.
st.markdown(
    """
    <style>
    .block-container { padding-top: 2.2rem; }
    .hero-title { font-size: 2.1rem; font-weight: 750; letter-spacing: -.02em; margin: 0; }
    .hero-sub { opacity: .72; margin: .2rem 0 1rem; font-size: 1.02rem; }

    .banner { border-radius: 14px; padding: 1rem 1.25rem; margin: .4rem 0 .8rem; border: 1px solid; }
    .banner .head { font-size: 1.4rem; font-weight: 700; line-height: 1.2; }
    .banner .body { margin-top: .3rem; font-size: 1rem; opacity: .9; }
    .banner.good { background: rgba(12,163,12,.08); border-color: rgba(12,163,12,.40); }
    .banner.bad  { background: rgba(208,59,59,.08); border-color: rgba(208,59,59,.45); }
    .banner.work { background: rgba(42,120,214,.08); border-color: rgba(42,120,214,.45); }
    .banner.warn { background: rgba(250,178,25,.10); border-color: rgba(250,178,25,.55); }

    .steps { display: flex; gap: .5rem; margin: .2rem 0 1.1rem; flex-wrap: wrap; }
    .step { flex: 1 1 140px; border-radius: 10px; padding: .5rem .75rem; border: 1px solid rgba(137,135,129,.35); opacity: .5; }
    .step .n { font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; }
    .step .t { font-weight: 700; }
    .step .s { font-size: .8rem; }
    .step.done { opacity: 1; border-color: rgba(12,163,12,.5); background: rgba(12,163,12,.07); }
    .step.active { opacity: 1; border-color: rgba(42,120,214,.7); background: rgba(42,120,214,.09); box-shadow: 0 0 0 2px rgba(42,120,214,.22); }
    .step.blocked { opacity: 1; border-color: rgba(250,178,25,.7); background: rgba(250,178,25,.10); }

    .section-kicker { font-size: .72rem; text-transform: uppercase; letter-spacing: .12em; opacity: .6; margin-top: 1.2rem; }
    .section-title { font-size: 1.45rem; font-weight: 720; margin: .1rem 0 .1rem; letter-spacing: -.01em; }
    .section-sub { opacity: .7; margin-bottom: .6rem; }

    .bigstat { display: flex; gap: 2.2rem; flex-wrap: wrap; align-items: flex-end; margin: .2rem 0 .4rem; }
    .bigstat .v { font-size: 3rem; font-weight: 760; line-height: 1; letter-spacing: -.03em; }
    .bigstat .l { font-size: .9rem; opacity: .75; margin-top: .3rem; max-width: 18rem; }
    .bigstat .vs .v { font-size: 1.9rem; opacity: .55; }

    .chip { display: inline-flex; align-items: center; gap: .35rem; margin-right: .9rem; font-size: .88rem; }
    .chip i { width: .7rem; height: .7rem; border-radius: 3px; display: inline-block; }

    .feed-item { display: flex; gap: .6rem; padding: .42rem .1rem; border-bottom: 1px solid rgba(137,135,129,.18); font-size: .92rem; }
    .feed-item .ic { width: 1.5rem; text-align: center; }
    .feed-item time { opacity: .55; font-size: .78rem; min-width: 4rem; font-variant-numeric: tabular-nums; padding-top: .1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the benchmark and preparing both databases (first run only)...")
def _load_context():
    cfg = load_config(CONFIG_PATH)
    set_seed(cfg.seed)
    return build_context(cfg)


ctx = _load_context()

if "live_sentry_state" not in st.session_state:
    st.session_state.live_sentry_state = LiveSentryState(ctx.benchmark, ctx.cfg.live_sentry.poll_interval_seconds)

if "live_sentry_thread_started" not in st.session_state:
    thread = threading.Thread(target=monitor_loop, args=(st.session_state.live_sentry_state, ctx), daemon=True, name="live-sentry-monitor")
    thread.start()
    st.session_state.live_sentry_thread_started = True

state: LiveSentryState = st.session_state.live_sentry_state
n_canary = ctx.canary_size


# --------------------------------------------------------------------------
# Plain-language translation of the engine's event log
# --------------------------------------------------------------------------

_EVENT_RULES: list[tuple[str, str, callable]] = [
    (r"Live Sentry started", "🟢", lambda m: "Monitoring started."),
    (r"Simulated embedding provider update: (\d+)", "💥",
     lambda m: f"Provider update: **{int(m[1]):,} documents** re-embedded by a different model."),
    (r"no clean documents remain", "ℹ️", lambda m: "Nothing left to break. Reset to start over."),
    (r"ALERT: .*\((\d+) documents flagged\)", "🚨",
     lambda m: f"Problem detected: **{int(m[1]):,} documents** look out of place."),
    (r"building prioritized schedule for (\d+)", "🧭",
     lambda m: f"Ranking {int(m[1]):,} suspicious documents by search traffic."),
    (r"Plan: (\d+) full re-embeds, (\d+) patches", "🛠️",
     lambda m: f"Plan: re-embed **{int(m[1]):,}**, quick-patch **{int(m[2]):,}**."),
    (r"Verification PASSED for (\d+) of (\d+) rollout wave\(s\) \(canary ([\d.]+) -> ([\d.]+)\)", "✅",
     lambda m: f"Safety check passed {m[1]} of {m[2]} batches; search health {float(m[3]):.0%} → **{float(m[4]):.0%}**."),
    (r"Verification FAILED \(canary would drop ([\d.]+) -> ([\d.]+)\).*budget level ([\d.]+)", "⚠️",
     lambda m: f"Safety check blocked every batch. Retrying with a {float(m[3]):.0%} budget."),
    (r"Verification failed \d+ times in a row.*\((\d+) documents\)", "✋",
     lambda m: f"**{int(m[1]):,} documents** set aside for a human to review."),
    (r"Detector nominal", "🎉", lambda m: "Back to normal."),
    (r"Clean-up: upgraded (\d+) patched documents.*canary ([\d.]+) -> ([\d.]+)\); (\d+) patches left", "✨",
     lambda m: f"Clean-up: **{int(m[1]):,}** quick patches replaced by exact re-embeds; search health **{float(m[3]):.1%}**."),
    (r"Automation paused", "✋", lambda m: "Paused: the rest needs a human decision."),
    (r"Reset:", "↺", lambda m: "Demo reset."),
]


def _friendly_event(message: str) -> tuple[str, str]:
    for pattern, icon, render in _EVENT_RULES:
        match = re.search(pattern, message)
        if match:
            return icon, render(match)
    return "•", message


def _md_bold(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def _section(kicker: str, title: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="section-kicker">{kicker}</div><div class="section-title">{title}</div>'
        + (f'<div class="section-sub">{_md_bold(sub)}</div>' if sub else ""),
        unsafe_allow_html=True,
    )


def _banner(kind: str, head: str, body: str) -> None:
    st.markdown(f'<div class="banner {kind}"><div class="head">{head}</div><div class="body">{_md_bold(body)}</div></div>',
                unsafe_allow_html=True)


def _stepper(status: str, incident_active: bool) -> None:
    steps = [("Watch", "Live vs. reference"), ("Detect", "Spot out-of-place vectors"),
             ("Prioritize & fix", "Busiest documents first"), ("Verify", f"Replay {n_canary} searches per batch")]
    if status == STATUS_ALERT or (incident_active and status not in (STATUS_REMEDIATING, STATUS_NEEDS_REVIEW)):
        active, css = 1, "active"
    elif status == STATUS_REMEDIATING:
        active, css = 2, "active"
    elif status == STATUS_NEEDS_REVIEW:
        active, css = 3, "blocked"
    elif status == STATUS_RESOLVED:
        active, css = 4, "done"
    else:
        active, css = 0, "active"
    html = []
    for i, (title, sub) in enumerate(steps):
        klass = "done" if i < active else (css if i == active else "")
        if status == STATUS_NEEDS_REVIEW and i == 3:
            title, sub = "Needs review", "Some fixes failed the check"
        html.append(f'<div class="step {klass}"><div class="n">Step {i + 1}</div><div class="t">{title}</div><div class="s">{sub}</div></div>')
    st.markdown(f'<div class="steps">{"".join(html)}</div>', unsafe_allow_html=True)


def _style(fig: go.Figure, height: int, *, legend: bool = False, hovermode: str = "closest") -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=4, r=12, t=8, b=4),
        hovermode=hovermode,
        showlegend=legend,
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=12)),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12),
        hoverlabel=dict(font_size=12),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, linecolor=PAL["axis"], tickfont=dict(color=PAL["muted"]), title_font=dict(color=PAL["ink2"]))
    fig.update_yaxes(gridcolor=PAL["grid"], gridwidth=1, zeroline=False, tickfont=dict(color=PAL["muted"]),
                     title_font=dict(color=PAL["ink2"]))
    return fig


def _chips(items: list[tuple[str, str]]) -> None:
    st.markdown("".join(f'<span class="chip"><i style="background:{c}"></i>{label}</span>' for label, c in items),
                unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

def _health_timeline(history: list[dict]) -> None:
    """Small multiples on a shared time axis: search health (top) and
    suspicious-document count (bottom). Two charts, never two y-axes."""
    df = pd.DataFrame(history)
    if len(df) < 2 or "health" not in df:
        st.caption("The timeline fills in after the monitor's first few checks.")
        return
    df = df.dropna(subset=["health"])
    df["health_pct"] = df["health"] * 100
    incident = df["status"].isin([STATUS_ALERT, STATUS_REMEDIATING, STATUS_NEEDS_REVIEW])

    fig = go.Figure()
    fig.add_hline(y=100, line=dict(color=PAL["reference"], width=1))
    fig.add_annotation(x=df["time"].iloc[0], y=100, text="Healthy reference", showarrow=False, xanchor="left",
                       yanchor="bottom", font=dict(color=PAL["muted"], size=11))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["health_pct"], mode="lines", line=dict(color=PAL["live"], width=2.5),
        fill="tozeroy", fillcolor="rgba(42,120,214,.10)", name="Live search health",
        hovertemplate="%{x|%H:%M:%S}<br><b>%{y:.1f}%</b> of normal results returned<extra></extra>",
    ))
    if incident.any():
        low = df.loc[df["health_pct"].idxmin()]
        fig.add_annotation(x=low["time"], y=low["health_pct"], text=f"Low point {low['health_pct']:.0f}%", showarrow=True,
                           arrowhead=0, ay=28, ax=0, font=dict(color=PAL["ink2"], size=11), arrowcolor=PAL["axis"])
    y_min = max(0, min(90, df["health_pct"].min() - 4))
    fig.update_yaxes(range=[y_min, 101.5], ticksuffix="%", title="Search health")
    st.plotly_chart(_style(fig, 230), width="stretch", config=PLOT_CONFIG)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["time"], y=df["db_a_flagged"], mode="lines", name="Normal background",
                              line=dict(color=PAL["reference"], width=1.5),
                              hovertemplate="%{y:,} flagged in the reference<extra></extra>"))
    fig2.add_trace(go.Scatter(x=df["time"], y=df["db_b_flagged"], mode="lines", name="Live database",
                              line=dict(color=PAL["live"], width=2), fill="tonexty", fillcolor="rgba(42,120,214,.10)",
                              hovertemplate="%{y:,} flagged in live<extra></extra>"))
    fig2.update_yaxes(title="Suspicious docs", rangemode="tozero")
    st.plotly_chart(_style(fig2, 150, hovermode="x unified"), width="stretch", config=PLOT_CONFIG)


def _priority_map(rows: pd.DataFrame) -> None:
    """Every suspicious document: search traffic (x, log) vs. chance it is
    broken (y), coloured by the scheduler's decision."""
    fig = go.Figure()
    # Detector probabilities come in steps of 1/12; a tiny deterministic
    # vertical spread keeps overlapping documents visible (hover shows the
    # exact value).
    jitter = (np.sin(rows["doc_index"].to_numpy() * 12.9898) * 43758.5453 % 1 - 0.5) * 0.03
    # Documents no logged search has shown carry only a tiny traffic prior;
    # pin everything below 0.01x average to the axis floor (hover keeps the
    # true value) so the long tail does not stretch the axis over 4 decades.
    floor = 0.01
    rows = rows.assign(x=rows["traffic_vs_average"].clip(lower=floor))
    for action in ("defer", "patch", "full_reembed"):  # later underneath, re-embeds on top
        sub = rows[rows["action"] == action]
        if sub.empty:
            continue
        j = jitter[rows["action"].to_numpy() == action]
        fig.add_trace(go.Scattergl(
            x=sub["x"], y=sub["severity"] + j, mode="markers", name=DECISION_LABEL[action],
            marker=dict(size=9 if action != "defer" else 7, color=DECISION_COLOR[action],
                        opacity=0.92 if action != "defer" else 0.45, line=dict(width=1, color=PAL["surface"])),
            customdata=np.c_[sub["doc_id"], sub["severity"], sub["traffic_vs_average"]],
            hovertemplate=("<b>Document %{customdata[0]}</b><br>" + DECISION_LABEL[action]
                           + "<br>Search traffic %{customdata[2]:.2f}× average<br>Chance broken %{customdata[1]:.0%}<extra></extra>"),
        ))
    x_max = max(20.0, float(rows["x"].max()) * 1.8)
    ticks = [t for t in (0.01, 0.1, 1, 10, 100, 1000) if t <= x_max]
    fig.update_xaxes(type="log", title="Search traffic (× average document)", range=[np.log10(floor * 0.7), np.log10(x_max)],
                     tickvals=ticks, ticktext=["≤0.01×" if t == floor else f"{t:g}×" for t in ticks])
    fig.update_yaxes(title="Chance it is broken", range=[0.42, 1.04], tickformat=".0%")
    top = rows.sort_values("traffic_vs_average", ascending=False).iloc[0]
    fig.add_annotation(x=np.log10(top["x"]), y=top["severity"], text="Busiest document",
                       showarrow=True, arrowhead=0, ax=-50, ay=-26, font=dict(size=11, color=PAL["ink2"]),
                       arrowcolor=PAL["axis"])
    st.plotly_chart(_style(fig, 380), width="stretch", config=PLOT_CONFIG)


@lru_cache(maxsize=16)
def _protection_curves(doc_index: tuple[int, ...], p: tuple[float, ...], n_points: int = 41):
    """Expected share of affected search traffic protected vs. budget, for
    VectorSentry's MCKP plan and for first-come-first-served re-embedding
    (ingestion order), over budgets from 0 to "re-embed everything"."""
    idx, prob = np.array(doc_index), np.array(p)
    affected = prob * ctx.phase2.traffic[idx]
    total = affected.sum()
    full_cost = ctx.cfg.remediation.cost_full_reembed
    budgets = np.linspace(0, len(idx) * full_cost, n_points)
    ours = [plan_remediation(ctx.cfg, ctx.phase2, idx, prob, b).plan.total_benefit / total for b in budgets]
    fifo_order = np.argsort(idx)  # ingestion order = document order in the corpus
    fifo_cum = np.r_[0.0, np.cumsum(affected[fifo_order])] / total
    fifo = [fifo_cum[min(len(idx), int(b // full_cost))] for b in budgets]
    return budgets / (len(idx) * full_cost), np.clip(ours, 0, 1), np.array(fifo)


def _protection_chart(rows: pd.DataFrame, budget_share: float) -> tuple[float, float]:
    x, ours, fifo = _protection_curves(tuple(int(i) for i in rows["doc_index"]), tuple(round(float(v), 6) for v in rows["severity"]))
    at_ours, at_fifo = float(np.interp(budget_share, x, ours)), float(np.interp(budget_share, x, fifo))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x * 100, y=fifo * 100, mode="lines", name="First-come-first-served",
                             line=dict(color=PAL["reference"], width=2),
                             hovertemplate="First-come-first-served<br>%{x:.0f}% budget → %{y:.0f}% protected<extra></extra>"))
    fig.add_trace(go.Scatter(x=x * 100, y=ours * 100, mode="lines", name="VectorSentry", fill="tonexty",
                             fillcolor="rgba(42,120,214,.14)", line=dict(color=PAL["reembed"], width=2.5),
                             hovertemplate="VectorSentry<br>%{x:.0f}% budget → %{y:.0f}% protected<extra></extra>"))
    fig.add_vline(x=budget_share * 100, line=dict(color=PAL["axis"], width=1))
    side = -1 if budget_share > 0.75 else 1  # labels to the right of the marker, or left near the edge
    fig.add_annotation(x=budget_share * 100, y=at_ours * 100, text=f"<b>{at_ours:.0%}</b>", showarrow=True, arrowhead=0,
                       ax=34 * side, ay=-22, font=dict(size=13, color=PAL["reembed"]), arrowcolor=PAL["reembed"])
    fig.add_annotation(x=budget_share * 100, y=at_fifo * 100, text=f"{at_fifo:.0%}", showarrow=True, arrowhead=0,
                       ax=30 * side, ay=18, font=dict(size=12, color=PAL["muted"]), arrowcolor=PAL["reference"])
    fig.add_annotation(x=budget_share * 100, y=2, text="This round's budget", showarrow=False, yanchor="bottom",
                       xanchor="left" if side > 0 else "right", xshift=4 * side, font=dict(size=11, color=PAL["muted"]))
    # Direct labels at the right-hand end replace a legend box.
    fig.add_annotation(x=60, y=float(np.interp(0.6, x, ours)) * 100, text="VectorSentry", showarrow=False, yanchor="bottom",
                       yshift=6, font=dict(size=12, color=PAL["reembed"]))
    fig.add_annotation(x=60, y=float(np.interp(0.6, x, fifo)) * 100, text="First-come-first-served", showarrow=False,
                       yanchor="top", yshift=-6, font=dict(size=12, color=PAL["muted"]))
    fig.update_xaxes(title="Budget spent (% of re-embedding every suspicious document)", ticksuffix="%", range=[0, 100])
    fig.update_yaxes(title="Affected search traffic protected", ticksuffix="%", range=[0, 110])
    st.plotly_chart(_style(fig, 380), width="stretch", config=PLOT_CONFIG)
    return at_ours, at_fifo


def _split_bars(rows: pd.DataFrame, budget: float) -> None:
    """Where the budget went vs. where the traffic is: two 100%-stacked bars."""
    costs = {"full_reembed": ctx.cfg.remediation.cost_full_reembed, "patch": ctx.cfg.remediation.cost_patch}
    spent = {a: float((rows["action"] == a).sum() * c) for a, c in costs.items()}
    unspent = max(0.0, budget - sum(spent.values()))
    affected = rows["severity"] * rows["traffic_share"]
    traffic = {a: float(affected[rows["action"] == a].sum()) for a in ("full_reembed", "patch", "defer")}
    t_total = sum(traffic.values()) or 1.0
    b_total = max(budget, 1e-9)

    segments = [
        ("full_reembed", spent["full_reembed"] / b_total, traffic["full_reembed"] / t_total),
        ("patch", spent["patch"] / b_total, traffic["patch"] / t_total),
        ("defer", unspent / b_total, traffic["defer"] / t_total),
    ]
    fig = go.Figure()
    for action, b_share, t_share in segments:
        label = {"defer": "Unspent / waiting"}.get(action, DECISION_LABEL[action])
        vals = [t_share * 100, b_share * 100]
        fig.add_trace(go.Bar(
            y=["Affected traffic", "Budget"], x=vals, orientation="h", name=label,
            marker=dict(color=DECISION_COLOR[action], line=dict(color=PAL["surface"], width=2)),
            text=[f"{v:.0f}%" if v >= 7 else "" for v in vals], textposition="inside", insidetextanchor="middle",
            textfont=dict(color="#ffffff", size=12),
            hovertemplate=f"{label}<br>%{{y}}: %{{x:.1f}}%<extra></extra>",
        ))
    fig.update_layout(barmode="stack", bargap=0.45)
    fig.update_xaxes(range=[0, 100], ticksuffix="%", showticklabels=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12, color=PAL["ink2"]))
    st.plotly_chart(_style(fig, 150), width="stretch", config=PLOT_CONFIG)


def _safety_chart(waves: list[dict]) -> None:
    if not waves:
        st.caption("No batch has been tested yet.")
        return
    df = pd.DataFrame(waves)
    df["attempt"] = np.arange(1, len(df) + 1)
    df["label"] = np.where(df["passed"], "✓", "✗")
    fig = go.Figure()
    for passed, name, color in ((True, "Passed: went live", PAL["good"]), (False, "Rejected: rolled back", PAL["critical"])):
        sub = df[df["passed"] == passed]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=sub["attempt"], y=sub["n_documents"], name=name, marker=dict(color=color, line=dict(color=PAL["surface"], width=1)),
            text=sub["label"], textposition="outside", textfont=dict(color=color, size=13),
            customdata=np.c_[sub["round"], sub["canary_before"] * 100, sub["canary_after"] * 100, sub["depth"]],
            hovertemplate=("Batch %{x} (round %{customdata[0]})<br>%{y} documents<br>"
                           "Search health %{customdata[1]:.1f}% → %{customdata[2]:.1f}%<extra>" + name + "</extra>"),
        ))
    y_top = float(df["n_documents"].max()) * 1.32
    for r, grp in df.groupby("round"):
        lo, hi = grp["attempt"].min() - 0.5, grp["attempt"].max() + 0.5
        if r > 1:
            fig.add_vline(x=lo, line=dict(color=PAL["axis"], width=1))
        label = "First plan" if r == 1 else ("Re-plan: rejected patches → re-embeds" if r == 2 else f"Re-plan {r - 1}")
        fig.add_annotation(x=(lo + hi) / 2, y=y_top, text=label, showarrow=False, yanchor="top",
                           font=dict(size=11, color=PAL["ink2"]))
    fig.update_layout(bargap=0.25)
    fig.update_xaxes(title="Batch, in the order it was tested", dtick=1 if len(df) <= 30 else None)
    fig.update_yaxes(title="Documents in batch", range=[0, y_top])
    st.plotly_chart(_style(fig, 260, legend=True), width="stretch", config=PLOT_CONFIG)


def _activity_feed(events: list[tuple[str, str]], limit: int = 9) -> None:
    if not events:
        st.caption("Waiting for the first check...")
        return
    rows = []
    for timestamp, message in reversed(events[-limit:]):
        icon, text = _friendly_event(message)
        rows.append(f'<div class="feed-item"><time>{timestamp}</time><span class="ic">{icon}</span><span>{_md_bold(text)}</span></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)


def _embedding_map() -> None:
    db_a, db_b = state.embedding_snapshot()
    sample_size = min(900, len(db_a))
    sample_idx = np.linspace(0, len(db_a) - 1, sample_size, dtype=int)
    coords = PCA(n_components=2, random_state=42).fit_transform(np.vstack([db_a[sample_idx], db_b[sample_idx]]))
    a_xy, b_xy = coords[:sample_size], coords[sample_size:]
    moved = np.linalg.norm(db_b[sample_idx] - db_a[sample_idx], axis=1) > 1e-6

    fig = go.Figure()
    line_x, line_y = [], []
    for a, b in zip(a_xy[moved], b_xy[moved]):
        line_x.extend([a[0], b[0], None])
        line_y.extend([a[1], b[1], None])
    fig.add_trace(go.Scattergl(x=a_xy[~moved, 0], y=a_xy[~moved, 1], mode="markers", name="Unchanged document",
                               marker=dict(size=5, color=PAL["reference"], opacity=.35), hoverinfo="skip"))
    if moved.any():
        fig.add_trace(go.Scattergl(x=line_x, y=line_y, mode="lines", name="How far it moved",
                                   line=dict(color=PAL["patch"], width=1), opacity=.5, hoverinfo="skip"))
        fig.add_trace(go.Scattergl(x=b_xy[moved, 0], y=b_xy[moved, 1], mode="markers", name="Changed document (live position)",
                                   marker=dict(size=7, color=PAL["patch"]), hoverinfo="skip"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    st.plotly_chart(_style(fig, 420, legend=True), width="stretch", config=PLOT_CONFIG)
    st.caption(f"{int(moved.sum()):,} of {sample_size:,} sampled documents sit somewhere different from the reference. "
               "Each orange line joins a document's correct position to where the live database put it."
               if moved.any() else "Every sampled document is exactly where the reference puts it.")


# --------------------------------------------------------------------------
# Header and controls
# --------------------------------------------------------------------------

BUDGETS = {0.1: "Tight · 10%", 0.3: "Lean · 30%", 0.6: "Roomy · 60%", 1.0: "Unlimited"}

head_l, head_r = st.columns([5, 1])
with head_l:
    st.markdown('<p class="hero-title">🛡️ VectorSentry Live Sentry</p>'
                '<p class="hero-sub">Break the live search index with one click. Watch VectorSentry find the damage, '
                'decide what to fix first, and repair it without ever making search worse.</p>', unsafe_allow_html=True)
with head_r:
    with st.popover("How it works", width="stretch"):
        st.markdown(
            f"""
**Two copies of the same search index.** The *reference* is never touched; the
*live* one is what users search, and the one we break.

**The incident.** An embedding provider silently changes its model, so some
documents get vectors that no longer match the rest. Search gets worse and
nothing errors.

**What VectorSentry does on its own:**
1. **Detects** vectors written by the wrong model, by comparing them with a
   small sample of documents embedded by both models.
2. **Prioritizes** under a budget: re-embeds the busiest documents, patches a
   wider tier cheaply, defers the rest (a knapsack optimization).
3. **Verifies** every batch by replaying {n_canary} logged searches against a
   shadow copy before it goes live, and rolls back batches that hurt search.
"""
        )
        poll = st.slider("Check every (seconds)", 2.0, 5.0, ctx.cfg.live_sentry.poll_interval_seconds, step=0.5)
        state.set_poll_interval(poll)

# --------------------------------------------------------------------------
# Live panel (auto-refreshing). The control bar lives inside it because a
# fragment may only place widgets in its own containers, and the Simulate
# button must grey out live while an incident runs.
# --------------------------------------------------------------------------

@st.fragment(run_every=UI_REFRESH_SECONDS)
def live_panel() -> None:
    snap = state.snapshot()
    status = snap["status"]
    ref, live = snap["last_db_a_flagged"], snap["last_db_b_flagged"]
    excess = max(0, live - ref)
    summary = snap["last_incident_summary"]
    health = snap["last_health"]

    with st.container(border=True):
        c_budget, c_safety, c_go, c_reset = st.columns([3.2, 1.8, 1.7, 0.9], vertical_alignment="bottom")
        with c_budget:
            budget_level = st.segmented_control(
                "Fix budget per round", list(BUDGETS), default=0.3, format_func=BUDGETS.get, key="budget_level",
                help="How many of the detected documents could be fully re-embedded each round. Tight budgets force "
                     "VectorSentry to choose; at Unlimited it can simply re-embed everything.",
            ) or 0.3
        with c_safety:
            safety = st.segmented_control(
                "Safety check", ["Balanced", "Strict"], default="Balanced", key="safety",
                help="Balanced allows a 1% wobble in search health per rollout; Strict allows none and may pause for a human.",
            ) or "Balanced"
        state.configure_demo(budget_level, 0.0 if safety == "Strict" else 0.01)
        with c_go:
            if st.button("💥 Simulate provider update", type="primary", width="stretch", disabled=snap["incident_active"],
                         key="simulate"):
                state.inject_contamination(ctx, ctx.cfg.live_sentry.injection_batch_fraction, seed=None)
                st.rerun()
        with c_reset:
            if st.button("↺ Reset", width="stretch", key="reset"):
                state.reset(ctx.benchmark)
                st.rerun()

    # ---- Status ----
    if snap["incident_active"] and status not in (STATUS_ALERT, STATUS_REMEDIATING):
        _banner("work", "🔎 Measuring the damage…", "A provider update just changed part of the live index.")
    elif status == STATUS_ALERT:
        _banner("bad", "🚨 Problem detected", f"**{live:,}** live documents look out of place (normal background: **{ref:,}**).")
    elif status == STATUS_REMEDIATING:
        _banner("work", "🛠️ Repairing automatically",
                f"Fixing the busiest documents first, in verified batches. **{excess:,}** suspicious documents left.")
    elif status == STATUS_NEEDS_REVIEW:
        _banner("warn", "✋ Paused for a human",
                f"**{excess:,}** documents remain, but their fixes failed the safety check or were already tried, "
                "so VectorSentry stopped rather than make search worse.")
    elif status == STATUS_RESOLVED:
        fixed = summary["remediated"] if summary else snap["n_remediated"]
        if snap["n_patched"]:
            _banner("good", "🎉 Incident resolved, polishing",
                    f"**{fixed:,}** documents repaired. **{snap['n_patched']:,}** of them carry a quick patch; VectorSentry is "
                    "now replacing those with exact re-embeds in the background, busiest first.")
        else:
            _banner("good", "🎉 Back to normal", f"**{fixed:,}** documents repaired; every change passed the safety check.")
    else:
        _banner("good", "✅ All clear", "The live index matches the reference. Press **💥 Simulate provider update** to start.")
    _stepper(status, snap["incident_active"])

    # ---- KPI tiles ----
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Search health", f"{health:.1%}" if health is not None else "–",
              f"{(health - 1) * 100:+.1f} pts vs. normal" if health is not None and health < 0.9995 else None,
              border=True, help=f"Share of the results {n_canary} popular logged searches returned before the incident that "
                                "they still get back from the live index.")
    k2.metric("Suspicious documents", f"{live:,}", f"{live - ref:+,} vs. normal" if live != ref else None,
              delta_color="inverse", border=True, help=f"The detector also flags {ref:,} in the untouched reference: its false-alarm floor.")
    k3.metric("Repaired & verified", f"{snap['n_remediated']:,}", border=True,
              help="Documents whose fix passed the safety check and went live.")
    budget_this_incident = snap["incident_budget_spent"]
    k4.metric("Budget spent this incident", f"{budget_this_incident:,.0f}", border=True,
              help="In full re-embed units: a re-embed costs 1, a quick patch 0.1.")

    # ---- Timeline + feed ----
    _section("Live", "Search health over time", "Live search quality drops when the update lands and climbs back as verified fixes ship.")
    left, right = st.columns([3, 1.3])
    with left:
        _health_timeline(snap["history"])
    with right:
        st.markdown("**What just happened**")
        _activity_feed(snap["event_log"])

    # ---- The decision ----
    rounds = snap["incident_rounds"]
    _section("The decision", "How VectorSentry chose what to fix",
             "Search traffic is extremely uneven: the busiest 1% of documents get about a third of all searches. "
             "VectorSentry spends its budget where the traffic is.")
    picked = None
    if len(rounds) > 1:
        picked = st.segmented_control(
            "Repair round", list(range(1, len(rounds) + 1)), default=1, key="round_pick",
            format_func=lambda r: f"Round {r} · {len(rounds[r - 1]['rows']):,} docs",
            help="Each round re-plans for the documents still suspicious. Round 1 is the big decision.",
        )
    current = rounds[(picked or 1) - 1] if rounds else None
    rows = pd.DataFrame(current["rows"]) if current else pd.DataFrame()
    if rows.empty:
        st.info("Press **💥 Simulate provider update**: the plan for each repair round appears here.", icon="🧭")
    else:
        budget_units = current["budget"]
        full_cost = ctx.cfg.remediation.cost_full_reembed
        budget_share = min(1.0, budget_units / max(1e-9, len(rows) * full_cost))
        counts = rows["action"].value_counts()

        map_col, curve_col = st.columns(2, gap="large")
        with curve_col:
            st.markdown("**Prioritized vs. first-come-first-served**")
            at_ours, at_fifo = _protection_chart(rows, budget_share)
        with map_col:
            st.markdown("**Every suspicious document, and what it gets**")
            _chips([(f"Re-embed · {counts.get('full_reembed', 0):,}", PAL["reembed"]),
                    (f"Quick patch · {counts.get('patch', 0):,}", PAL["patch"]),
                    (f"Later · {counts.get('defer', 0):,}", PAL["later"])])
            _priority_map(rows)

        st.markdown(
            f'<div class="bigstat"><div><div class="v">{at_ours:.0%}</div><div class="l">of the affected search traffic '
            f'protected this round, using <b>{budget_share:.0%}</b> of the cost of re-embedding everything</div></div>'
            f'<div class="vs"><div class="v">{at_fifo:.0%}</div><div class="l">with the same budget spent first-come-first-served '
            f'(the existing practice)</div></div></div>', unsafe_allow_html=True)
        _split_bars(rows, budget_units)
        if budget_share >= 0.999:
            st.caption("The budget covers re-embedding every suspicious document, so there is nothing to choose. "
                       "Pick **Tight** or **Lean** and run another incident to watch the trade-offs.")
        else:
            st.caption("A re-embed costs 10× a quick patch and restores a document exactly; a patch is cheap but only "
                       "partly accurate, and would damage a document that was not actually broken, so it is only used "
                       "when the detector is confident.")

    # ---- Safety check ----
    _section("Safety", "Every batch is tested before it goes live",
             f"Each batch is built in a shadow copy and {n_canary} popular logged searches are replayed against it. "
             "A batch that would make search worse is rolled back, and its budget is re-planned as exact re-embeds."
             + (f" Showing repair round {picked or 1}." if current else ""))
    _safety_chart(current["waves"] if current else [])

    # ---- Incident report ----
    if summary and status in (STATUS_RESOLVED, STATUS_NEEDS_REVIEW):
        db_a, db_b = state.embedding_snapshot()
        still_wrong = int(np.sum(np.any(db_b != db_a, axis=1)))
        caught = max(0, summary["initial_flagged"] - ref)
        with st.container(border=True):
            st.markdown("#### 📋 Incident report")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Changed by the update", f"{summary['injected']:,}")
            r2.metric("Caught", f"{caught:,}", f"{caught / max(1, summary['injected']):.0%} of changed", delta_color="off")
            r3.metric("Repaired", f"{summary['remediated']:,}")
            r4.metric("Still differs from reference", f"{still_wrong:,}", delta_color="off",
                      help="Patched documents are close to the original but not identical; only knowable in a simulation.")

    # ---- Under the hood ----
    with st.expander("Under the hood: embedding map and raw engine log"):
        if st.toggle("Show embedding map", help="Projects document vectors onto 2D. Slower, so off by default."):
            _embedding_map()
        st.code("\n".join(f"{t}  {m}" for t, m in reversed(snap["event_log"][-50:])) or "(empty)", language=None)


live_panel()
