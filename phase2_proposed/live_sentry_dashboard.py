"""Visual Phase 2 showcase -- the autonomous VectorSentry Live Sentry.

The page tells one story, top to bottom:
  1. What state is the system in, in plain English, and what should I do?
  2. How far along is the automatic fix?
  3. The numbers that matter, and how they changed over time.
  4. A plain-language activity feed.
  5. Optional "under the hood" detail for technical viewers.
"""

from __future__ import annotations

import re
import sys
import threading
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
    STATUS_STABLE,
    LiveSentryState,
    build_context,
    monitor_loop,
)

CONFIG_PATH = "phase2_proposed/configs/default.yaml"
UI_REFRESH_SECONDS = 1.5

COLOR_REFERENCE = "#64748b"
COLOR_LIVE = "#f97316"
COLOR_GOOD = "#16a34a"
COLOR_BAD = "#dc2626"
COLOR_WARN = "#d97706"
COLOR_INFO = "#2563eb"

st.set_page_config(page_title="VectorSentry Live Sentry", page_icon="🛡️", layout="wide")

# Colors use translucent fills and inherit the text color so the page reads
# correctly in both Streamlit's light and dark themes.
st.markdown(
    """
    <style>
    .banner { border-radius: 14px; padding: 1.1rem 1.3rem; margin: .2rem 0 1rem; border: 1px solid; }
    .banner .head { font-size: 1.55rem; font-weight: 700; line-height: 1.2; }
    .banner .body { margin-top: .35rem; font-size: 1.02rem; opacity: .9; }
    .banner .next { margin-top: .6rem; font-size: .95rem; font-weight: 600; }
    .banner.good { background: rgba(22,163,74,.10); border-color: rgba(22,163,74,.45); }
    .banner.bad  { background: rgba(220,38,38,.10); border-color: rgba(220,38,38,.45); }
    .banner.work { background: rgba(37,99,235,.10); border-color: rgba(37,99,235,.45); }
    .banner.warn { background: rgba(217,119,6,.12); border-color: rgba(217,119,6,.50); }

    .steps { display: flex; gap: .5rem; margin: .2rem 0 1rem; flex-wrap: wrap; }
    .step { flex: 1 1 140px; border-radius: 10px; padding: .55rem .75rem; border: 1px solid rgba(100,116,139,.35); opacity: .55; }
    .step .n { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; }
    .step .t { font-weight: 700; }
    .step .s { font-size: .8rem; }
    .step.done { opacity: 1; border-color: rgba(22,163,74,.55); background: rgba(22,163,74,.08); }
    .step.active { opacity: 1; border-color: rgba(37,99,235,.7); background: rgba(37,99,235,.10); box-shadow: 0 0 0 2px rgba(37,99,235,.25); }
    .step.blocked { opacity: 1; border-color: rgba(217,119,6,.7); background: rgba(217,119,6,.12); }

    .feed-item { display: flex; gap: .6rem; padding: .45rem .2rem; border-bottom: 1px solid rgba(100,116,139,.18); }
    .feed-item .ic { width: 1.6rem; text-align: center; }
    .feed-item time { opacity: .6; font-size: .8rem; min-width: 4.2rem; font-variant-numeric: tabular-nums; }
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
n_golden = ctx.canary_size


# --------------------------------------------------------------------------
# Plain-language translation of the engine's event log
# --------------------------------------------------------------------------

_EVENT_RULES: list[tuple[str, str, callable]] = [
    (r"Live Sentry started", "🟢", lambda m: "Monitoring started. Watching the reference and live databases."),
    (r"Simulated embedding provider update: (\d+)", "💥",
     lambda m: f"Provider update simulated: **{int(m[1]):,} documents** in the live database were re-embedded by a different model."),
    (r"no clean documents remain", "ℹ️", lambda m: "Nothing left to break: every live document has already been changed once. Reset the demo to start over."),
    (r"ALERT: .*\((\d+) documents flagged\)", "🚨",
     lambda m: f"Problem detected: **{int(m[1]):,} documents** in the live database look out of place."),
    (r"building prioritized schedule for (\d+)", "🧭",
     lambda m: f"Ranking {int(m[1]):,} suspicious documents by how much search traffic depends on them."),
    (r"Plan: (\d+) full re-embeds, (\d+) patches", "🛠️",
     lambda m: f"Fix plan ready: fully re-embed **{int(m[1]):,}** documents and quick-patch **{int(m[2]):,}**."),
    (r"Verification PASSED for (\d+) of (\d+) rollout wave\(s\) \(canary ([\d.]+) -> ([\d.]+)\)", "✅",
     lambda m: f"Safety check passed for {m[1]} of {m[2]} batches (search results match {float(m[3]):.0%} → {float(m[4]):.0%} of normal). Those fixes are now live."),
    (r"Verification FAILED \(canary would drop ([\d.]+) -> ([\d.]+)\).*budget level ([\d.]+)", "⚠️",
     lambda m: f"Safety check failed: search results would match {float(m[1]):.0%} → {float(m[2]):.0%} of normal. Fixes were discarded, nothing went live. Retrying with a larger budget ({float(m[3]):.0%})."),
    (r"Verification failed \d+ times in a row.*\((\d+) documents\)", "✋",
     lambda m: f"The same fix keeps failing the safety check, so **{int(m[1]):,} documents** were set aside for a human to review."),
    (r"Detector nominal", "🎉", lambda m: "Back to normal: the live database looks as healthy as the reference."),
    (r"Automation paused", "✋", lambda m: "Automation paused: the remaining suspicious documents need a human decision."),
    (r"Reset:", "↺", lambda m: "Demo reset: the live database was restored to its clean state."),
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

def _banner(kind: str, head: str, body: str, next_step: str) -> None:
    st.markdown(
        f'<div class="banner {kind}"><div class="head">{head}</div>'
        f'<div class="body">{_md_bold(body)}</div><div class="next">👉 {_md_bold(next_step)}</div></div>',
        unsafe_allow_html=True,
    )


def _stepper(status: str) -> None:
    steps = [
        ("Watch", "Compare live vs reference"),
        ("Detect", "Spot out-of-place documents"),
        ("Fix & verify", f"Test fixes on {n_golden} key queries"),
        ("Back to normal", "Live matches reference"),
    ]
    if status == STATUS_ALERT:
        active, last_css = 1, "active"
    elif status == STATUS_REMEDIATING:
        active, last_css = 2, "active"
    elif status == STATUS_NEEDS_REVIEW:
        active, last_css = 2, "blocked"
    elif status == STATUS_RESOLVED:
        active, last_css = 4, "done"
    else:
        active, last_css = 0, "active"
    html = []
    for i, (title, sub) in enumerate(steps):
        if i < active:
            css = "done"
        elif i == active:
            css = last_css
        else:
            css = ""
        if status == STATUS_NEEDS_REVIEW and i == 2:
            title, sub = "Needs review", "Fix failed the safety check"
        html.append(f'<div class="step {css}"><div class="n">Step {i + 1}</div><div class="t">{title}</div><div class="s">{sub}</div></div>')
    st.markdown(f'<div class="steps">{"".join(html)}</div>', unsafe_allow_html=True)


def _history_chart(history: list[dict]) -> None:
    if len(history) < 2:
        st.caption("The chart fills in after the monitor's first few checks.")
        return
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["db_a_flagged"], mode="lines", name="Normal background (reference)",
        line=dict(color=COLOR_REFERENCE, width=2, dash="dash"),
        hovertemplate="%{y:,} flagged in reference<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["db_b_flagged"], mode="lines", name="Live database",
        line=dict(color=COLOR_LIVE, width=3), fill="tonexty", fillcolor="rgba(249,115,22,.15)",
        hovertemplate="%{y:,} flagged in live<extra></extra>",
    ))
    fig.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0),
        yaxis=dict(title="Suspicious documents", rangemode="tozero"), xaxis=dict(title=None),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("The shaded gap is the problem. When the orange line falls back onto the dashed line, the live database is healthy again.")


def _activity_feed(events: list[tuple[str, str]], limit: int = 8) -> None:
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
                               marker=dict(size=5, color=COLOR_REFERENCE, opacity=.35), hoverinfo="skip"))
    if moved.any():
        fig.add_trace(go.Scattergl(x=line_x, y=line_y, mode="lines", name="How far it moved",
                                   line=dict(color=COLOR_LIVE, width=1), opacity=.5, hoverinfo="skip"))
        fig.add_trace(go.Scattergl(x=b_xy[moved, 0], y=b_xy[moved, 1], mode="markers", name="Changed document (live position)",
                                   marker=dict(size=7, color=COLOR_LIVE), hoverinfo="skip"))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.08, x=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    if moved.any():
        st.caption(f"{int(moved.sum()):,} of {sample_size:,} sampled documents currently sit somewhere different from the reference. "
                   "Each orange line joins a document's correct position to where the live database put it. "
                   "Note that a document re-embedded by the fix moves back and disappears from this view.")
    else:
        st.caption("Every sampled document is exactly where the reference puts it.")


def _plan_table(snap: dict) -> None:
    plan_rows = snap["last_plan_rows"]
    if not plan_rows:
        st.caption("No fix plan yet. A plan appears once a problem has been detected.")
        return
    action_names = {"full_reembed": "🔁 Re-embed", "patch": "🩹 Quick patch", "defer": "⏸️ Later"}
    counts = pd.Series(row["action"] for row in plan_rows).value_counts()
    st.markdown(
        f"Out of **{snap['last_plan_candidate_count']:,}** suspicious documents, the latest plan re-embeds "
        f"**{int(counts.get('full_reembed', 0)):,}**, patches **{int(counts.get('patch', 0)):,}** and leaves "
        f"**{int(counts.get('defer', 0)):,}** for a later round, using **{snap['last_plan_cost']:.1f}** of "
        f"**{snap['last_plan_budget']:.1f}** budget units."
    )
    df = pd.DataFrame(plan_rows[:25])
    df = pd.DataFrame({
        "Document": df["doc_id"],
        "Decision": df["action"].map(action_names),
        "Chance it is broken": df["severity"],
        "Search traffic": df["traffic_share"] * 100,
        "Value per cost": df["priority_ratio"],
    })
    st.dataframe(
        df, hide_index=True, width="stretch",
        column_config={
            "Chance it is broken": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f",
                                                          help="The detector's probability that this document was embedded by the new model."),
            "Search traffic": st.column_config.NumberColumn(format="%.3f%%", help="Share of all search traffic that touches this document."),
            "Value per cost": st.column_config.NumberColumn(format="%.4f", help="Expected search-quality gain per budget unit. Higher values are fixed first."),
        },
    )
    st.caption("Rule of thumb: documents that are **almost certainly broken** *and* **heavily searched** get fixed first. "
               "A full re-embed costs 10× a quick patch but restores the document completely; a patch is cheap but "
               "only partly accurate, and it damages a document that was not actually broken.")


# --------------------------------------------------------------------------
# Sidebar: settings and help, kept out of the way
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Demo settings")
    scenario = st.radio(
        "Scenario",
        ["Smooth run", "Strict safety"],
        captions=["Shows a complete automatic recovery.", "Rejects any fix that hurts search, even slightly."],
        help="Smooth run allows tiny measurement noise (0.01) in the safety check, so you can watch a full recovery. "
             "Strict safety allows none, which often shows VectorSentry stopping and asking for a human.",
    )
    strict = scenario == "Strict safety"
    with st.expander("Advanced"):
        starting_budget = st.slider(
            "Fix budget per round", 0.1, 1.0, 0.3 if strict else 1.0, step=0.1,
            help="Share of the initially detected documents that can be fully re-embedded in each round. "
                 "Lower budgets fix the problem over more rounds.",
        )
        poll_interval = st.slider("Check every (seconds)", 2.0, 5.0, ctx.cfg.live_sentry.poll_interval_seconds, step=0.5)
    verification_tolerance = 0.0 if strict else 0.01
    state.configure_demo(starting_budget, verification_tolerance)
    state.set_poll_interval(poll_interval)

    st.divider()
    st.header("📖 What am I looking at?")
    st.markdown(
        f"""
**Two copies of the same search database:**
- **Reference**: never touched. It shows what "healthy" looks like.
- **Live**: what users search. This is the one we break.

**The incident:** an embedding provider quietly changes its model, so some
documents get vectors that no longer match the rest. Search results get
worse and nobody is told.

**What VectorSentry does on its own:**
1. **Detects** live documents that look out of place.
2. **Prioritizes** the ones that matter most for search traffic.
3. **Verifies** each batch of fixes on {n_golden} important queries before it goes live.
4. **Stops and asks a human** if a fix would make search worse.

**Why the reference shows a few suspicious documents:** the detector always
flags around 1% of any database as unusual. That level is the normal
background, so the live database counts as healthy once it drops to it.
"""
    )


# --------------------------------------------------------------------------
# Main page
# --------------------------------------------------------------------------

st.title("🛡️ VectorSentry Live Sentry")
st.caption("Break the live search database with one click, then watch VectorSentry find and repair the damage on its own, without ever making search worse.")


@st.fragment(run_every=UI_REFRESH_SECONDS)
def live_panel() -> None:
    snap = state.snapshot()
    status = snap["status"]
    ref, live = snap["last_db_a_flagged"], snap["last_db_b_flagged"]
    excess = max(0, live - ref)
    summary = snap["last_incident_summary"]

    # ---- 1. Controls ----
    c1, c2, c3 = st.columns([2, 1, 3])
    with c1:
        if st.button("💥 Simulate provider update", type="primary", width="stretch", disabled=snap["incident_active"],
                     help="Re-embeds 10% of the live database with a different model, the way a silent provider update would."):
            state.inject_contamination(ctx, ctx.cfg.live_sentry.injection_batch_fraction, seed=None)
            st.rerun()
    with c2:
        if st.button("↺ Reset", width="stretch", help="Restore the live database to its clean state."):
            state.reset(ctx.benchmark)
            st.rerun()
    with c3:
        st.caption(f"Scenario: **{scenario}** · Live check every {snap['poll_interval_seconds']:.1f}s · Check #{snap['tick_count']:,}")

    # ---- 2. Status in plain English ----
    if snap["incident_active"] and status not in (STATUS_ALERT, STATUS_REMEDIATING):
        _banner("work", "🔎 Checking the live database...",
                "A provider update just changed part of the live database. VectorSentry's next check will measure the damage.",
                "Give it a few seconds.")
    elif status == STATUS_ALERT:
        _banner("bad", "🚨 Problem detected",
                f"**{live:,}** live documents look out of place, against a normal background of **{ref:,}**.",
                "Nothing to do yet. VectorSentry is starting the repair automatically.")
    elif status == STATUS_REMEDIATING:
        _banner("work", "🛠️ Repairing automatically",
                f"VectorSentry fixes the most important documents in batches and tests every batch on {n_golden} key search "
                f"queries before it goes live. **{excess:,}** extra suspicious documents remain.",
                "Sit back and watch the progress bar and chart below.")
    elif status == STATUS_NEEDS_REVIEW and snap["n_deferred"] == 0:
        _banner("warn", "✋ Almost there, paused for review",
                f"The live database is **{excess:,}** above normal, but every remaining suspicious document has already "
                f"been repaired once. Repairing them again would change nothing, so VectorSentry stopped instead of looping. "
                f"They are most likely ordinary false alarms.",
                "A human can confirm and close the incident, or press Reset and run another one.")
    elif status == STATUS_NEEDS_REVIEW:
        _banner("warn", "✋ Needs a human decision",
                f"Some fixes kept making search worse in testing, so VectorSentry **refused to apply them** and paused. "
                f"**{snap['n_deferred']:,}** documents are waiting for review, and the live database is still **{excess:,}** above normal.",
                "Open “Under the hood → Safety check” below to see why"
                + (", or press Reset and try the Smooth run scenario." if strict else ", or press Reset and run another incident."))
    elif status == STATUS_RESOLVED:
        fixed = summary["remediated"] if summary else snap["n_remediated"]
        _banner("good", "🎉 Back to normal",
                f"The live database is back to the normal background level. **{fixed:,}** documents were repaired and every batch passed the safety check.",
                "Press Simulate provider update again to run another incident.")
    else:
        _banner("good", "✅ All clear",
                "The live database matches the reference. Searches are returning the results they should.",
                "Press **💥 Simulate provider update** to start the demo.")

    _stepper(status)

    # ---- 3. Progress during an incident ----
    initial = snap["incident_initial_flagged_count"]
    if snap["incident_active"] and initial > ref:
        progress = 1.0 - excess / max(1, initial - ref)
        st.progress(float(np.clip(progress, 0.0, 1.0)), text=f"Cleanup progress: {np.clip(progress, 0, 1):.0%}  ({excess:,} extra suspicious documents left)")

    # ---- 4. Key numbers ----
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Suspicious in live database", f"{live:,}", f"{live - ref:+,} vs normal" if live != ref else None,
              delta_color="inverse", border=True, help="Documents the detector flags as out of place in the live database.")
    m2.metric("Normal background level", f"{ref:,}", border=True,
              help="The detector always flags a few documents in any database. The untouched reference shows how many is normal.")
    m3.metric("Repaired & verified", f"{snap['n_remediated']:,}", border=True,
              help="Documents whose fix passed the search-quality test and went live.")
    m4.metric("Waiting for a human", f"{snap['n_deferred']:,}", border=True,
              help="Documents whose fix kept failing the safety check, so they were not changed.")

    # ---- 5. Over time + activity ----
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Suspicious documents over time")
        _history_chart(snap["history"])
    with right:
        st.subheader("What just happened")
        _activity_feed(snap["event_log"])

    # ---- 6. Incident report ----
    if summary and status in (STATUS_RESOLVED, STATUS_NEEDS_REVIEW):
        with st.container(border=True):
            st.markdown("#### 📋 Incident report")
            # Ground truth is only available because this is a simulation: a
            # live document is still wrong if it differs from the reference.
            db_a, db_b = state.embedding_snapshot()
            still_wrong = int(np.sum(np.any(db_b != db_a, axis=1)))
            caught = max(0, summary["initial_flagged"] - ref)
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Changed by the update", f"{summary['injected']:,}",
                      help="Documents the simulated provider update re-embedded with a different model.")
            r2.metric("Caught (above normal)", f"{caught:,}",
                      f"{caught / max(1, summary['injected']):.0%} of changed", delta_color="off",
                      help=f"The detector flagged {summary['initial_flagged']:,} documents, but about {ref:,} of those are its "
                           f"normal background false alarms, which it raises on any database.")
            r3.metric("Repaired", f"{summary['remediated']:,}",
                      help="Documents re-embedded or patched after passing the safety check. Can exceed 'Caught': "
                           "each round re-scores the database and may flag a few more, and some repairs hit background "
                           "false alarms (re-embedding an unchanged document leaves it as it was).")
            r4.metric("Still wrong, undetected", f"{still_wrong:,}", delta_color="off",
                      help="Live documents that still differ from the reference. Only knowable here because this is a simulation.")
            st.caption(
                "**How detection works:** every stored vector is compared with a small sample of documents that were "
                "embedded by *both* models on purpose. A vector that sits among the new model's sample vectors was "
                "almost certainly written by the new model. "
                + (f"{still_wrong:,} live documents still differ from the reference: either the detector missed them, or "
                   f"they were patched (a patch is close to the original but not identical)." if still_wrong else
                   "Every live document matches the reference again.")
                + (f" {summary['deferred']:,} were left for human review." if summary["deferred"] else "")
            )

    # ---- 7. Under the hood ----
    st.subheader("🔍 Under the hood")
    tab_plan, tab_check, tab_map = st.tabs(["Fix plan", "Safety check", "Embedding map"])
    with tab_plan:
        _plan_table(snap)
    with tab_check:
        if snap["last_verification_before"] is None:
            st.caption("No safety check has run yet.")
        else:
            before, after = snap["last_verification_before"], snap["last_verification_after"]
            passed = snap["last_verification_passed"]
            st.markdown(
                f"Fixes go live in batches, most valuable first. Each batch is built in a **shadow copy**, and "
                f"{n_golden} popular logged searches are replayed against it: the score is how much of the results "
                f"those searches returned *before* the incident come back (0 to 1, higher is better). A batch ships "
                f"only if it is shown, with 95% confidence, to be no worse than the best state so far by more than "
                f"**{snap['demo_regression_tolerance'] if snap['demo_regression_tolerance'] is not None else ctx.cfg.migration.regression_tolerance:.2f}**. "
                f"A rejected batch is split in half and retried, and money saved on rejected fixes is spent on "
                f"full re-embeds instead."
            )
            k1, k2, k3 = st.columns(3)
            k1.metric("Results matching normal, before", f"{before:.3f}")
            k2.metric("Results matching normal, after", f"{after:.3f}", f"{after - before:+.3f}")
            k3.metric("Result", "✅ Applied" if passed else "❌ Rejected")
            budget = float(snap["current_budget_level"])
            if snap["incident_active"]:
                st.progress(min(1.0, budget), text=f"Current fix budget: {budget:.0%} (rises after a failed check, so the next plan can cover more documents)")
    with tab_map:
        if st.toggle("Show embedding map", help="Projects all document vectors onto 2D. Slower, so it is off by default."):
            _embedding_map()
        else:
            st.caption("Turn this on to see where the changed documents moved in the embedding space.")
    with st.expander("Raw engine log"):
        st.code("\n".join(f"{t}  {m}" for t, m in reversed(snap["event_log"][-50:])) or "(empty)", language=None)


live_panel()
