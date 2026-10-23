"""Phase 2 showcase -- the autonomous "Live Sentry" demo (sub-phase 2D).

Two databases: Database A ("Stable") is an untouched control, never
modified. Database B ("Live") is where you can simulate a real incident.
The ONE manual control on this page is "Simulate embedding provider
update" -- everything that happens after that is fully autonomous, driven
by a real background thread: detection fires on its own, an alert and
live event log appear on their own, Phase 2's scheduler (2A) and
migrate+verify loop (2B) run and log their plan/progress on their own, and
the alert clears itself once verification confirms Database B is back to
its baseline noise floor.

Run with:
    streamlit run phase2_proposed/live_sentry_dashboard.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in ("common/src", "phase1_baseline/src", "phase2_proposed/src"):
    sys.path.insert(0, str(_REPO_ROOT / _src))

import streamlit as st  # noqa: E402

from common.config import load_config  # noqa: E402
from common.seed import set_seed  # noqa: E402
from phase2.live_sentry import (  # noqa: E402
    STATUS_ALERT,
    STATUS_REMEDIATING,
    STATUS_RESOLVED,
    STATUS_STABLE,
    LiveSentryState,
    build_context,
    monitor_loop,
)

CONFIG_PATH = "phase2_proposed/configs/default.yaml"
UI_REFRESH_SECONDS = 1.5  # how often the PAGE redraws from the shared state; independent of the monitor's own poll interval

st.set_page_config(page_title="VectorSentry -- Live Sentry", layout="wide")


@st.cache_resource(show_spinner="Loading benchmark and building Live Sentry context (cached after first run)...")
def _load_context():
    cfg = load_config(CONFIG_PATH)
    set_seed(cfg.seed)
    return build_context(cfg)


ctx = _load_context()

if "live_sentry_state" not in st.session_state:
    st.session_state.live_sentry_state = LiveSentryState(ctx.benchmark, ctx.cfg.live_sentry.poll_interval_seconds)

if "live_sentry_thread_started" not in st.session_state:
    thread = threading.Thread(
        target=monitor_loop, args=(st.session_state.live_sentry_state, ctx), daemon=True, name="live-sentry-monitor"
    )
    thread.start()
    st.session_state.live_sentry_thread_started = True

state: LiveSentryState = st.session_state.live_sentry_state

st.title("VectorSentry -- Live Sentry")
st.caption(
    "Database A is an untouched control. Database B is where you can simulate a real incident. "
    "Everything past the one button below runs on its own: detection, alerting, prioritized scheduling, "
    "and a verified migration -- driven by a real background thread, not a scripted animation."
)

# --- Sidebar: the ONE manual control, plus display-only tuning ---
st.sidebar.header("Controls")
if st.sidebar.button("Simulate embedding provider update", type="primary"):
    n_injected = state.inject_contamination(ctx, ctx.cfg.live_sentry.injection_batch_fraction, seed=None)
    st.sidebar.success(f"Injected {n_injected} contaminated documents into Database B.")

poll_interval = st.sidebar.slider("Monitor poll interval (seconds)", 2.0, 5.0, ctx.cfg.live_sentry.poll_interval_seconds, step=0.5)
state.set_poll_interval(poll_interval)

if st.sidebar.button("Reset demo"):
    state.reset(ctx.benchmark)

st.sidebar.caption(
    f"Alert threshold: {ctx.cfg.live_sentry.alert_threshold_percentile:.0f}th percentile of Database A's "
    f"own KNN-OOD score distribution (calibrated fresh each tick). Initial remediation budget: "
    f"{ctx.cfg.live_sentry.initial_budget_level:.2f}, escalating by "
    f"{ctx.cfg.live_sentry.budget_escalation_step:.2f} on each verification failure."
)

snap = state.snapshot()

STATUS_BADGES = {
    STATUS_STABLE: ("🟢", "Stable"),
    STATUS_ALERT: ("🔴", "Alert"),
    STATUS_REMEDIATING: ("🟡", "Remediating"),
    STATUS_RESOLVED: ("🟢", "Resolved"),
}

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Database A -- Stable")
    st.metric("Flagged documents", snap["last_db_a_flagged"])
    st.caption("Untouched control. Any nonzero count here is the detector's own baseline noise floor.")

with col_b:
    st.subheader("Database B -- Live")
    icon, label = STATUS_BADGES[snap["status"]]
    st.markdown(f"### {icon} {label}")
    st.metric("Flagged documents", snap["last_db_b_flagged"])
    if snap["incident_active"]:
        st.caption(f"Current remediation budget level: {snap['current_budget_level']:.2f}")
    if snap["n_deferred"] > 0:
        st.warning(
            f"{snap['n_deferred']} document(s) deferred for manual review -- verification repeatedly "
            f"blocked their fix even at maximum budget."
        )

st.divider()

st.subheader("Live event log")
st.caption(f"Tick {snap['tick_count']} -- polling every {snap['poll_interval_seconds']:.1f}s")
log_text = "\n".join(f"[{ts}] {msg}" for ts, msg in reversed(snap["event_log"][-40:]))
st.text_area("Event log (most recent first)", log_text, height=320, label_visibility="collapsed")

# Lightweight auto-refresh: redraws the page from the shared state on a
# timer, independent of how often the background thread actually ticks.
time.sleep(UI_REFRESH_SECONDS)
st.rerun()
