"""VectorSentry Live Sentry -- web dashboard server.

A small Starlette app around the SAME engine the Streamlit dashboard uses
(phase2.live_sentry): one shared LiveSentryState, one background monitor
thread, and a JSON API that a custom front end (phase2_proposed/web/) polls.
No detection, scheduling or migration logic lives here -- only translation
of the engine's state into what the charts draw.

Run:
    .venv\\Scripts\\python.exe phase2_proposed/web_app.py          # http://localhost:8600
    .venv\\Scripts\\python.exe phase2_proposed/web_app.py --port 9000
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in ("common/src", "phase1_baseline/src", "phase2_proposed/src"):
    sys.path.insert(0, str(_REPO_ROOT / _src))

import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import FileResponse, JSONResponse  # noqa: E402
from starlette.routing import Mount, Route  # noqa: E402
from starlette.staticfiles import StaticFiles  # noqa: E402

from common.benchmark import _embedding_dir  # noqa: E402
from common.config import load_config  # noqa: E402
from common.seed import set_seed  # noqa: E402
from phase2.live_sentry import LiveSentryState, build_context, monitor_loop  # noqa: E402
from phase2.pipeline import plan_remediation  # noqa: E402

CONFIG_PATH = _REPO_ROOT / "phase2_proposed" / "configs" / "default.yaml"
WEB_DIR = Path(__file__).resolve().parent / "web"

# Per-document status codes for the index map.
HEALTHY, AFFECTED, FLAGGED, REEMBEDDED, PATCHED = 0, 1, 2, 3, 4

print("Loading the benchmark and preparing both databases...", flush=True)
cfg = load_config(CONFIG_PATH)
set_seed(cfg.seed)
ctx = build_context(cfg)
state = LiveSentryState(ctx.benchmark, cfg.live_sentry.poll_interval_seconds)
state.configure_demo(0.3, 0.01)
threading.Thread(target=monitor_loop, args=(state, ctx), daemon=True, name="live-sentry-monitor").start()


def _map_layout() -> np.ndarray:
    """2-D positions for every document: t-SNE of the clean (generation-A)
    embeddings, computed once and cached next to them. Positions are FIXED,
    so the map shows documents changing state, not moving."""
    path = _embedding_dir(cfg) / "tsne_a.npy"
    if path.exists():
        return np.load(path)
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    print("Computing the 2-D index map layout (one time, ~1 minute)...", flush=True)
    reduced = PCA(n_components=50, random_state=cfg.seed).fit_transform(ctx.benchmark.emb_a)
    xy = TSNE(n_components=2, init="pca", perplexity=40, random_state=cfg.seed).fit_transform(reduced)
    xy = (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0))
    np.save(path, xy.astype(np.float32))
    return xy


LAYOUT = _map_layout()
TRAFFIC = ctx.phase2.traffic
TRAFFIC_SHARE = TRAFFIC / TRAFFIC.sum()
# Marker size grows with search traffic (sqrt so the busiest do not swamp the map).
MARKER_SIZE = np.round(2.2 + 16 * np.sqrt(TRAFFIC_SHARE / TRAFFIC_SHARE.max()), 1)
print("Ready.", flush=True)


# --------------------------------------------------------------------------
# Plain-language event feed
# --------------------------------------------------------------------------

_EVENT_RULES = [
    (r"Live Sentry started", "start", lambda m: "Monitoring started."),
    (r"Simulated embedding provider update: (\d+)", "incident",
     lambda m: f"Provider update: <b>{int(m[1]):,}</b> documents re-embedded by a different model."),
    (r"no clean documents remain", "info", lambda m: "Nothing left to break. Reset to start over."),
    (r"ALERT: .*\((\d+) documents flagged\)", "alert", lambda m: f"Detected <b>{int(m[1]):,}</b> out-of-place documents."),
    (r"building prioritized schedule for (\d+)", "plan", lambda m: f"Ranking {int(m[1]):,} suspicious documents by search traffic."),
    (r"Plan: (\d+) full re-embeds, (\d+) patches", "plan",
     lambda m: f"Plan: re-embed <b>{int(m[1]):,}</b>, quick-patch <b>{int(m[2]):,}</b>."),
    (r"Verification PASSED for (\d+) of (\d+) rollout wave\(s\) \(canary ([\d.]+) -> ([\d.]+)\)", "pass",
     lambda m: f"{m[1]} of {m[2]} batches passed the safety check. Search health <b>{float(m[4]):.1%}</b>."),
    (r"Verification FAILED", "fail", lambda m: "Safety check blocked every batch; retrying with a bigger budget."),
    (r"Verification failed \d+ times in a row.*\((\d+) documents\)", "fail",
     lambda m: f"<b>{int(m[1]):,}</b> documents set aside for human review."),
    (r"Detector nominal", "good", lambda m: "Incident resolved: detector back to its normal background."),
    (r"Clean-up: upgraded (\d+) patched documents.*canary ([\d.]+) -> ([\d.]+)\)", "polish",
     lambda m: f"Clean-up: <b>{int(m[1]):,}</b> patches replaced by exact re-embeds. Search health <b>{float(m[3]):.1%}</b>."),
    (r"Automation paused", "fail", lambda m: "Paused: the rest needs a human decision."),
    (r"Reset:", "info", lambda m: "Demo reset."),
]


def _friendly(message: str) -> tuple[str, str] | None:
    for pattern, kind, render in _EVENT_RULES:
        match = re.search(pattern, message)
        if match:
            return kind, render(match)
    return None  # internal chatter (e.g. "Tick 3: ...") stays out of the feed


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

def _doc_status() -> list[int]:
    with state._lock:
        db_a, db_b = state.db_a_embeddings, state.db_b_embeddings
        differs = np.any(db_b != db_a, axis=1)
        status = np.full(len(db_a), HEALTHY, dtype=np.int8)
        status[differs] = AFFECTED
        status[state.last_flagged_b & ~state.db_b_remediated_once] = FLAGGED
        status[state.db_b_remediated_once] = REEMBEDDED
        status[state.db_b_patched] = PATCHED
    return status.tolist()


def _round_payload(rnd: dict) -> dict:
    keep = ("doc_index", "action", "severity", "traffic_vs_average", "traffic_share", "rollout_rank")
    rows = [{k: r[k] for k in keep} for r in rnd["rows"]]
    return {"budget": rnd["budget"], "rows": rows, "waves": rnd["waves"]}


async def api_state(request: Request) -> JSONResponse:
    snap = state.snapshot()
    events = []
    for timestamp, message in snap["event_log"][-40:]:
        friendly = _friendly(message)
        if friendly:
            events.append({"time": timestamp, "kind": friendly[0], "text": friendly[1]})
    history = [
        {
            "t": int(h["time"].timestamp() * 1000),
            "health": h.get("health"),
            "flagged_live": h["db_b_flagged"],
            "flagged_ref": h["db_a_flagged"],
            "status": h["status"],
        }
        for h in snap["history"]
    ]
    return JSONResponse({
        "status": snap["status"],
        "incident_active": snap["incident_active"],
        "tick": snap["tick_count"],
        "health": snap["last_health"],
        "flagged_live": snap["last_db_b_flagged"],
        "flagged_ref": snap["last_db_a_flagged"],
        "n_remediated": snap["n_remediated"],
        "n_patched": snap["n_patched"],
        "n_deferred": snap["n_deferred"],
        "budget_spent": snap["incident_budget_spent"],
        "budget_level": snap["demo_initial_budget_level"],
        "tolerance": snap["demo_regression_tolerance"],
        "injected": snap["incident_injected_count"],
        "summary": snap["last_incident_summary"],
        "history": history,
        "events": events[::-1][:14],
        "rounds": [_round_payload(r) for r in snap["incident_rounds"]],
        "doc_status": _doc_status(),
        "canary_size": ctx.canary_size,
        "full_cost": cfg.remediation.cost_full_reembed,
        "patch_cost": cfg.remediation.cost_patch,
    })


async def api_layout(request: Request) -> JSONResponse:
    return JSONResponse({
        "xy": np.round(LAYOUT, 4).tolist(),
        "size": MARKER_SIZE.tolist(),
        "traffic_vs_average": np.round(TRAFFIC / TRAFFIC.mean(), 3).tolist(),
        "doc_ids": ctx.doc_ids,
    })


@lru_cache(maxsize=32)
def _curves(doc_index: tuple[int, ...], p: tuple[float, ...]) -> dict:
    """Expected share of affected search traffic protected vs. budget:
    VectorSentry's MCKP plan vs. first-come-first-served re-embedding."""
    idx, prob = np.array(doc_index), np.array(p)
    affected = prob * TRAFFIC[idx]
    total = affected.sum()
    full = cfg.remediation.cost_full_reembed
    budgets = np.linspace(0, len(idx) * full, 51)
    ours = [plan_remediation(cfg, ctx.phase2, idx, prob, b).plan.total_benefit / total for b in budgets]
    fifo_cum = np.r_[0.0, np.cumsum(affected[np.argsort(idx)])] / total
    fifo = [fifo_cum[min(len(idx), int(b // full))] for b in budgets]
    return {
        "x": np.round(budgets / (len(idx) * full) * 100, 2).tolist(),
        "ours": np.round(np.clip(ours, 0, 1) * 100, 2).tolist(),
        "fifo": np.round(np.array(fifo) * 100, 2).tolist(),
    }


async def api_curve(request: Request) -> JSONResponse:
    snap = state.snapshot()
    rounds = snap["incident_rounds"]
    k = int(request.query_params.get("round", 1)) - 1
    if not rounds or not 0 <= k < len(rounds):
        return JSONResponse({"x": [], "ours": [], "fifo": []})
    rows = rounds[k]["rows"]
    return JSONResponse(_curves(tuple(r["doc_index"] for r in rows), tuple(round(r["severity"], 6) for r in rows)))


async def api_simulate(request: Request) -> JSONResponse:
    n = state.inject_contamination(ctx, cfg.live_sentry.injection_batch_fraction, seed=None)
    return JSONResponse({"injected": n})


async def api_reset(request: Request) -> JSONResponse:
    state.reset(ctx.benchmark)
    return JSONResponse({"ok": True})


async def api_config(request: Request) -> JSONResponse:
    body = await request.json()
    budget = float(body.get("budget", 0.3))
    tolerance = 0.0 if body.get("safety") == "strict" else 0.01
    state.configure_demo(budget, tolerance)
    return JSONResponse({"budget": budget, "tolerance": tolerance})


async def index(request: Request) -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app = Starlette(routes=[
    Route("/", index),
    Route("/api/state", api_state),
    Route("/api/layout", api_layout),
    Route("/api/curve", api_curve),
    Route("/api/simulate", api_simulate, methods=["POST"]),
    Route("/api/reset", api_reset, methods=["POST"]),
    Route("/api/config", api_config, methods=["POST"]),
    Mount("/static", StaticFiles(directory=WEB_DIR), name="static"),
])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"VectorSentry Live Sentry: http://localhost:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
