"""Fast, network-free tests for the web dashboard API (phase2_proposed/
web_app.py), served over a small synthetic index via Starlette's TestClient."""

import sys
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient
from test_live_sentry import _synthetic_context

from phase2.live_sentry import LiveSentryState, run_one_tick

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import web_app  # noqa: E402


@pytest.fixture()
def runtime():
    ctx = _synthetic_context(n=60)
    state = LiveSentryState(ctx.benchmark)
    layout = np.random.default_rng(0).random((len(ctx.doc_ids), 2))
    return web_app.Runtime(ctx, state, layout)


@pytest.fixture()
def client(runtime):
    return TestClient(web_app.create_app(runtime))


def test_index_page_and_static_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "VectorSentry" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/vendor/echarts.min.js").status_code == 200


def test_layout_has_one_point_and_size_per_document(client, runtime):
    body = client.get("/api/layout").json()
    n = len(runtime.ctx.doc_ids)
    assert len(body["xy"]) == len(body["size"]) == len(body["doc_ids"]) == n
    assert all(s > 0 for s in body["size"])


def test_state_reports_a_healthy_index_before_any_incident(client, runtime):
    run_one_tick(runtime.state, runtime.ctx)
    body = client.get("/api/state").json()
    assert body["status"] == "stable" and body["incident_active"] is False
    assert body["rounds"] == []
    assert set(body["doc_status"]) <= {web_app.HEALTHY, web_app.FLAGGED}


def test_simulate_then_repair_is_reflected_in_state_and_curve(client, runtime):
    injected = client.post("/api/simulate").json()["injected"]
    assert injected > 0
    assert client.get("/api/state").json()["incident_active"] is True

    for _ in range(6):
        run_one_tick(runtime.state, runtime.ctx)
    body = client.get("/api/state").json()
    assert body["rounds"], "a repair round should have been planned"
    assert web_app.REEMBEDDED in body["doc_status"] or web_app.PATCHED in body["doc_status"]
    assert any(e["kind"] == "incident" for e in body["events"])

    curve = client.get("/api/curve?round=1").json()
    assert len(curve["x"]) == len(curve["ours"]) == len(curve["fifo"]) > 0
    # The knapsack plan never protects less traffic than first-come-first-served.
    assert all(o >= f - 1e-6 for o, f in zip(curve["ours"], curve["fifo"]))
    assert curve["ours"][-1] == pytest.approx(100, abs=1)


def test_config_validates_input(client, runtime):
    assert client.post("/api/config", json={"budget": 2}).status_code == 400
    assert client.post("/api/config", content=b"not json").status_code == 400
    ok = client.post("/api/config", json={"budget": 0.1, "safety": "strict"}).json()
    assert ok == {"budget": 0.1, "tolerance": 0.0}
    assert runtime.state.snapshot()["demo_initial_budget_level"] == pytest.approx(0.1)


def test_curve_rejects_a_bad_round_and_handles_no_incident(client):
    assert client.get("/api/curve?round=x").status_code == 400
    assert client.get("/api/curve?round=1").json() == {"x": [], "ours": [], "fifo": []}


def test_reset_restores_a_clean_index(client, runtime):
    client.post("/api/simulate")
    client.post("/api/reset")
    state = runtime.state
    assert np.array_equal(state.db_b_embeddings, runtime.ctx.benchmark.emb_a)
    assert client.get("/api/state").json()["incident_active"] is False


def test_friendly_event_hides_internal_chatter():
    assert web_app.friendly_event("Tick 3: idle") is None
    kind, text = web_app.friendly_event("Simulated embedding provider update: 500 documents in Database B")
    assert kind == "incident" and "500" in text
