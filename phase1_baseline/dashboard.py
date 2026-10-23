"""Phase 1 dashboard -- demonstrates the existing methodology's detect-then-
remediate loop, and its central limitation: FIFO remediation fixes flagged
documents in ingestion order only, with no notion of which ones matter most.
A high-traffic document can stay broken purely because of where it happened
to land in ingestion order, not because it wasn't worth fixing. That visual
gap is the setup for Phase 2's contribution (a budget-aware prioritization
scheduler) -- see DEVLOG.md's sub-phase 1D/1E entries for the full write-up.

Run with:
    streamlit run phase1_baseline/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in ("common/src", "phase1_baseline/src"):
    sys.path.insert(0, str(_REPO_ROOT / _src))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common.benchmark import build_benchmark, build_contaminated_ratio, mixed_embeddings  # noqa: E402
from common.config import load_config  # noqa: E402
from common.embeddings import embed_texts  # noqa: E402
from common.index import build_index, search  # noqa: E402
from common.metrics import evaluate_retrieval  # noqa: E402
from common.patch import apply_patch, fit_patch_transform, sample_calibration_indices  # noqa: E402
from common.seed import set_seed  # noqa: E402
from phase1.detection import DetectionResult, detect_knn_ood, evaluate_detection  # noqa: E402
from phase1.remediation import fifo_remediation_plan  # noqa: E402

CONFIG_PATH = "phase1_baseline/configs/default.yaml"

st.set_page_config(page_title="VectorSentry -- Phase 1 Baseline", layout="wide")


# --- Expensive, ratio-independent setup: loaded once per server process ---


@st.cache_resource(show_spinner="Loading benchmark (cached after first run)...")
def load_benchmark():
    cfg = load_config(CONFIG_PATH)
    set_seed(cfg.seed)
    benchmark = build_benchmark(cfg)
    return cfg, benchmark


@st.cache_resource(show_spinner="Fitting patch transform...")
def load_patched_embeddings(_benchmark, seed: int, calibration_sample_size: int):
    calibration_indices = sample_calibration_indices(len(_benchmark.documents), calibration_sample_size, seed)
    transform = fit_patch_transform(_benchmark.emb_a, _benchmark.emb_b, calibration_indices)
    return apply_patch(_benchmark.emb_b, transform)


@st.cache_resource(show_spinner="Embedding eval queries...")
def load_eval_queries(_benchmark, generation_a_model: str):
    eval_queries = [q for q in _benchmark.queries if q.query_id in _benchmark.qrels_eval]
    query_embeddings = embed_texts([q.text for q in eval_queries], generation_a_model)
    return eval_queries, query_embeddings


cfg, benchmark = load_benchmark()
patched_emb_b = load_patched_embeddings(benchmark, cfg.seed, cfg.remediation.calibration_sample_size)
eval_queries, query_embeddings = load_eval_queries(benchmark, cfg.embedding_models.generation_a)
doc_ids = [d.doc_id for d in benchmark.documents]
doc_by_id = {d.doc_id: d for d in benchmark.documents}


# --- Ratio-dependent state: cached per ratio (Streamlit reruns the whole
# script on every widget interaction, so this avoids re-detecting on every
# budget-slider tick) ---


@st.cache_data(show_spinner="Building contaminated index and running detection...")
def compute_ratio_state(ratio: float):
    contaminated = build_contaminated_ratio(cfg, benchmark, ratio)
    mixed = mixed_embeddings(benchmark, contaminated)
    detection_result = detect_knn_ood(mixed, ratio=ratio, k=cfg.detection.knn_k)
    return contaminated.mask, mixed, detection_result.predicted


@st.cache_data(show_spinner="Sweeping budget levels for both strategies...")
def compute_budget_sweep(ratio: float) -> pd.DataFrame:
    mask, mixed, flagged = compute_ratio_state(ratio)
    n_flagged = int(flagged.sum())
    rows = []
    for budget_level in cfg.budget.levels:
        budget_units = budget_level * n_flagged * cfg.remediation.cost_full_reembed

        fixed_full, _ = fifo_remediation_plan(flagged, budget_units, cfg.remediation.cost_full_reembed)
        emb_full = mixed.copy()
        emb_full[fixed_full] = benchmark.emb_a[fixed_full]
        metrics_full = evaluate_retrieval(
            emb_full, doc_ids, eval_queries, query_embeddings, benchmark.qrels_eval, benchmark.query_traffic
        )

        fixed_patch, _ = fifo_remediation_plan(flagged, budget_units, cfg.remediation.cost_patch)
        emb_patch = mixed.copy()
        emb_patch[fixed_patch] = patched_emb_b[fixed_patch]
        metrics_patch = evaluate_retrieval(
            emb_patch, doc_ids, eval_queries, query_embeddings, benchmark.qrels_eval, benchmark.query_traffic
        )

        rows.append({"budget": budget_level, "strategy": "FIFO full re-embed", "ndcg@10": metrics_full["ndcg@10"]})
        rows.append({"budget": budget_level, "strategy": "Uniform patch", "ndcg@10": metrics_patch["ndcg@10"]})
    return pd.DataFrame(rows)


# --- Page ---

st.title("VectorSentry -- Phase 1: Existing Methodology (Baseline)")
st.caption(
    "Detect mixed-generation embedding contamination, then remediate under a fixed budget "
    "with **no prioritization** -- documents get fixed strictly in ingestion order. This is "
    "what current practice looks like. The gap this leaves is what Phase 2 fills."
)

st.sidebar.header("Scenario controls")
ratio = st.sidebar.select_slider(
    "Contamination ratio", options=cfg.contamination.ratios, value=cfg.contamination.ratios[2]
)
budget_level = st.sidebar.slider(
    "Remediation budget (fraction of flagged docs' full-reembed cost)", 0.0, 1.0, 0.2, step=0.05
)
strategy = st.sidebar.radio("Remediation strategy", ["FIFO full re-embed", "Uniform patch"])
st.sidebar.caption(
    f"Detector: KNN-OOD (k={cfg.detection.knn_k}). Assumes the operator knows the approximate "
    "contamination ratio, per sklearn's own `contamination=` convention -- see DEVLOG.md."
)

mask, mixed, flagged = compute_ratio_state(ratio)
n_contaminated = int(mask.sum())
n_flagged = int(flagged.sum())
detection_metrics = evaluate_detection(
    DetectionResult("knn_ood", scores=np.zeros(len(mask)), predicted=flagged), mask
)

budget_units = budget_level * n_flagged * cfg.remediation.cost_full_reembed
cost_per_fix = cfg.remediation.cost_full_reembed if strategy == "FIFO full re-embed" else cfg.remediation.cost_patch
fixed, cost_spent = fifo_remediation_plan(flagged, budget_units, cost_per_fix)

remediated = mixed.copy()
if strategy == "FIFO full re-embed":
    remediated[fixed] = benchmark.emb_a[fixed]
else:
    remediated[fixed] = patched_emb_b[fixed]

# --- Key numbers ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Documents", f"{len(benchmark.documents):,}")
col2.metric("Contaminated (ground truth)", f"{n_contaminated:,}", f"{n_contaminated / len(mask):.1%} of corpus")
col3.metric("Flagged by detector", f"{n_flagged:,}", f"recall {detection_metrics['recall']:.0%}")
col4.metric("Fixed under this budget", f"{int(fixed.sum()):,}", f"of {n_flagged:,} flagged")
col5.metric("Cost spent", f"{cost_spent:.1f}", f"budget {budget_units:.1f} units")

with st.expander("Detector quality at this ratio (precision / recall / F1)"):
    st.write(
        f"Precision **{detection_metrics['precision']:.3f}** &nbsp;|&nbsp; "
        f"Recall **{detection_metrics['recall']:.3f}** &nbsp;|&nbsp; "
        f"F1 **{detection_metrics['f1']:.3f}**"
    )
    st.caption(
        "Recall < 1.0 means some truly contaminated documents are never flagged at all -- "
        "no budget, however large, can fix a document remediation never learns about."
    )

st.divider()

# --- The central visual: which high-traffic documents stay broken, and why ---
st.subheader("Which high-traffic documents are still broken?")
st.write(
    "FIFO fixes flagged documents in the order they were ingested -- **not** by how much "
    "search traffic depends on them. The table below lists the highest-traffic documents "
    "that are contaminated in this scenario. Whether one gets fixed is down to where it "
    "happened to land in ingestion order, not whether it matters."
)

contaminated_idx = np.flatnonzero(mask)
order = contaminated_idx[np.argsort(-benchmark.doc_traffic[contaminated_idx])]
top_rows = []
for i in order[:15]:
    doc = benchmark.documents[i]
    top_rows.append(
        {
            "doc_id": doc.doc_id,
            "title": doc.title[:90] if doc.title else doc.text[:90],
            "traffic_share": float(benchmark.doc_traffic[i]),
            "flagged_by_detector": bool(flagged[i]),
            "fixed_under_budget": bool(fixed[i]),
        }
    )
top_df = pd.DataFrame(top_rows)


def _highlight_broken(row):
    still_broken = row["flagged_by_detector"] and not row["fixed_under_budget"]
    return ["background-color: #ffe0e0" if still_broken else "" for _ in row]


if len(top_df) > 0:
    st.dataframe(top_df.style.apply(_highlight_broken, axis=1), use_container_width=True, hide_index=True)
else:
    st.info("No contaminated documents carry any measured traffic at this ratio.")

st.divider()

# --- Quality vs budget, both strategies, current selection marked ---
st.subheader("Quality vs. budget -- both baseline strategies")
sweep_df = compute_budget_sweep(ratio)
fig, ax = plt.subplots(figsize=(9, 4))
for strat_name, group in sweep_df.groupby("strategy"):
    group = group.sort_values("budget")
    ax.plot(group["budget"], group["ndcg@10"], marker="o", label=strat_name)
ax.axvline(budget_level, color="red", linestyle="--", label="current selection")
ax.set_xlabel("Budget level")
ax.set_ylabel("nDCG@10 (unweighted)")
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

st.divider()

# --- Before / after search example ---
st.subheader("Before / after: one query's search results")

doc_id_to_eval_queries: dict[str, list] = {}
for q in eval_queries:
    for did in benchmark.qrels_eval[q.query_id]:
        doc_id_to_eval_queries.setdefault(did, []).append(q)

default_query_idx = 0
for i in order:
    if not fixed[i]:
        did = benchmark.documents[i].doc_id
        if did in doc_id_to_eval_queries:
            target_query = doc_id_to_eval_queries[did][0]
            default_query_idx = eval_queries.index(target_query)
            break

query_labels = [f"{q.text[:90]}  (traffic={benchmark.query_traffic.get(q.query_id, 0):.4f})" for q in eval_queries]
selected = st.selectbox(
    "Query (defaults to one whose relevant document is still broken, if any)",
    options=range(len(eval_queries)),
    index=default_query_idx,
    format_func=lambda i: query_labels[i],
)

query = eval_queries[selected]
query_embedding = query_embeddings[selected : selected + 1]
relevant_ids = benchmark.qrels_eval[query.query_id]

before_index = build_index(mixed)
after_index = build_index(remediated)
_, before_top10 = search(before_index, query_embedding, 10)
_, after_top10 = search(after_index, query_embedding, 10)


def _render_results(col, label, indices):
    col.markdown(f"**{label}**")
    for rank, j in enumerate(indices):
        doc_id = doc_ids[j]
        marker = "✅ " if doc_id in relevant_ids else ""
        title = doc_by_id[doc_id].title or doc_by_id[doc_id].text
        col.write(f"{rank + 1}. {marker}{title[:80]}")


col_before, col_after = st.columns(2)
_render_results(col_before, "Before remediation", before_top10[0])
_render_results(col_after, f"After remediation ({strategy}, budget={budget_level:.2f})", after_top10[0])

st.caption("✅ marks a document actually relevant to this query, per BEIR's SciFact qrels.")
