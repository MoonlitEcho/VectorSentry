# VectorSentry — Project Handoff Document

**Purpose:** This document lets any engineer or AI coding agent pick up the VectorSentry project cold and continue it correctly. It states what the project is, what exists, what is verified versus unverified, and exactly what remains, with acceptance criteria for each piece.

**Repository:** https://github.com/MoonlitEcho/VectorSentry (branch `master`, public, MIT license, 3 commits at time of writing)

**How this document was produced (read this first):** It was compiled from the repository's `README.md`, `TODO.md`, and `WORK_PLAN.md`, plus the project history discussed with the owner. The source files under `src/`, the `config/default.yaml` contents, and `.agents.md` were **not** directly inspected. Anything stated about the internals of those files is inferred from the three documents above and is marked **[VERIFY]**. Before changing code, an agent should read the actual source and treat this document as a map, not ground truth.

---

## 1. Project in one paragraph

Production RAG / vector-search systems degrade silently when one vector database holds embeddings from two different embedding-model "generations" (for example, some documents embedded with Model A and later ones with Model B after a provider upgrade). Cosine similarity is not comparable across generations, so retrieval quality decays with no error. Prior work either **detects** this (LOF, Isolation Forest, KNN-OOD, Drift-Adapter-style transforms) or **moves data safely** (full re-index, shadow/blue-green indexing, dual-write backfill). Nothing decides, **under a real budget, what to fix first**. VectorSentry adds that decision layer.

**The core contribution** is formulating remediation as a **Multiple-Choice Knapsack Problem (MCKP)**: for each contaminated document pick exactly one action (`full_reembed`, `patch`, or `defer`) to maximize recovered retrieval quality subject to a total cost budget.

## 2. Context an implementer must know

| Item | Detail |
|---|---|
| Nature of project | Final-year engineering capstone (UG Capstone Projects 2026-27), also intended as the basis of a research paper |
| Institution template | Kumaraguru College of Technology / Kumaraguru School of Innovation (KSI) |
| Review structure | **Phase I** = implement and demonstrate the *existing* methodology (baselines). **Phase II** = implement and demonstrate *our* methodology (the prioritization layer) with a head-to-head comparison |
| Review I rubric | Scored on Research, Innovation, Design (50 marks each). Research and Design use identical criteria: Problem Identification, Objectives, Engineering Applications, Literature (scholarly sources only, with an explicit gap), Methodology. Innovation: Concept, Novelty, Creativity, Real World Problems, Technical Feasibility |
| Paperwork | Formal reports are written later against rubrics supplied about one week before each review. **Do not write formal academic reports unprompted.** Keep a running dev log instead |
| Owner's expectation | Working code first; the owner must be able to explain every design decision to examiners, so code needs thorough comments and clear docs |

### Core scientific claims (do not overclaim)

1. A contaminated index is better modeled as a mixture of two populations than as "outliers in one population". Existing detectors assume a single population (LOF, Isolation Forest) or a clean reference set (KNN-OOD).
2. Remediation priority should combine **contamination severity** with **real query traffic**, under a **budget**, solved as an MCKP.
3. Evidence is from a **simulated** benchmark. Conclusions must distinguish synthetic evidence from claims about production systems (see Section 9).

### Literature base (scholarly only; keep citations accurate)

Breunig et al. (2000) LOF, ACM SIGMOD. Liu, Ting & Zhou (2008) Isolation Forest, IEEE ICDM. Sun, Ming, Zhu & Li (2022) KNN-OOD, ICML (PMLR 162). Sinha & Zoltners (1979) Multiple-Choice Knapsack Problem, Operations Research 27(3). Vejendla (2025) Drift-Adapter, arXiv:2509.23471. Cao et al. (2025) TAD-Bench, arXiv:2501.11960. Xiao et al. (2025) Text-ADBench, arXiv:2507.12295. Thakur et al. (2021) BEIR, NeurIPS Datasets & Benchmarks. Hu et al. (2022) Learning Backward Compatible Embeddings, KDD. Meng et al. (2021) Learning Compatible Embeddings, ICCV.

---

## 3. System architecture

Four stages, each corresponding to a package under `src/vectorsentry/`:

```
Vector store (mixed generations)
   │
   ▼
1. DETECT       detection/       per-document contamination severity, unsupervised
   │
   ▼
2. PRIORITIZE   prioritization/  MCKP: severity × traffic × recovery factor, under budget
   │                              (exact PuLP solver + greedy fallback)
   ▼
3. MIGRATE      migration/       shadow index, patch transform, gated alias swap
   │
   ▼
4. VERIFY       (in migration/)  golden-query nDCG@k / Recall@k gate before promotion
```

### Repository layout (from README)

```
config/default.yaml        every experiment knob: ratios, budgets, model names, seeds
src/vectorsentry/
  config.py                loads and validates config into typed Settings
  benchmark/               corpus, embeddings, contamination, traffic simulation, ANN index
  detection/               GMM neighborhood scorer + LOF / IsolationForest / KNN baselines
  prioritization/          MCKP scheduler (exact PuLP + greedy)
  migration/               shadow index, patch transform, quality gate, plan execution
  dashboard/               Streamlit app
  utils/                   RNG seeding etc.
tests/                     pytest suite mirroring src/
data/raw|processed|results gitignored; regenerated by scripts
scripts/                   thin CLI wrappers, one per experiment step
notebooks/                 exploratory only, not part of reproducible pipeline
.agents.md                 agent progress/decision log [VERIFY contents]
TODO.md, WORK_PLAN.md      execution checklist and scope/acceptance criteria
REPRODUCE.md               exact commands (exists; completeness [VERIFY])
```

### Technology choices (all free, open source, CPU-only, no API keys)

`sentence-transformers` (Generation A = `all-MiniLM-L6-v2`, Generation B = `bge-small-en-v1.5`), `beir` (SciFact corpus with real queries and relevance judgments), `faiss-cpu` (`IndexFlatIP`, exact search), `scikit-learn` (GMM, LOF, Isolation Forest), `pulp` (MILP with bundled CBC), `matplotlib`/`plotly`, `streamlit`, `pytest`. Python 3.11+.

### Hard design constraints (from WORK_PLAN.md; do not violate)

1. **Reproducibility:** all randomness uses explicit seeded RNGs; every run records its config and metadata.
2. **No model/data downloads in unit tests.** Use small synthetic vectors for fast deterministic tests.
3. **Modules independently callable;** scripts are thin CLI wrappers over package code.
4. **Thresholds and experiment knobs live in YAML,** validated in `config.py`. Nothing hardcoded that belongs in config.
5. **Artifact schemas first.** Store document IDs alongside positional arrays, or validate their shared ordering.
6. **Meaningful baselines and honest limitations.** Simulated contamination is not evidence about every production vector DB.
7. **No label leakage:** ground-truth contamination labels are used only for evaluation, never for detector fitting or features.

---

## 4. Current status, phase by phase

Status below follows the README status table, which is the most recent source. See Section 5 for a documented inconsistency between repo documents.

| Phase | Area | Status per README | What exists | What is NOT yet done |
|---|---|---|---|---|
| 1 | Scaffolding | Done | Packaging (`pyproject.toml`, `requirements.txt`), typed config with `Settings.validate()`, seeding utils, tests for config/seeding | none |
| 2 | Synthetic benchmark | Implemented; run/validation pending | Code that builds a SciFact corpus, embeds with Generation A and B, assigns contamination labels at configured ratios, simulates Zipfian query traffic, writes FAISS indexes; benchmark inspection; tests for contamination assignment and traffic simulation | Benchmark has **not been confirmed run end to end**; artifacts are gitignored; artifact schemas and document-ordering invariants not validated |
| 3 | Detection | Scoring foundation implemented; evaluation pending | Scoring API taking a row-aligned embedding matrix plus doc-ID list, returning percentile scores in [0,1], higher = more suspicious; GMM detector using nearest-neighbor cosine-distance summaries; LOF, Isolation Forest, k-NN distance baselines; `scripts/evaluate_detection.py` writing `data/results/<dataset>/detection_metrics.json` | Detector **validation** and the multi-ratio, multi-seed comparison; unit fixtures for degenerate cases [VERIFY coverage]; runtime/stability reporting |
| 4 | Prioritization | Solvers implemented; evaluation pending | `prioritize(...)`: exact PuLP solver up to a configured instance size, deterministic greedy marginal-recovery-per-cost heuristic above it; objective = maximize `severity × traffic_weight × quality_recovery_factor` s.t. one action per document and total cost ≤ budget; `scripts/plan_migration.py` and `scripts/evaluate_prioritization.py` | Empirical calibration of recovery factors; brute-force validation; greedy-vs-exact optimality-gap study (see Section 6); ablations |
| 5 | Migration + verification | Planning, patch hook, metrics, quality gate implemented; index adapter pending | `plan_migration` (auditable per-document plan), `apply_patch_transform` (shape-preserving temporary transform hook), `evaluate_quality_gate` (blocks promotion when nDCG@k/Recall@k regression exceeds tolerance), `ShadowIndex` and `execute_plan` (local exact-search adapter) | No external vector-DB adapter; rollback/block-path tests with a fake index [VERIFY]; real patch-transform learning (alignment between generations) [VERIFY] |
| 6 | Dashboard | Initial artifact dashboard implemented; migration views pending | `scripts/dashboard.py` (Streamlit) showing contamination and traffic summaries and detector comparison | Action recommendations, budget use, quality recovery, migration outcomes, gate status, missing-artifact guidance |
| 7 | Final evaluation report | Not started | none | Everything (Section 7, workstream F) |

### Commands that exist today (from README)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
python scripts/build_benchmark.py --config config/default.yaml
python scripts/evaluate_detection.py --config config/default.yaml
python scripts/plan_migration.py --ratio 0.20 --budget 300
python scripts/evaluate_prioritization.py --ratio 0.20
python scripts/dashboard.py
```

---

## 5. Known inconsistency between repo documents (resolve first)

- `WORK_PLAN.md` ("Current baseline") states that the detection, prioritization, migration, and dashboard packages are **stubs**.
- `README.md` (status table and body) describes all of them as **implemented** at some level.

The README appears newer and `WORK_PLAN.md` appears stale, but this is unconfirmed. The first TODO item ("Reconcile README phase status with the implemented benchmark code") exists for this reason. **Action:** read each package, run `pytest`, and update both documents to match reality before building anything new. Also note `TODO.md` lists "Add budget sweep and cost-versus-quality-recovered evaluation" twice (a duplicate line).

---

## 6. Critical known issue: greedy scheduler optimality gap

During stress testing (reported by the owner in session), a simple **flat benefit-to-cost ratio greedy** produced an **88% optimality gap** against the exact solver on a stress-test configuration, while the gap was about 0% on the primary data. Cause as diagnosed: flat ratio ranking grabs a cheap early option for a document and never reconsiders upgrading it, which is a known failure mode of naive greedy on MCKP (not on plain 0/1 knapsack).

**Whether the repository's current greedy has this flaw is unknown [VERIFY].** The README describes the heuristic as "marginal-recovery-per-cost", which may already be the correct approach, or may be the flawed one under a different name.

**Required approach (standard MCKP method, rooted in Sinha & Zoltners 1979):**

1. For each document, build its option list `(cost, benefit)` for `defer`, `patch`, `full_reembed`.
2. **Prune dominated options:** drop any option with higher cost and not-higher benefit than another.
3. **Keep only the upper convex hull** (efficient frontier) of the remaining options; drop LP-dominated options.
4. Compute **incremental efficiency** for each step up the frontier: `Δbenefit / Δcost` between consecutive options.
5. Sort all incremental upgrades globally by efficiency descending; apply them while budget remains. An upgrade replaces the document's previous choice (this is what lets it "reconsider").
6. Handle the final partially affordable step deterministically (document the tie-break), optionally followed by a cheap improvement pass.

**Decision recorded:** keep the naive ratio greedy in the codebase as a named alternate mode (for example `--scheduler naive-ratio` versus `--scheduler efficient-frontier`), make the efficient-frontier version the default, and **report the naive result as a documented ablation** ("we found an 88% gap under stress condition X, diagnosed the mechanism, and closed it to Y%"). Before fixing, confirm the gap really comes from this mechanism and not from a bug in the exact-solver comparison or in how the stress config generates benefit/cost values.

---

## 7. Remaining work, by workstream

Each workstream lists tasks and its "done when" criteria. Suggested order: 0 → A → B → C → D → E → F.

### Workstream 0 — Baseline and benchmark readiness

**Tasks**
- Reconcile README, WORK_PLAN, and TODO with actual code (Section 5).
- Run the benchmark on a small, reproducible configuration; inspect the generated artifacts.
- Document artifact schemas: document IDs, embeddings for Gen A and Gen B, contamination labels, traffic weights, FAISS index files, run metadata. Add a validator that asserts row alignment across all of them.
- Document benchmark commands, expected outputs, and resource requirements (RAM, disk, time).

**Done when**
- A clean checkout can build or reuse benchmark artifacts from documented commands.
- Counts, dimensions, ordering, labels, and traffic totals can be inspected and validated programmatically.

### Workstream A — Contamination detection (Phase 3)

**Tasks**
- Confirm the detector contract: input = row-aligned embedding matrix + doc IDs (optionally the index for neighborhoods); output = `DetectionResult` with one finite percentile score in [0,1] per document, higher = more suspicious. All detectors share this orientation.
- Validate the GMM neighborhood scorer: for each document, take k-NN cosine-distance summary, fit 1- versus 2-component Gaussian mixtures, choose by BIC, derive severity from how far the document sits from its assigned sub-population. Document the exact score formula.
- Add synthetic unit fixtures: clean single-cluster, clearly mixed two-cluster, tiny `n` (smaller than k), duplicate vectors, constant vectors, NaN guarding.
- Run `evaluate_detection.py` across all configured ratios (5, 10, 20, 35, 50 percent) and multiple seeds. Report precision, recall, F1, PR-AUC, runtime, and score stability across seeds, for all four detectors.
- Save every result with config, model names, seed, and dataset metadata.

**Done when**
- Every detector returns finite, consistently oriented scores, one per document.
- Labels are used for evaluation only.
- Results are reproducible from saved configs. The owner's target is at least 90 percent precision and recall across ratios **as an objective, not a guarantee**: report honestly if it is not met, and explain where the detector fails (for example, neighborhoods dominated by semantic content rather than model generation).

### Workstream B — Budget-aware prioritization (Phase 4)

**Tasks**
- Write the objective down formally: inputs per document (severity, traffic weight, candidate actions, cost, expected recovery), score normalization, and the benefit definition `severity × traffic × recovery_factor(action)`.
- Validate the exact PuLP solver against brute force on tiny instances (every document gets exactly one action, cost ≤ budget, optimal value matches).
- Implement the efficient-frontier greedy (Section 6) and keep the naive mode for ablation.
- Edge cases: empty input, zero budget, budget smaller than the cheapest non-defer action, ties, infeasible actions, all-zero severities.
- Budget sweep: expected recovery versus budget for all strategies. Baselines to compare at matching budgets: FIFO/arbitrary-order full re-embed, uniform patch applied to everything (Drift-Adapter-style), naive ratio greedy, efficient-frontier greedy, exact solver (small instances).
- Ablations: severity-only, traffic-only, combined. Sensitivity analysis of recovery-factor assumptions.
- Report the greedy-versus-exact optimality gap across instance sizes and including a stress configuration.

**Done when**
- Every document receives exactly one valid action; total cost respects the budget.
- Exact solver matches brute force; heuristic is benchmarked against exact on tractable instances.
- Recovery factors and their assumptions are visible in saved results and clearly labeled as assumptions until empirically calibrated.

### Workstream C — Migration and quality gate (Phase 5)

**Tasks**
- Define the index abstraction and persisted plan/result schemas before adding more actions.
- Full re-embedding into a **shadow index**; the active index is untouched until promotion.
- Temporary **patch transform**: specify and implement a real, simple alignment between Gen A and Gen B spaces (for example, a learned linear or orthogonal Procrustes map fitted on documents available in both spaces), keep it shape-preserving, and document its limitations explicitly. If currently only a hook, implement the fit.
- **Gated alias swap:** promotion only after required validation passes; a failed batch or failed check must never publish a partially built index, and must report why promotion was blocked.
- Retrieval evaluation using the golden query set and qrels: nDCG@k and Recall@k (k configured, for example 10), measured before and after each batch.
- Enforce the configured regression tolerance. Record before/after metrics and migration cost per batch.
- Tests with a small in-memory fake index covering success, block, and rollback paths. No external service in tests.

**Done when**
- Actions are repeatable and produce auditable per-document outcomes.
- Promotion happens only after all validation passes; failure paths preserve the active index.
- An external vector-DB adapter is optional. If not built, labels in code, docs, and UI must say migration is a local prototype (see risks).

### Workstream D — Dashboard and demo (Phase 6)

**Tasks**
- Run/artifact selection with helpful guidance when artifacts are missing (no stack traces).
- Views: contamination map, recommended actions under a chosen budget, budget used, quality recovered, migration outcomes, gate status.
- Keep computation in package modules; the UI calls the same APIs as the scripts.
- Clearly distinguish **measured** results from **simulated estimates** in every chart. Charts expose their data and units.
- **Autonomous "Live Sentry" demo (Phase II showcase):** two databases side by side, "Database A — Stable" (untouched control) and "Database B — Live". A genuine background monitor (own thread or async task, polling every 2 to 5 seconds, configurable) scores both with the detector. The **only manual control** is "Simulate embedding provider update", which injects wrong-generation vectors into Database B. Everything after is autonomous: alert and event log appear, prioritization plan is logged, migration and verification run, alert clears when verification passes. Stretch goal: a third panel "Database C" healed with the baseline FIFO method for a live side-by-side comparison.
- Documented local launch command and a small demo workflow.

**Done when**
- One documented command launches the demo against a reproducible example run.

### Workstream E — Phase I / Phase II packaging for reviews

- Keep Phase I (baselines: LOF, Isolation Forest, KNN-OOD detection; FIFO and uniform-patch remediation under a fixed budget) clearly separable from Phase II (MCKP scheduler, migrate/verify loop, head-to-head comparison, live demo).
- Suggested: top-level `phase1_baseline/` and `phase2_proposed/` result folders, shared benchmark in a common location, and a git tag (for example `phase1-review`) at the exact review-ready commit so Phase I can always be checked out and demoed in isolation.
- Save Phase I metrics and charts; Phase II evaluation must reuse the **identical** benchmark, budgets, and ratios for a fair comparison.

### Workstream F — Final evaluation and reproducibility (Phase 7)

**Tasks**
- Full experiment matrix: 4 detectors × 5 contamination ratios × remediation strategies × multiple budget levels × multiple seeds where compute permits.
- Report retrieval quality, detection quality, recovery per cost, runtime, and variation across seeds (not single-run numbers).
- Save charts and tables under `data/results`, with an index file listing every artifact and the command that produced it.
- Complete `REPRODUCE.md` with exact environment and commands that regenerate every number and chart.
- Update README status, architecture, limitations, and demo instructions to match what actually runs.
- Repository hygiene: confirm generated datasets and results are not committed.
- Draft written findings in a form that could be expanded into a short paper (methodology, results, optimality-gap analysis, limitations). Formal paperwork waits for rubrics.

**Done when**
- Every claim in a report can be regenerated from documented commands and saved configs.
- Conclusions separate synthetic-benchmark evidence from claims about production systems.

---

## 8. Data and artifact conventions

- **Document identity:** every positional array (embeddings, labels, traffic, scores) must be accompanied by, or validated against, one shared ordered list of document IDs.
- **Run metadata:** each saved result includes config snapshot (or hash), dataset name, model names, seed, contamination ratio, budget, timestamp, and code version if available.
- **Scores:** percentile in [0,1], higher = more suspicious, across all detectors.
- **Contamination ratios:** 5%, 10%, 20%, 35%, 50% (configurable). Traffic follows a Zipfian distribution so few documents carry most queries.
- **Gitignored:** `data/raw`, `data/processed`, `data/results`. Regenerated by scripts.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Embedding neighborhoods dominated by semantic content rather than generation, hurting the GMM detector | Evaluate across seeds and ratios; document failure regimes honestly |
| Label leakage into detectors | Labels only in evaluation code paths; add a test that detector `fit`/`score` never receive labels |
| Misleading expected-recovery numbers | Treat recovery factors as explicit assumptions; run sensitivity analysis; calibrate empirically against measured nDCG recovery |
| Greedy scheduler optimality gap on adversarial instances | Efficient-frontier greedy + stress-config ablation (Section 6) |
| Resource-heavy full benchmark | Small synthetic fixtures for tests; make full embedding runs optional and cache-aware |
| Migration overclaim | Label local shadow-index abstractions as prototype behavior unless a real vector-DB integration exists |
| Suspiciously clean results | If a metric looks too good, hunt for evaluation bugs before reporting it |

## 10. Definition of done for the whole project

1. Clean clone → documented setup → `pytest` passes → benchmark builds → every script runs.
2. Detection metrics, prioritization sweeps, and migration/quality-gate results exist for all configured ratios and multiple seeds.
3. Head-to-head chart: quality recovered versus budget spent for FIFO, uniform patch, and the prioritized scheduler, on the identical benchmark, with the improvement quantified.
4. Autonomous live demo runs reliably (and a recorded backup exists).
5. `REPRODUCE.md` regenerates every reported number.
6. README and WORK_PLAN accurately describe the current state and limitations.

## 11. Instructions for an AI agent taking over

1. Read `README.md`, `WORK_PLAN.md`, `TODO.md`, `.agents.md`, and `config/default.yaml`, then the source in `src/vectorsentry/`. Update this document's **[VERIFY]** items with what you find.
2. Run `pytest` and the benchmark on a small config before writing new code. Report failures to the owner rather than silently working around them.
3. Work in sub-phases. After each, summarize what you built, show results, list decisions needing the owner's input, and wait for a go-ahead.
4. Comment code thoroughly; the owner must defend every decision to examiners.
5. Never add paid APIs or keys. Keep everything runnable on a laptop or free-tier GPU.
6. Do not write formal reports or paperwork unless the owner provides a rubric and asks.
7. Record notable decisions and progress in `.agents.md` and tick items in `TODO.md`.
8. If a result looks too clean, say so and investigate before presenting it.
