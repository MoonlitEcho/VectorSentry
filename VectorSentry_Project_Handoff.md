# VectorSentry — Project Handoff Document

**Purpose:** This document lets any engineer or AI coding agent pick up the VectorSentry project cold and continue it correctly. It states what the project is, what exists, what is verified versus unverified, and exactly what remains, with acceptance criteria for each piece.

**Repository:** https://github.com/MoonlitEcho/VectorSentry (public, MIT license). The nested `Final year project/` directory is a separate checkout of the GitHub repository at commit `37db1be`. This handoff describes the active outer checkout, currently at commit `09d258e`.

**How this document was updated (read this first):** It was reconciled against the active outer checkout's `README.md`, `DEVLOG.md`, source tree, result artifacts, tests, and recent commit history. The nested `Final year project/` checkout is retained for reference and is not the active implementation. The outer checkout is the source of truth for current progress.

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

The active checkout separates the baseline and proposed systems while sharing one benchmark and utility layer:

```
Vector store (mixed generations)
   │
   ▼
1. DETECT       Phase 1: KNN-OOD / LOF / IF (assumed ratio)
   │            Phase 2: generation fingerprint vs. the probe batch -> P(contaminated)   (phase2/fingerprint.py)
   ▼
2. PRIORITIZE   Phase 2: expected-benefit MCKP over full re-embed / patch / defer          (phase2/scheduler.py)
   │            benefit = P(contaminated) x impression traffic x replay-estimated recovery (phase2/calibration.py)
   │            (frontier greedy; exact PuLP solver for the optimality gap)
   ▼
3. MIGRATE      staged, priority-ordered rollout; patch = ridge adapter on the stored vector (phase2/migrate.py)
   │
   ▼
4. VERIFY       per-wave non-inferiority gate: replay logged non-eval queries vs. pre-incident
                results; rejected budget recycled into a re-plan                            (phase2/pipeline.py)
```

The 2026-10-24 "Redesign" entry in `DEVLOG.md` records why each stage changed
and the measured effect (Phase 2 now wins 30/30 combinations, +52% of headroom).

### Active repository layout

```
common/                    shared benchmark, data, metrics, indexing, config, and patch utilities
phase1_baseline/           existing detectors, FIFO/uniform remediation, dashboard, tests, results
phase2_proposed/           MCKP scheduler, migration/verification, comparison, Live Sentry, tests, results
docs/                      supporting notes
DEVLOG.md                  authoritative running record of implementation and results
scratch_review1/           review decks, scripts, and speaker materials
VectorSentry_Project_Handoff.md  this handoff
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

The active outer checkout is substantially ahead of the nested GitHub checkout described in the original handoff.

| Area | Current status | Evidence / location | Remaining work |
|---|---|---|---|
| Shared benchmark | Complete and reused by both phases | `common/`, `phase1_baseline/results/`, `phase2_proposed/results/` | Optional clean-checkout reproduction pass |
| Phase I detection | Complete and demo-ready | `phase1_baseline/src/phase1/detection.py`, detection results and tests | No core implementation work |
| Phase I remediation | Complete | `phase1_baseline/src/phase1/remediation.py`, remediation results and tests | No core implementation work |
| Phase II scheduler | Complete, including efficient-frontier greedy and stress evaluation | `phase2_proposed/src/phase2/scheduler.py`, scheduler results and tests | Document final findings in the report |
| Migration and verification | Complete local prototype | `phase2_proposed/src/phase2/migrate.py`, migration results and tests | External vector-DB adapter remains optional |
| Live Sentry | Complete: web demo (`phase2_proposed/web_app.py`, port 8600) with Streamlit fallback; post-incident patch clean-up | `phase2_proposed/src/phase2/live_sentry.py`, `web_app.py`, `web/`, `live_sentry_dashboard.py`, tests | Automated tests for the web server |
| Phase comparison | Complete | `phase2_proposed/scripts/run_comparison.py`, comparison results | No core implementation work |
| Review packaging | Complete for current review materials | `scratch_review1/`, recent commits, `DEVLOG.md` | Keep materials synchronized if code changes |
| Final evaluation/report | Not yet assembled as the final academic submission | Existing results are available; no final report in this checkout | Write findings, limitations, and reproducibility instructions |

### Active commands

```bash
.venv\Scripts\python.exe -m pytest
python common/scripts/build_benchmark.py
python phase1_baseline/scripts/run_detection.py
python phase1_baseline/scripts/run_remediation.py
python phase2_proposed/scripts/run_scheduler.py
python phase2_proposed/scripts/run_migration.py
python phase2_proposed/scripts/run_comparison.py
streamlit run phase2_proposed/live_sentry_dashboard.py
```

Use the phase-specific YAML files under `phase1_baseline/configs/` and `phase2_proposed/configs/` for experiment variants.

---

## 5. Reconciled status and known limitations

The earlier handoff described the single-package repository as if detection, prioritization, migration, and dashboard work were still pending. That description is stale for the active outer checkout. The outer project has completed Phase I and Phase II implementation and has saved result artifacts.

The remaining limitations are real but do not block the capstone prototype:

- The evidence is from a simulated benchmark, not a production vector database.
- Migration uses a local exact-search/shadow-index prototype; no external vector-database adapter is included.
- Recovery factors and some quality estimates are experimental assumptions and must be labeled as such.
- The formal final evaluation/report has not yet been assembled.
- The nested `Final year project/` directory has its own `.git` and virtual environment and should not be treated as part of the active outer repository.

The active `DEVLOG.md` is the authoritative record of completed sub-phases and decisions. Update it when new work changes the status.

---

## 6. Scheduler status

The previously reported naive flat benefit-to-cost greedy failure is now addressed in the active implementation. The Phase II scheduler retains the simpler strategy for comparison and uses an efficient-frontier approach for the production-scale schedule. The stress-test results and optimality-gap analysis are stored under `phase2_proposed/results/` and documented in `DEVLOG.md`.

Do not replace the frontier scheduler with a flat ratio sort. Any future change must preserve the exact-solver comparison, the stress configuration, deterministic tie-breaking, and the existing scheduler tests.

---

## 7. Genuine remaining work

### A. Final evaluation and academic packaging

- Assemble the final report from the existing Phase I and Phase II results.
- State clearly which results are measured on the simulated benchmark and which values are model-based recovery estimates.
- Include detection quality, retrieval quality, recovery per cost, scheduler optimality gaps, runtime, limitations, and the Phase I versus Phase II comparison.
- Record exact commands, environment assumptions, configurations, and artifact locations needed to reproduce reported numbers.

### B. Reproducibility verification

- Run the complete test suite from the project virtual environment and record the result.
- Verify that a clean setup can reuse or regenerate the benchmark and reproduce the saved result tables and charts.
- Confirm that generated data remains outside version control and that result copies in `phase2_proposed/results/from_phase1_baseline/` are synchronized with the canonical Phase I results.

### C. Documentation maintenance

- Keep `README.md`, `DEVLOG.md`, this handoff, and review materials aligned with the active split architecture.
- Document the local-prototype boundary wherever migration is described; do not imply a live external vector-database integration.

### D. Optional enhancements

- Add the deferred third Live Sentry panel showing the Phase I FIFO baseline.
- Add a real vector-database adapter if production integration is required.
- Replace the temporary patch/recovery assumptions with a learned alignment and empirically calibrated recovery factors if the research scope is expanded.

The core implementation does not require further architectural work before the final report and reproducibility pass.

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

## 10. Definition of done for the current capstone implementation

1. Clean clone → documented setup → `pytest` passes → benchmark builds → every script runs.
2. Detection metrics, prioritization sweeps, and migration/quality-gate results exist for all configured ratios and multiple seeds.
3. Head-to-head chart: quality recovered versus budget spent for FIFO, uniform patch, and the prioritized scheduler, on the identical benchmark, with the improvement quantified.
4. Autonomous live demo runs reliably (and a recorded backup exists).
5. `REPRODUCE.md` regenerates every reported number.
6. README and WORK_PLAN accurately describe the current state and limitations.

The core implementation is considered complete when the existing Phase I/Phase II artifacts, tests, comparison, and Live Sentry demo remain intact. The remaining completion gate is the reproducibility pass and final academic evaluation. The third Live Sentry comparison panel and an external vector-database adapter are optional enhancements, not completion requirements.

## 11. Instructions for an AI agent taking over

1. Read `README.md`, `DEVLOG.md`, this handoff, and the relevant phase source/config files before changing code. Treat the outer checkout as authoritative; inspect `Final year project/` only for historical comparison.
2. Run the existing tests and a small reproduction command before writing new code. Report failures to the owner rather than silently working around them.
3. Work in sub-phases. After each, summarize what you built, show results, list decisions needing the owner's input, and wait for a go-ahead.
4. Comment code thoroughly; the owner must defend every decision to examiners.
5. Never add paid APIs or keys. Keep everything runnable on a laptop or free-tier GPU.
6. Do not write formal reports or paperwork unless the owner provides a rubric and asks.
7. Record notable decisions and progress in `DEVLOG.md`; update this handoff when the remaining-work status changes.
8. If a result looks too clean, say so and investigate before presenting it.
