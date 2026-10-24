# VectorSentry

Detect, prioritize, and remediate mixed-generation embedding contamination in
production vector databases.

Production RAG / vector-search systems degrade silently when an index ends
up with a mix of embeddings from two different embedding-model generations
(e.g. some documents embedded with Model A, others later re-embedded with
Model B after a provider update). Cosine similarity is not comparable across
generations, so retrieval quality quietly degrades with no error thrown.

- **Phase 1** ([`phase1_baseline/`](phase1_baseline/)) implements the
  existing state of the art honestly: unsupervised contamination detection
  (KNN-OOD, LOF, Isolation Forest) plus non-prioritized remediation under a
  budget (FIFO order-of-detection, and a uniform patch transform).
- **Phase 2** ([`phase2_proposed/`](phase2_proposed/)) adds the proposed
  system, compared head-to-head against Phase 1 on the same benchmark and
  budget:
  - *Detect*: a generation-fingerprint detector that treats the index as a
    mixture of two embedding populations, using the probe batch an operator
    dual-embeds anyway (no assumed contamination ratio).
  - *Prioritize*: a Multiple-Choice Knapsack Problem (MCKP) over full
    re-embed / patch / defer, with expected benefits built from the
    detector's probability, impression-log traffic, and patch
    recovery/harm estimated by replaying the query log.
  - *Migrate*: a staged, priority-ordered rollout, each wave gated by
    replaying logged queries against the shadow index; budget from rejected
    fixes is recycled.
  - Headline: Phase 2 beats the better Phase 1 baseline in 30/30 ratio x
    budget combinations, +52% of the recoverable nDCG@10 on average (see
    [`phase2_proposed/results/README.md`](phase2_proposed/results/README.md)).
- **[`common/`](common/)** holds the benchmark-generation code and shared
  utilities both phases are evaluated against, so the comparison is fair.

No paid API keys are required anywhere -- everything runs locally with open
models and a laptop-sized corpus.

## Repo layout

```
common/               shared benchmark generation, config, seeding, metrics
phase1_baseline/       Phase 1 code, configs, tests, results
phase2_proposed/       Phase 2 code, configs, tests, results
docs/                  supporting notes (not formal report content)
DEVLOG.md              running log of decisions and results
```

Phase 1 is tagged at `phase1-review` once it is demo-ready, so it can always
be checked out and demoed in isolation from later Phase 2 changes:

```bash
git checkout phase1-review
```

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -e ".[dev]"
```

This does one editable install that makes `common`, `phase1`, and `phase2`
all importable (`import common`, `import phase1`, `import phase2`) regardless
of which top-level folder you're running a script from. See
[`pyproject.toml`](pyproject.toml) for why: the three folders stay separate
at the filesystem/git level (for the two independent review panels) but
share one virtual environment for convenience.

## Reproduction

```bash
pytest                                  # fast unit tests (~1 min)
pytest -m integration                   # + slower tests hitting real BEIR data/models

python common/scripts/build_benchmark.py                    # build the shared benchmark (~10-20 min, one-time)
python phase1_baseline/scripts/run_detection.py             # baseline detectors -> phase1_baseline/results/
python phase1_baseline/scripts/run_remediation.py           # baseline remediation -> phase1_baseline/results/
python phase2_proposed/scripts/run_detection.py             # fingerprint vs. baseline detectors -> phase2_proposed/results/
python phase2_proposed/scripts/run_scheduler.py             # MCKP schedule + optimality gap
python phase2_proposed/scripts/run_migration.py             # full pipeline, ablations, gate negative control
python phase2_proposed/scripts/run_comparison.py            # Phase 1 vs. Phase 2 head-to-head
python -m streamlit run phase1_baseline/dashboard.py        # interactive Phase 1 dashboard
python phase2_proposed/web_app.py                           # Phase 2 live demo (web UI) -> http://localhost:8600
python -m streamlit run phase2_proposed/live_sentry_dashboard.py # Phase 2 live dashboard (Streamlit fallback)
```

Run these commands after activating `.venv` so Streamlit uses the same Python
environment as the project dependencies. On Windows, the unambiguous form is
`.venv\\Scripts\\python.exe -m streamlit run ...`.

Each script accepts `--config PATH` to point at a different experiment
config (e.g. `phase1_baseline/configs/aniso_stress_bge.yaml`, a documented
stress-test pairing -- see `DEVLOG.md`'s sub-phase 1C entry).

See [`DEVLOG.md`](DEVLOG.md) for the running record of what exists, what
was measured, and every non-obvious design decision along the way.

## Configuration

All experiment parameters (contamination ratios, budget levels, model names,
corpus size, random seed) live in YAML, not hardcoded in scripts --
see [`phase1_baseline/configs/default.yaml`](phase1_baseline/configs/default.yaml)
and the loader in [`common/src/common/config.py`](common/src/common/config.py).
