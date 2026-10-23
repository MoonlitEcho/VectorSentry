# DEVLOG

Running log of decisions and results, kept incrementally so the formal
paperwork (written later, against whatever rubric is provided) can be
assembled quickly from real notes instead of reconstructed from memory.
Not a report -- just dated, honest notes.

---

## 2026-08-21 -- Sub-phase 1A: Scaffolding

**Built:**
- Git repo initialized at repo root (`VectorSentry/`).
- Three top-level folders: `common/`, `phase1_baseline/`, `phase2_proposed/`,
  each with a `src/<pkg>` layout, plus `tests/`, `configs/`, `results/` under
  each phase folder.
- Single root `pyproject.toml` doing one editable install across all three
  packages (`common`, `phase1`, `phase2`) -- see the comment in that file for
  why one virtualenv is fine even though the folders stay physically
  separate for the two review panels.
- Typed config system: `common/src/common/config.py` (dataclasses, YAML-backed)
  + `phase1_baseline/configs/default.yaml`. Every tunable (seed, corpus name/
  size, embedding model names, contamination ratios, traffic skew, budget
  levels) lives in YAML.
- `common/src/common/seed.py`: single `set_seed()` entry point (python
  `random` + numpy + torch-if-present) so Phase 1 and Phase 2 benchmarks are
  built from the same seed -- required for the head-to-head comparison in 2C
  to be apples-to-apples.
- `.gitignore`, `README.md` with setup + reproduction steps.

**Decisions worth remembering:**
- Chose **dataclasses + PyYAML** over pydantic for the config system: zero
  extra dependency, every field/default visible in one file for a viva
  walkthrough.
- Chose **not** to depend on the official `beir` pip package for corpus
  loading (planned for 1B). Its pinned dependency versions conflict with
  sentence-transformers. Plan: load BEIR datasets (SciFact / FiQA) from
  their HuggingFace mirrors (`BeIR/scifact` etc.) instead -- same documents,
  no dependency risk. Revisit if the HF mirror is missing something BEIR's
  loader provides (e.g. official qrels format).
- Chose **not** to depend on `pytrec_eval` for nDCG@10 / Recall@10 (needed
  in 1D). It's a compiled C binding with uncertain Windows wheel support;
  these metrics are simple enough to implement directly in
  `common` with numpy, which also makes the exact formula inspectable for
  examiners. Will implement in 1B/1C alongside the benchmark.
- Included `pulp` in root dependencies now (Phase 2's exact MCKP solver)
  even though it's unused until 2A, to avoid a second dependency-set churn
  later. It's pure-Python with a bundled CBC solver, no compilation risk.

**Status:** scaffolding only -- no benchmark, detector, or dashboard code
yet. Next: 1B (synthetic contaminated benchmark).

---

## 2026-08-21 -- Sub-phase 1B: Synthetic contaminated benchmark

**Built:**
- `common/src/common/data.py`: loads BEIR SciFact (corpus + queries + qrels)
  from its HuggingFace mirror; subsamples the corpus to `max_docs` while
  guaranteeing every qrels-referenced document survives the cut; simulates
  Zipfian search traffic at the query level and aggregates it to a
  per-document traffic score.
- `common/src/common/embeddings.py`: cached sentence-transformers encoding
  (L2-normalized, so cosine similarity = inner product).
- `common/src/common/index.py`: FAISS `IndexFlatIP` build/search helpers
  (exact search, not ANN -- see decisions below).
- `common/src/common/benchmark.py`: two-layer disk cache tying it together --
  a ratio-independent layer (documents, queries, qrels, traffic, both
  generations' embeddings) built once, and a ratio-dependent layer
  (contamination mask + mixed-generation FAISS index) built once per
  contamination ratio from cheap numpy indexing, not re-embedding.
- `common/scripts/build_benchmark.py`: CLI entry point that builds the full
  benchmark for every ratio in the config and prints a summary.
- Tests: `test_data.py` (fast, synthetic fixtures, no network) and
  `test_benchmark.py` (real BEIR + real models, marked `integration` and
  excluded from the default `pytest` run -- see decisions below).

**Real run against `phase1_baseline/configs/default.yaml`** (SciFact,
max_docs=5000, seed=42):
- 5,000 documents, 1,109 queries carry at least one qrels judgment (300 of
  those are in the official test split, used for retrieval-quality eval in
  1C/1D).
- Both embedding models confirmed 384-dimensional (verified before writing
  any pipeline code -- this is what makes the contamination "silent" rather
  than a loud dimension-mismatch error).
- Embedding the full corpus with both generations took ~984s (~16.4 min) on
  this CPU-only machine, one-time cost.
- Contamination ratios landed within noise of their targets (5%→256/5000,
  10%→473/5000, 20%→1044/5000, 35%→1760/5000, 50%→2528/5000 docs), each
  built in <1s thanks to the embedding cache.
- Doc traffic: **4,333 of 5,000 documents (86.7%) get exactly zero simulated
  traffic** -- only the 667 documents SciFact's qrels ever mark as relevant
  to some query receive nonzero traffic, max single-document traffic share
  0.230.

**Flagging for review (not obviously a bug, but worth watching going into
1C/2A):** the 86.7%-zero-traffic figure is a much more extreme version of
"a small number of documents dominate traffic" than I expected going in.
It falls directly out of using BEIR's real qrels as the traffic signal
(SciFact's qrels only ever mark ~13% of the corpus as relevant to any
query -- the rest are BEIR's own "distractor" documents). This is
defensible as realistic (most stored documents are never a direct search
target) and it isn't a computed/derived number that could hide an
evaluation bug -- it's a direct, inspectable property of the raw qrels data
(667 unique doc IDs, counted independently in DEVLOG's 1A verification
step). But it does mean Phase 2's MCKP benefit function
(`severity x traffic x recovery`) will assign exactly zero benefit to fixing
~87% of contaminated documents regardless of severity, which could make the
Phase 1 (FIFO) vs Phase 2 (prioritized) gap larger than a less extreme
traffic distribution would produce. Something to sanity-check narratively
once 1D/2A numbers exist -- if the story becomes "Phase 2 just ignores 87%
of the corpus," that's honest but should be stated plainly, not obscured.

**Decisions worth remembering:**
- **Traffic is simulated at the query level, aggregated to documents via
  qrels**, not synthesized per-document independent of BEIR's structure.
  Chose real qrels over synthetic per-doc queries so traffic reuses BEIR's
  actual judged-relevance structure rather than inventing a parallel
  fictional one -- more defensible to examiners, at the cost of the extreme
  skew noted above.
- **Traffic uses train+test qrels combined** (more coverage), but
  retrieval-quality evaluation (1C/1D) will use **test-split qrels only**
  (`qrels_eval`), matching standard BEIR practice.
- **Contamination assignment is uniform random over documents, independent
  of traffic**, not correlated with popularity -- modeling a real embedding-
  provider migration touching documents in arbitrary/ingestion order. Doc
  load order from BEIR doubles as the synthetic "ingestion order" the FIFO
  baseline (1D) will use.
- **Exact FAISS search (`IndexFlatIP`), not ANN (HNSW)** -- the corpus is
  small enough that exact search is free, and it removes ANN approximation
  error as a confound when attributing retrieval-quality changes to
  contamination specifically.
- **Embed each generation once, reuse across all ratios** via
  `where(mask, emb_b, emb_a)` -- confirmed empirically: the one-time
  embedding pass took ~984s, every subsequent ratio took <1s.
- **Integration tests are opt-in** (`pytest -m integration`), deselected
  from the default `pytest` run via `addopts` in `pyproject.toml`. Measured
  the naive per-test-fresh-fixture version at 918s for 4 tests; switching to
  a module-scoped fixture that builds the (800-doc) benchmark once dropped
  it to 188s. The default `pytest` run (11 unit tests) is ~8s.

**Status:** shared benchmark pipeline complete and run end-to-end at full
scale (5000 docs x 5 ratios). Next: 1C (baseline detection: KNN-OOD, LOF,
Isolation Forest).

---

## 2026-08-21 -- Sub-phase 1C: Baseline detection (KNN-OOD, LOF, Isolation Forest)

**Built:**
- `phase1_baseline/src/phase1/detection.py`: three unsupervised detectors --
  KNN-OOD (cosine distance to k-th nearest neighbor; primary baseline),
  LOF (`sklearn.neighbors.LocalOutlierFactor`, cosine metric; secondary),
  Isolation Forest (`sklearn.ensemble.IsolationForest`; secondary, weaker,
  non-distance-based comparison point). All three threshold their
  continuous anomaly score by flagging the top `round(ratio * N)` documents,
  mirroring sklearn's own `contamination=` convention -- an operator with a
  rough prevalence estimate (e.g. from a migration window), not oracle
  knowledge of which specific documents are contaminated.
- `evaluate_detection()`: precision/recall/F1 against ground-truth labels.
- `phase1_baseline/scripts/run_detection.py`: CLI that runs all three
  detectors across every configured contamination ratio and saves
  `detection_metrics.csv` + `detection_f1_by_ratio.png` to the config's
  results dir.
- Fast synthetic-cluster tests (`test_detection.py`) -- caught a real bug in
  my FIRST version of the test fixture (see decisions below) before it ever
  touched real data.

**Major finding, fully diagnosed before trusting any number (per the "flag
anything too clean/too broken" instruction):** the first real run, using
the sub-phase 1A/1B default pairing (MiniLM-L6 as Generation A, bge-small-
en-v1.5 as Generation B), produced KNN-OOD/Isolation Forest F1 scores of
essentially **0.000 at every ratio** (worse than random guessing) and LOF
F1 scores that matched *exactly* what a zero-signal detector produces by
construction (precision = recall = ratio, e.g. 0.206/0.197 at ratio=0.20 --
that's the closed-form expected overlap of flagging a random ratio-fraction
of documents). That pattern -- one method inverted, two at exact chance --
does not look like "a weak but real detector," it looks like a broken
evaluation. Diagnosed via three checks against the real cached benchmark
before accepting or rejecting the result:
1. Random pairwise cosine similarity *within* generation: MiniLM-L6 = 0.14,
   bge-small = **0.63**. bge-small's raw document embeddings occupy a
   narrow cone of the embedding space -- nearly any two unrelated documents
   are "similar." (A quick check of two other popular current embedding
   models, gte-small at 0.75 and e5-small-v2 at 0.81, showed this is common
   across modern contrastively-trained models, not a bge-small fluke --
   only same-family MiniLM-L6/L12 stayed comparably low, ~0.13.)
2. Centroid separation between the two generations was small (cosine sim
   0.086 between mean-A and mean-B directions) -- so the two generations
   aren't simply overlapping; B is just internally much denser than A.
3. Per-group KNN-OOD score means: contaminated (B) docs averaged **0.247**,
   clean (A) docs averaged **0.491** -- the anomaly signal is directly
   INVERTED, not merely weak. Because B's embeddings are so densely packed
   together, a contaminated document's nearest neighbors are mostly other
   contaminated documents at short distance, so it looks *more* normal, not
   less. Every distance/density-based detector's core assumption (anomalies
   are sparser than normal data) is violated by construction here.

This is a real, reproducible, well-understood result -- not an evaluation
bug -- but it changes the scope of what the benchmark's primary pairing
should be, so it was raised with the project owner rather than decided
unilaterally. **Decision (owner's call):** run both.
- **Primary pairing, used for the main Phase 1/Phase 2 comparison going
  forward:** Generation A = MiniLM-L6, Generation B = **MiniLM-L12**
  (`configs/default.yaml`). Same architecture family, comparable anisotropy
  (~0.13 both), so detectors behave the way the project brief assumes
  ("existing methodology CAN detect this"). Real run:
  KNN-OOD F1 ~0.51-0.57 across all five ratios (strongest, roughly flat),
  LOF ~0.20-0.51 (weaker at low ratios, recovers by ratio=0.50), Isolation
  Forest ~0.11-0.46 (weakest at low ratios, closes the gap by ratio=0.50).
  This is the expected qualitative ordering (KNN-OOD > LOF > Isolation
  Forest as primary/secondary/secondary) and is now the number Phase 1D's
  remediation and Phase 2's comparison build on.
- **Documented stress-test pairing, kept separate:** Generation B =
  bge-small-en-v1.5 (`configs/aniso_stress_bge.yaml`, results under
  `phase1_baseline/results/aniso_stress_bge/`). Reproduced the same
  inversion/chance-level pattern under the corrected cache (see bug below),
  confirming it wasn't a caching artifact. Kept as an explicitly-labeled
  bonus finding: naive cosine-distance unsupervised detectors can fail
  outright -- not just degrade -- when a "provider update" moves to a newer
  contrastively-trained embedding model with different intrinsic
  anisotropy. Worth a paragraph in the eventual report as a genuine
  limitation of current practice, clearly separated from the main results.

**Bug found and fixed while setting up the second config (not from the
detector work itself):** `common/benchmark.py`'s on-disk cache path was
keyed only on `corpus.name/max_docs/seed`, not on the embedding model
names. Building the MiniLM-L12 benchmark right after the bge-small one,
with the same corpus/seed, would have silently reused bge-small's cached
`emb_b.npy` -- corrupting the comparison without any error. Caught before
it produced a wrong number: fixed by nesting embeddings and every ratio's
mask/index under an `embeddings/<generation_a_slug>__vs__<generation_b_slug>/`
subdirectory (`common/src/common/benchmark.py`), with document/query/traffic
metadata staying model-independent (shared, as intended). Added
`test_benchmark_caching.py` -- a fast, network-free regression test
asserting the metadata dir is shared across model choices but the embedding
dir is not.

**Decisions worth remembering:**
- **Thresholding by assumed contamination ratio** (flag the top
  `round(ratio*N)` scores), matching sklearn's own `contamination=`
  parameter convention for LOF/IsolationForest, applied consistently to
  KNN-OOD too. Makes all three detectors' precision/recall directly
  comparable (same assumed prevalence, not three different implicit
  thresholds) -- realistic in that an operator usually has *some* estimate
  of how much a migration touched, even without knowing which documents.
- **Synthetic test fixture bug, fixed before it could mask a real bug**:
  the first version of `test_detection.py`'s synthetic fixture built two
  Gaussian blobs at different EUCLIDEAN offsets (centered at 0 vs. centered
  at 6). Under cosine distance, a cluster offset far from the origin in a
  consistent direction is nearly collinear with itself -- so it looks like
  the *tightest, most normal* cluster of all, inverting the intended
  signal. All three detectors failed this fixture (recall 0.0-0.2). Fixed
  by generating unit-normalized vectors directly (a tight cone around one
  direction for the "normal" cluster, scattered random directions for
  "contaminated" points) -- i.e. constructing the synthetic case in the
  same geometry the real pipeline uses. Notably, this is the exact same
  class of anisotropy mistake that turned out to be the REAL finding above
  -- worth remembering when reasoning about cosine-space detectors anywhere
  else in this project.
- **Isolation Forest operates on raw embedding components**, not cosine
  distance -- included specifically as a non-distance-based comparison
  point per the project brief ("secondary comparisons"), and it does
  underperform KNN-OOD/LOF at low contamination ratios in the primary
  pairing, as expected.

**Status:** baseline detection complete for the primary pairing (real
numbers saved to `phase1_baseline/results/detection_metrics.csv` +
`.png`), plus a documented stress-test finding. Next: 1D (baseline
remediation -- FIFO and uniform-patch under a budget).

---

## 2026-08-21 -- Sub-phase 1D: Baseline remediation (FIFO, uniform patch)

**Built:**
- `common/src/common/patch.py`: shared Drift-Adapter-style fix -- an
  orthogonal Procrustes rotation fit from a small, FIXED calibration
  ("probe batch") sample of documents, independent of any contamination
  ratio. Deliberately shared with Phase 2 (which reuses it for the "Patch"
  action in 2A) so neither phase can win by having access to a better fix.
- `common/src/common/metrics.py`: `ndcg_at_k`/`recall_at_k` (binary
  relevance, implemented directly -- see 1A's pytrec_eval decision) and
  `evaluate_retrieval()`, which reports BOTH the standard unweighted
  macro-average AND a traffic-weighted variant.
- `phase1_baseline/src/phase1/remediation.py`: `fifo_remediation_plan()` --
  given a flagged mask, a budget, and a cost-per-fix, decides which
  documents get fixed purely by ingestion order. No severity, no traffic,
  no prioritization -- that gap is the whole point of Phase 2.
- `phase1_baseline/scripts/run_remediation.py`: sweeps every ratio x budget
  level for two strategies (FIFO full re-embed, uniform patch), plus two
  reference points (no remediation; an "oracle" that fully re-embeds every
  TRULY contaminated doc, not just flagged ones). Saves both an unweighted
  and a traffic-weighted quality-vs-budget chart.
- Refactored `compute_query_traffic()` to also return per-query weights
  (previously only the doc-aggregated array was exposed) -- needed to
  actually compute the traffic-weighted metric. Benchmark cache now also
  persists `query_traffic.json`; old caches missing that file are detected
  and rebuilt automatically (cheap: no re-embedding needed, just a BEIR
  reload + traffic recompute).

**Design decisions:**
- **Remediation operates on the DETECTOR's flagged set, not oracle ground
  truth.** A real operator only has what detection gave them; false
  negatives (contaminated docs the detector missed) can never be fixed
  here regardless of budget. This is why even FIFO at budget=1.0 (fixing
  100% of *flagged* docs) doesn't reach the oracle line -- the gap is
  specifically the cost of imperfect detection recall, not remediation.
- **Patch calibration uses a fixed random "probe batch,"** not the
  ratio's true-clean documents and not the detector's not-flagged set.
  Modeled as a small sample an operator would deliberately dual-embed
  BEFORE a migration specifically to validate a new provider -- realistic,
  and avoids any risk of the fit leaking information about which documents
  later happen to become contaminated in a given ratio. Fit once per
  model-pair, reused across every ratio/budget (it doesn't depend on
  either).
- **Budget is in "full-reembed-equivalent" units**: `budget_units =
  budget_level * n_flagged * cost_full_reembed`, so `BudgetConfig.levels`
  stays meaningful regardless of corpus size or contamination ratio.
  `cost_patch = 0.1` (10x cheaper) means uniform patch reaches 100% of
  flagged documents at almost every configured budget level -- itself an
  illustrative result (see below), not a config mistake.

**Two findings, both diagnosed before trusting the numbers (same discipline
as 1C's detector-inversion check):**

1. **The traffic-weighted metric has very low effective sample size.** The
   first plot (traffic-weighted nDCG@10) looked suspiciously flat across
   every budget level and both strategies. Diagnosed: with `zipf_s=1.2`
   applied to the ~300-query eval set, effective sample size (1/sum(w^2)
   of the renormalized weights) is only **~6.5 queries** -- one single
   query carries 37% of the total weight, the top 3 carry 51%. This isn't
   a bug (unit tests in `test_metrics.py` confirm the weighting logic
   itself is correct on controlled cases) -- it's what "a small number of
   documents get most of the simulated search volume" looks like when
   taken to its logical conclusion with this particular skew parameter
   over a modest query pool. Consequence: the traffic-weighted metric is
   thematically the right one for this project's story, but is a noisy,
   low-resolution statistic on its own. **Response:** report both metrics
   always (already the design), and treat the unweighted nDCG@10/Recall@10
   as the primary, statistically robust quantitative comparison, with the
   traffic-weighted number and its ESS caveat reported alongside it, not
   in place of it. `run_remediation.py` now saves both charts explicitly
   labeled by which metric they show. Worth deciding later (with input,
   before 2C) whether to soften `traffic.zipf_s` for the final head-to-head
   comparison, given how much it concentrates the eval signal -- flagging
   here rather than changing it unilaterally.
   - **Sanity check that validated the pipeline overall:** the "oracle"
     (fully generation-A, zero contamination) unweighted nDCG@10 is
     **0.652**, consistent with published BEIR leaderboard numbers for
     `all-MiniLM-L6-v2` on SciFact. Good evidence the embedding/index/eval
     pipeline itself is sound, independent of the traffic-weighting issue.

2. **Uniform patch barely beats no-remediation on the actual retrieval
   metric, despite measurably improving embedding similarity.** At
   ratio=0.35, patched docs' nDCG@10 (0.4824) is statistically
   indistinguishable from doing nothing (0.4823) -- flat across every
   budget level (expected, since patch reaches 100% of flagged docs almost
   immediately) but suspiciously close to zero benefit. Diagnosed directly
   against real cached embeddings before accepting it: computed cosine
   similarity between patched embeddings and the TRUE generation-A
   embedding on held-out (non-calibration) documents. The patch clearly
   DOES work in embedding space -- mean similarity to the true embedding
   rises from 0.447 (raw, contaminated) to 0.634 (patched), helping 95% of
   documents. So the patch transform itself is not broken (also independently
   confirmed by `test_patch.py`'s synthetic recovery test). The
   disconnect: retrieval quality is a discrete top-k operation, and a
   partial realignment (0.634 average similarity, well short of the true
   embedding's 1.0) usually isn't enough to move a document from outside
   the top-10 to inside it for the specific queries where it matters --
   so a real, measurable embedding-space improvement often produces close
   to zero end-to-end retrieval benefit. This is a genuine, defensible
   empirical finding (a known-in-the-literature gap between embedding
   proximity and ranking behavior), not an evaluation bug, and it sharpens
   the FIFO-vs-patch tradeoff this project studies: FIFO is expensive but
   each fix reliably crosses the quality threshold; patch is cheap and
   covers everyone, but each individual fix is much less likely to
   actually change what a user sees.

**Status:** baseline remediation complete for the primary pairing --
`phase1_baseline/results/remediation_metrics.csv` +
`remediation_quality_by_budget_{unweighted,traffic_weighted}.png`. This is
the reference point Phase 2's prioritized scheduler must beat in 2C. Next:
1E (Streamlit dashboard demoing the contaminated index + FIFO remediation
gap live), then 1F (wrap-up, tag `phase1-review`).

**Follow-up:** asked whether to soften `traffic.zipf_s` before 2C given the
low-ESS finding above; decided to proceed as-is for now and revisit later
if needed -- not changed in this sub-phase.

---

## 2026-08-21 -- Sub-phase 1E: Streamlit dashboard

**Built:** `phase1_baseline/dashboard.py`, run with `streamlit run
phase1_baseline/dashboard.py`. Sidebar controls: contamination ratio,
remediation budget (0-1 slider), strategy (FIFO full re-embed / uniform
patch). Main panel: live metric cards (documents, ground-truth
contamination, detector flags + recall, docs fixed, cost spent), a
collapsible detector-quality panel, the **central visual** -- a table of
the highest-traffic contaminated documents showing whether each was
flagged and whether it got fixed under the current budget, a live
quality-vs-budget chart for both strategies with the current budget marked,
and a before/after top-10 search example (defaults to a query whose
relevant document is still broken, if one exists at the current
ratio/budget).

Two-tier caching to keep it responsive: `st.cache_resource` for the
one-time-per-process load (benchmark, patch transform, query embeddings),
`st.cache_data` for the per-ratio state (contaminated index + detection)
and the per-ratio budget sweep -- both keyed only on `ratio`, so moving the
budget/strategy widgets doesn't retrigger detection or the sweep, only the
cheap (~0.05s) remediation-plan + before/after computation.

**Verified in a real browser, not just "it imports":** started the app via
a `.claude/launch.json` config and drove it with the Claude Browser tools
(`get_page_text`/`read_page` for content, since headless screenshot
compositing wasn't available in this environment). Confirmed: the page
renders with real numbers on load (5,000 docs, 1,044 ground-truth
contaminated at the default ratio=0.20, matching 1B/1C's numbers exactly);
the data table and chart render as proper widgets (grid with
Search/Download/Fullscreen controls; chart as an embedded image); and,
critically, **the widgets are genuinely reactive** -- dragging the budget
slider to 1.0 via keyboard changed "Fixed under this budget" from 200/1,000
to 1,000/1,000 and recomputed cost spent accordingly; changing the ratio
slider to 0.50 correctly recomputed ground-truth contamination to
2,528/50.6%, matching the cached benchmark exactly. (Note: `form_input`
sets a slider's visible value but does not fire Streamlit's underlying
change event -- had to click the slider then send a keyboard `End` press
to actually trigger a rerun. Worth remembering for testing any future
Streamlit UI in this project.)

**Design note:** the before/after example's default query intentionally
targets a document that is CURRENTLY still broken (not yet fixed at the
selected budget) -- so by construction its "before" and "after" top-10
results are often identical (nothing changed for that specific document).
This is the correct behavior for illustrating the sub-phase's stated goal
("show clearly some high-traffic documents remain broken"), but it does
mean the default example alone doesn't demonstrate a successful recovery
case. The query dropdown lets a user pick any other eval query to see a
case where remediation did change the top-10 -- worth pointing this out
explicitly during a live demo rather than assuming the default view tells
the whole story.

**Status:** dashboard built and verified interactive end-to-end. Next: 1F
(wrap-up: consolidate results, summarize what Phase 1 achieves and where it
falls short, tag `phase1-review`).

---

## 2026-08-21 -- Sub-phase 1F: Phase 1 wrap-up

**Built:** `phase1_baseline/results/README.md` -- an index of every
artifact in the results folder, what produced it, and how to reproduce it,
so Phase 2's evaluation (2C) can point at exact numbers instead of
re-deriving them. Ran the full test suite (unit + integration) one more
time as a final check before tagging.

### What Phase 1 (the existing methodology, honestly implemented) achieves

- A complete, reproducible, seeded pipeline: a real BEIR corpus (SciFact,
  5,000 docs) mixed across two real embedding-model generations at 5
  contamination ratios, cached so the whole sweep reruns in seconds after
  the one-time embedding pass.
- Three real unsupervised detectors evaluated honestly against ground
  truth, not cherry-picked: KNN-OOD is a genuinely useful baseline (F1
  ~0.51-0.57, roughly flat across ratios), meaningfully better than chance
  and better than the two secondary detectors (LOF, Isolation Forest) it's
  compared against, matching the expected literature ordering.
- Two non-prioritized remediation strategies measured with REAL retrieval
  metrics (nDCG@10/Recall@10 on an actual FAISS index + actual queries),
  not a simulated proxy: FIFO full re-embed shows real, budget-proportional
  quality recovery; uniform patch reaches far more documents per unit
  budget but, measured honestly, provides far less retrieval benefit than
  its embedding-similarity improvement alone would suggest.
- A live, interactive dashboard that makes the core limitation (below)
  concrete and explorable, not just asserted in a chart.
- Two rigorously diagnosed findings that sharpen the project's own
  motivation rather than just filling a checklist: naive cosine-distance
  detectors can be actively fooled (not just weakened) by embedding-model
  anisotropy (the bge-small stress test); and a measurably-improved
  embedding often produces close to zero retrieval-quality improvement,
  because ranking is a discrete top-k operation a partial fix often
  doesn't cross.

### Where it falls short -- the bridge to Phase 2

- **Detection has a hard ceiling.** Recall is ~52-58% at the operating
  point used throughout (assumed-ratio thresholding). No amount of budget
  lets FIFO fix a document it was never told about -- every quality-vs-
  budget chart shows FIFO approaching but never reaching the oracle line,
  and that gap is entirely attributable to missed detections, not
  insufficient budget.
- **No prioritization exists in current practice.** FIFO's only decision
  rule is ingestion order. It has no way to know that one flagged document
  sits behind a query almost nobody asks, while another sits behind the
  single highest-traffic query in the whole corpus. The dashboard's
  "which high-traffic documents are still broken?" table makes this
  concrete: under a limited budget, whether a critical document gets fixed
  is a function of where it happened to land in the corpus, not whether
  fixing it matters.
- **Only uniform, corpus-wide strategy choices are available.** Current
  practice picks ONE strategy ("FIFO full re-embed" or "patch everything")
  and applies it blindly to every flagged document. There is no way to say
  "spend the expensive, reliable fix on the handful of documents that
  matter most, cheaply patch a wider tier, and defer the rest" -- a
  genuinely different DECISION per document, budget-constrained across all
  of them jointly. That is precisely the Multiple-Choice Knapsack
  formulation Phase 2 introduces: same detection infrastructure, same
  migration mechanics (full re-embed, patch), but a scheduler that chooses
  WHICH action for WHICH document under ONE shared budget, instead of one
  action for everyone.
- **A methodological note to carry forward, not a limitation to hide:**
  the traffic-weighted metric -- thematically the one that most directly
  measures real user-facing impact -- is statistically noisy at the
  `zipf_s=1.2` used throughout (effective sample size ~6.5 of ~300 eval
  queries). Phase 2's head-to-head comparison (2C) should read that metric
  with this in mind, and lean on the unweighted metric as the primary
  robust comparison, exactly as this phase already does.

**Status:** Phase 1 is complete and demo-ready. Tagged `phase1-review` at
this commit -- see the tag message for the exact commit this corresponds
to. Phase 2 work starts from here, in `/phase2_proposed`, reusing
`/common`'s benchmark infrastructure without modifying anything under
`/phase1_baseline`.

---

## 2026-08-25 -- Sub-phase 2A: MCKP prioritization scheduler

**Built:**
- `phase2_proposed/src/phase2/scheduler.py`: the MCKP formulation --
  `compute_severity()` (min-max normalized detector anomaly score),
  `build_mckp_items()` / `build_efficient_frontier_segments()` (benefit/cost
  construction), and three solvers (`solve_exact`, `solve_greedy_naive`,
  `solve_greedy` -- see below for why there are two greedys).
- `common/config.py`'s new `SchedulerConfig`: `recovery_full_reembed=1.0`
  and `recovery_defer=0.0` by construction; `recovery_patch=0.064`,
  **empirically measured** from Phase 1's real 1D results (mean fraction of
  the oracle-vs-no-remediation nDCG@10 gap that uniform patch actually
  closed across all 5 ratios -- computed directly from
  `remediation_metrics.csv`, not assumed from the literature).
- `phase2_proposed/configs/default.yaml`: identical
  corpus/embedding_models/contamination/traffic/index/budget/detection/
  remediation to `phase1_baseline/configs/default.yaml` on purpose (hits
  the same cached benchmark, no re-embedding) -- only `scheduler:` and
  `paths` differ.
- `phase2_proposed/scripts/run_scheduler.py`: full-scale schedule (every
  ratio x budget level, production greedy) + a three-way optimality
  comparison (naive greedy vs. frontier greedy vs. exact) on a tractable
  300-document subsample.

**Finding #1 -- the primary config's optimality gap is genuinely ~0%, for
an explainable structural reason, not because the problem is trivial.**
`Benefit(i, action) = severity(i) * traffic(i) * recovery_factor(action)`
means severity/traffic always cancel out of a same-document, cross-action
efficiency (benefit/cost) comparison -- so under this formula, whichever
action has the better recovery/cost ratio is uniformly better for EVERY
document, all at once. With the empirically-measured recovery_patch=0.064,
patch's efficiency (0.064/0.1=0.64) is uniformly WORSE than full
re-embed's (1.0/1.0=1.0), so patch is essentially never worth choosing
(confirmed: 0 patches chosen in 28 of 30 ratio x budget combinations, 5 or
fewer in the other 2, all leftover-budget scraps). The MCKP degenerates
into a uniform-cost 0/1 knapsack, for which greedy-by-value is *provably*
optimal -- hence ~0% gap. Verified this wasn't a bug: `exact_benefit >=
greedy_benefit` held in every single row of `scheduler_optimality_gap.csv`
(differences were float noise ~1e-10).

**Finding #2 -- a supplementary stress-test config exposed a real 88%
optimality gap in the naive greedy, from a genuine, well-understood flaw.**
Built `configs/balanced_actions_stress_test.yaml` (recovery_patch=0.12,
tuned so patch's efficiency, 1.2, is only slightly better than full
re-embed's, 1.0, rather than uniformly dominated) specifically to break
the degenerate case above and exercise real three-way tradeoffs. Result:
`solve_greedy_naive` (flat "sort every candidate by ratio, take
greedily") landed at a constant **88.00% optimality gap across every
budget level** -- not noise, a real structural failure. Diagnosed: because
patch's ratio beats full's for EVERY document uniformly, the naive greedy
grabs the cheap patch candidate for a document THE MOMENT it's affordable
and never reconsiders -- even when abundant leftover budget would make
switching that same document to the pricier, more valuable full re-embed
strictly better (full always gives MORE benefit per document than patch,
1.0 vs. 0.12 -- only the RATIO favors patch, not the absolute value). This
is a known, textbook flaw of flat ratio-sort-with-first-fit for
knapsack-style problems: ratio only matters under genuine scarcity;
"first-fit and never look back" can't recognize when scarcity isn't
actually binding.

**Response -- implemented the standard, textbook-correct MCKP greedy**
(`build_efficient_frontier_segments` + the new `solve_greedy`): for each
document, prune any choice dominated on its LP-relaxation "efficient
frontier" (patch is only a genuine intermediate stop if its OWN ratio from
the origin exceeds full's -- geometrically, its point must sit above the
straight line from defer to full re-embed on a cost-vs-benefit plot),
then rank the resulting INCREMENTAL upgrade steps (defer->patch,
patch->full_reembed) by incremental ratio globally, applying them in
order per document. This is provably optimal for the LP relaxation of the
MCKP, with any remaining gap possible only at the single item where
budget runs out mid-upgrade. Result on the SAME stress-test instance:
**mean gap dropped from 88.00% to 0.00% (max 0.14%)** -- essentially
matching the exact solver. On the primary config, frontier and naive
greedy are identical (~0% either way), confirming no regression.

Kept `solve_greedy_naive` in the codebase (not deleted) specifically so
this before/after comparison is reproducible and inspectable, not just
asserted -- `scheduler_optimality_gap.csv` under both config's results
folders has the full naive-vs-frontier-vs-exact numbers per ratio/budget,
and `scheduler_optimality_gap.png` plots the mean gap comparison directly.

**Decisions worth remembering:**
- **Severity comes from the detector's continuous score, not oracle
  ground truth** -- min-max normalized over the flagged set only,
  consistent with Phase 1's "only what detection actually found"
  epistemics (see 1C/1D). The MCKP's candidate pool is the SAME flagged
  set Phase 1's FIFO baseline works from, for a fair 2C comparison later.
- **recovery_patch is empirically calibrated from Phase 1's own real
  measurements, not a literature assumption** -- and that calibration
  choice directly produced Finding #1 (patch degenerately unused in the
  primary config). This is judged correct and left as-is: an optimizer
  that has real cost/benefit information should NOT invest budget in an
  action that real measurement shows barely helps, unlike Phase 1's
  uniform-patch baseline, which patches everyone regardless (see 1D).
  That said, this specific formula structure (severity*traffic canceling
  out of cross-action ratios) means the "one action always uniformly wins"
  degeneracy will recur for ANY single pair of global recovery constants
  where one dominates the other -- worth remembering if recovery factors
  are recalibrated later.
- **The stress-test config exists specifically to exercise the solver
  code under genuine three-way tension**, the same pattern as 1C's
  bge-small stress test: an honest primary result plus a clearly-labeled,
  separately-run supplementary config that stresses a different part of
  the system, rather than tuning the primary config to look more
  interesting.
- **PULP_CBC_CMD deprecation warning** (pulp 3.3.2, bundled solver) is
  cosmetic -- functionality confirmed correct via extensive testing. A
  future pulp major-version bump may require switching to `COIN_CMD` with
  the `pulp[cbc]` extra; not worth doing pre-emptively.

**Status:** MCKP scheduler complete for both the primary and stress-test
configs, with a genuine algorithmic improvement (naive -> frontier greedy)
measured and documented, not just implemented. Next: 2B (migrate + verify
loop) or 2C (head-to-head comparison against Phase 1), per direction.

---

## 2026-08-25 -- Sub-phase 2B: Migrate + Verify loop

**Built:**
- `phase2_proposed/src/phase2/migrate.py`: `build_shadow_embeddings()`
  (applies a scheduler plan to a COPY of the current embeddings -- never
  mutates the live state, modeling a real shadow-index build),
  `select_golden_query_indices()` (top-N eval queries by traffic weight,
  the canary set), and `execute_migration()` (shadow-build -> verify
  golden-query nDCG@10 doesn't regress -> atomic swap-or-abort, all-or-
  nothing).
- `common/config.py`'s new `MigrationConfig`: `golden_query_count=20`,
  `regression_tolerance=0.0` (strict no-regression, per the brief's own
  wording).
- `phase2_proposed/scripts/run_migration.py`: runs the full loop across
  every ratio x budget level using 2A's production frontier-greedy
  schedule, records the verification outcome AND the resulting full-corpus
  quality (on whatever actually ends up live), saved to
  `migration_metrics.csv` -- the artifact 2C will read Phase 2's numbers
  from.

**Real run, primary config: 10 of 30 ratio x budget combinations aborted
verification.** Diagnosed two specific, surprising-looking cases before
accepting this number (per the established "flag anything that looks off"
discipline):

1. **Why does fixing MORE contamination sometimes cause the golden set to
   regress?** Traced one concrete case (ratio=0.50, budget=1.00) query by
   query. Real mechanism, confirmed directly: fixing document A (via full
   re-embed) can shift A's embedding to be MORE similar to an UNRELATED
   golden query than it was before -- displacing that query's actual
   relevant document out of the top-10 entirely (one golden query's
   relevant doc, previously ranked 9th, disappeared from the top-10
   completely after the fix; a different golden query improved from
   ndcg=0.0 to 0.63 in the same run). This is real and defensible, not a
   bug: BEIR's relevance judgments are famously sparse/incomplete, so it's
   entirely plausible a "fixed" document is genuinely topically related to
   an unjudged query and legitimately starts competing for its top-10 slot.
   The scheduler's benefit function only ever reasons about each document
   IN ISOLATION (severity x traffic x recovery) -- it has no way to see
   this kind of cross-document ranking interaction. That gap between
   "planned in isolation" and "verified in the actual index" is exactly
   what the migrate+verify loop exists to catch.
2. **Why does budget=1.00 abort at ratio=0.50 when budget=0.75 didn't, and
   why does ratio=0.05 abort at EVERY budget level?** Checked what the
   shadow's FULL-corpus (all ~300 eval queries, not just the 20 golden
   ones) quality would have been had it NOT been rolled back:
   ratio=0.05: +0.0092 nDCG@10 (would-be improvement); ratio=0.50,
   budget=1.00: **+0.1173** nDCG@10 -- a large, genuine improvement on the
   broad metric. The strict, zero-tolerance golden-query gate is
   deliberately conservative: it protects the curated high-value query set
   even at the cost of forgoing a substantial AGGREGATE win, because a
   handful of individual golden queries (out of only 20) showed a small
   regression. This is a faithful, direct consequence of implementing the
   brief's own specification ("gated on golden-query verification not
   regressing quality") literally and strictly -- not a bug, and not
   loosened after the fact to produce nicer-looking numbers. It is,
   however, a real and reportable tension in verification-gated deployment
   design: a small canary set is more sensitive to individual-query noise
   than a broad metric, which is exactly why real production canary
   analysis often uses a tolerance band or statistical significance test
   rather than a hard zero-tolerance gate -- `MigrationConfig.regression_tolerance`
   is already a named, adjustable constant for exactly this reason, kept
   at the brief's literal 0.0 for this primary run.

**Reading the plot:** `migration_quality_by_budget.png` marks aborted
points in red. At ratio=0.50 the LIVE quality visibly DROPS at budget=1.00
relative to budget=0.75 -- this is not the migration getting worse with
more budget; it's the gate correctly refusing to deploy a plan that would
have hurt the golden set, even though it would have substantially helped
the broader corpus. Worth narrating explicitly in any write-up of this
chart, since the shape is counter-intuitive without that context.

**Decisions worth remembering:**
- **Golden queries are the top-N by traffic weight** (N=20), not a random
  sample -- modeling a real canary set an ops team would actually choose
  to protect (their highest-value queries), consistent with this
  project's traffic-simulation thread throughout.
- **The swap is all-or-nothing** (`live_embeddings` is either the full
  shadow or the exact original, never a partial mix) -- matches a real
  atomic alias-swap, which by definition has no in-between state.
- **`run_migration.py` evaluates full-corpus quality on `report.live_embeddings`**,
  i.e. on whatever ACTUALLY ends up live after the gate -- not on the
  shadow unconditionally. This is why the primary quality curve
  (`migration_quality_by_budget.png`) is the honest, deployed-quality
  story for 2C to compare against Phase 1, even though it understates what
  the scheduler's plan alone could have achieved.

**Status:** migrate+verify loop complete and run for real; abort mechanism
validated with a constructed unit test (`test_execute_migration_aborts_and_rolls_back_when_shadow_regresses`)
AND observed firing organically on real data with a diagnosed cause. Next:
2C (head-to-head comparison against Phase 1's baselines, using
`migration_metrics.csv` alongside `phase1_baseline/results/remediation_metrics.csv`).

---

## 2026-08-25 -- Sub-phase 2C: Head-to-head comparison vs. Phase 1

**Built:** `phase2_proposed/scripts/run_comparison.py` -- combines
`phase1_baseline/results/remediation_metrics.csv` and
`phase2_proposed/results/migration_metrics.csv` (identical
ratio/budget_level/ndcg@10/... schema, no reshaping needed) into one long
dataframe and one chart per metric
(`comparison_quality_by_budget_{unweighted,traffic_weighted}.png`), plus
`comparison_improvement.csv` quantifying Phase 2 vs. the BETTER of Phase
1's two baselines at every matched (ratio, budget_level) pair. Phase 2's
numbers are read exactly as `migration_metrics.csv` reports them -- the
VERIFIED, actually-deployed quality including every 2B verification abort
-- not an idealized unconstrained-scheduler number.

**Real result -- honest and two-sided, not a uniform win:**
- Phase 2 beat both Phase 1 baselines in **18 of 30** ratio x budget
  combinations; underperformed both in the other **10** -- and those 10
  are EXACTLY the 10 combinations 2B's verification gate aborted (checked
  directly: every negative row in `comparison_improvement.csv` corresponds
  to an aborted row in `migration_metrics.csv`). Nothing about the
  comparison script hides this -- `abs_improvement` is reported per row,
  not just as a favorable aggregate.
- **Where Phase 2 wins, it wins decisively, and specifically at LOW
  budgets** -- exactly the constrained-budget regime this whole project
  is about. At ratio=0.50, budget=0.10 (10% of the full-reembed-equivalent
  budget): Phase 2 recovers **+0.108 nDCG@10** more than the better Phase
  1 baseline, **42.5% of the entire oracle-vs-no-remediation headroom**,
  from a tenth of the budget. At ratio=0.20, budget=0.10: **+45.0% of
  headroom**. Mean improvement across all 30 combinations: **+0.0123
  nDCG@10 (+3.6% of headroom)** -- a modest-looking aggregate specifically
  because it averages in the 10 abort cases; the real story is in the
  per-budget breakdown, not the mean.
- **Budget needed to reach 50% of the oracle-vs-no-remediation headroom**
  (the brief's "Y% less budget for the same quality" framing), per ratio:
  ratio=0.20 -- FIFO needs the ENTIRE budget (1.0) to reach it; Phase 2
  reaches it at budget=0.1, a **10x** reduction. ratio=0.10 -- FIFO needs
  0.75, Phase 2 needs 0.1 (**7.5x**). ratio=0.50 -- FIFO never reaches the
  target within the swept budget range at all; Phase 2 reaches it at 0.1.
  ratio=0.05 -- NEITHER reaches it comfortably (both start near the
  oracle ceiling already, since only 5% of the corpus is contaminated);
  Phase 2 specifically never reaches it because ratio=0.05 aborts
  verification at every single budget level (see 2B's entry) -- an honest
  consequence of the gate, not a separate new finding.
- **Where Phase 2 loses, it's exactly where the conservative gate declines
  a real net win**: at ratio=0.50/budget=1.0 and ratio=0.20/budget=0.75-1.0
  and ratio=0.35/budget=1.0, Phase 2's LIVE quality drops below FIFO's
  because verification aborted a shadow that (per 2B's diagnosis) would
  have substantially improved the full-corpus metric. This is the same
  safety-vs-aggregate-improvement tension documented in 2B, now visible
  directly in the head-to-head chart as green dropping below blue at high
  budgets in `comparison_quality_by_budget_unweighted.png`.

**Reading the chart:** in `comparison_quality_by_budget_unweighted.png`,
Phase 2 (green) starts ABOVE Phase 1's FIFO (blue) at low budgets in 4 of
5 ratios -- the core value proposition, prioritization pays off most when
budget is scarce -- then in ratio=0.20 and ratio=0.50 specifically, green
visibly dips below blue at high budgets, exactly where verification
aborted. Both effects are real and both are shown, not just the favorable
one.

**Decisions worth remembering:**
- **Quantification uses `pct_of_headroom`** (improvement as a fraction of
  oracle-vs-no-remediation quality range), not raw nDCG delta alone --
  raw deltas aren't comparable across ratios with different available
  headroom (e.g. ratio=0.05 has only ~0.018 headroom total, so even a
  "small" 0.009 delta there is actually huge in relative terms). Both are
  reported in `comparison_improvement.csv` for anyone who wants the raw
  numbers too.
- **Did not cherry-pick a favorable budget level or hide the loss cases**
  to present a cleaner "Phase 2 always wins" story. The mean-improvement
  number is REPORTED alongside an explicit note that it understates the
  low-budget win because it's dragged down by the 10 abort cases --
  precisely so a reader isn't misled by the aggregate alone.

**Status:** head-to-head comparison complete, with an honest, two-sided
result: decisive wins at low-to-moderate budgets (the realistic,
budget-constrained scenario), real losses at a few high-budget points
where the safety gate correctly declines to deploy despite the
scheduler's plan. Next: 2D (autonomous Live Sentry demo) or 2E (final
results package), per direction.

---

## 2026-08-25 -- Sub-phase 2D: Autonomous "Live Sentry" demo

**Built:**
- `phase2_proposed/src/phase2/live_sentry.py`: the engine, with NO
  Streamlit dependency (plain Python + a `threading.Lock`-protected
  `LiveSentryState`), so it's directly unit-testable. `build_context()`
  loads the benchmark/patch transform/query embeddings once;
  `LiveSentryState` holds Database A (a permanent copy of `emb_a`, never
  touched) and Database B (mutable); `inject_contamination()` is the ONE
  manual control; `run_one_tick()` is one full detect -> schedule ->
  migrate/verify cycle, safe to call repeatedly from a background thread;
  `monitor_loop()` is the actual thread target.
- `phase2_proposed/live_sentry_dashboard.py`: Streamlit UI. A background
  `threading.Thread` (guarded by `st.session_state` so it starts exactly
  once per session) runs `monitor_loop` against a shared `LiveSentryState`;
  the page itself just renders a snapshot of that state and calls
  `time.sleep(1.5); st.rerun()` to redraw on a timer, independent of the
  monitor's own poll interval.

**Design choice -- calibrated alert threshold, not an assumed ratio.**
1C/2A's detectors threshold by an ASSUMED contamination ratio (a real
operator's rough prevalence estimate). A continuous live monitor doesn't
have that -- it doesn't know in advance how big the next incident is, or
if there even is one. Instead: take Database A's (known-clean) KNN-OOD
score distribution as the "normal" baseline, and alert on any Database B
document scoring above a high percentile (99th) of it. This is standard
anomaly-monitoring practice (calibrate normal, alert on deviation) and is
exactly what having a stable control database is FOR.

**Verified in a real browser across multiple full runs, not just unit
tested** -- and three real, non-obvious bugs were found and fixed this
way before the demo could be trusted, each confirmed both by a standalone
diagnostic script AND a regression test:

1. **Persistent false positives starved genuinely-contaminated documents
   of budget forever.** First real run: flagged count froze at a constant
   73 for 9+ consecutive ticks despite "fixing" ~21 documents every
   cycle. Diagnosed by comparing the flagged SET (not just the count)
   tick-to-tick: it was completely static (zero new, zero gone). Root
   cause: a 99th-percentile threshold flags ~1% of ANY similarly-
   distributed sample by construction, including Database B's genuinely-
   clean majority -- these persistent false positives have benefit=0 in
   the MCKP (traffic=0 for ~87% of the corpus, per 1B), tie with each
   other, and Python's stable sort always breaks that tie the same way,
   so the scheduler kept re-selecting the SAME already-correct (no-op)
   documents every tick and never reached the genuinely-contaminated ones
   sitting later in index order. Fix: track `db_b_remediated_once` (real
   operational history -- which documents have already had a passed
   remediation applied -- not oracle ground truth) and exclude them from
   future scheduling rounds. A persistent false positive can't be fixed by
   re-fixing it again; re-attempting is pure waste.
2. **Resolution required literally zero flagged documents, which a
   percentile-threshold detector can never produce.** Related to #1: even
   after fix #1, the incident could never reach `flagged_b == 0`, because
   Database A's OWN re-scored baseline also always has ~50 flagged
   documents (self-referential percentile). Fix: resolution is relative to
   Database A's concurrently-measured baseline count
   (`flagged_b <= flagged_a`), not to absolute zero -- what a real
   anomaly monitor actually means by "back to nominal."
3. **Per-tick budget shrank to zero as the candidate pool shrank, stalling
   just short of resolution.** After fix #1, a real run still stalled: `budget_units
   = budget_level * n_candidates` meant that as `n_candidates` fell (down
   to 3), the ABSOLUTE budget (0.3 x 3 = 0.9) dropped below the cost of
   even one fix (1.0) -- "Plan: 0 full re-embeds," verification trivially
   "passing" on a no-op, forever. Fix: size `budget_units` off
   `incident_initial_flagged_count`, a STABLE reference captured once when
   the incident is first detected, not the shrinking remaining pool.
4. **Escalating budget on verification failure can be a no-op, causing an
   infinite stuck loop at maximum budget.** A later real run: the exact
   same 14-15 document candidate set kept producing the exact same
   golden-query regression at EVERY budget level from 0.3 up through 1.0
   (once budget already covers the whole candidate set, which it did from
   the start, more budget changes nothing about the plan) -- this is 2B's
   already-documented "verification can correctly block a plan" finding,
   now manifesting as a permanent stall rather than a one-shot data point.
   Fix: track consecutive verification failures; after
   `MAX_CONSECUTIVE_FAILURES` (3) failures AT maximum budget for the
   identical candidate set, give up and defer those documents (excluded
   from future scheduling, distinctly logged as needing manual review, not
   silently dropped) -- exactly what a real automated pipeline should do
   rather than retry an identical failing plan forever. Also fixed a
   smaller, related display bug found in the same pass: `last_db_b_flagged`
   was hardcoded to 0 whenever no incident was active, misrepresenting a
   resolved (but not perfectly zero-noise) database as having no anomalies
   at all -- now scores Database B for real in that branch too.

**A complete, clean real run** (after all four fixes), captured from the
live event log: injection at T+0 (500 of 5000 documents) -> ALERT fires
autonomously at T+8s (92 flagged) -> scheduler runs every ~5-8s, steady
progress (92 -> 67 -> 40 -> 14 -> 1) -> alert clears itself at T+44s. A
separate run exercised the give-up path end-to-end live: budget escalated
0.30 -> 0.50 -> 0.70 -> 0.90 -> 1.00 over 5 consecutive failures on the
same 15-document candidate set, gave up after the 3rd failure at max
budget, deferred those 15, and resolved with an honest UI warning ("15
document(s) deferred for manual review -- verification repeatedly blocked
their fix even at maximum budget") rather than silently declaring victory.

**Decisions worth remembering:**
- **All four bugs above were found by actually running the full-scale
  (5000-doc) dashboard in a browser and watching it over multiple real
  minutes**, not just from unit tests on a tiny synthetic case -- the
  40-document synthetic fixture in `test_live_sentry.py` (generous budget,
  single tick resolution) never exercised the shrinking-budget or
  give-up-after-repeated-failure code paths at all. Two new tests
  (`test_incident_resolves_with_a_constrained_per_tick_budget`,
  `test_repeated_verification_failure_defers_instead_of_looping_forever`)
  were added specifically to give these paths real regression coverage
  going forward, using a smaller budget and an impossible
  `regression_tolerance` respectively to force the relevant code path
  deterministically rather than relying on chance.
- **The stretch goal (a third "Database C -- healed via Phase 1 baseline"
  panel) was not attempted this sub-phase.** The brief marks it optional
  ("only attempt after 2A-2D are solid"), and getting the core two-database
  loop genuinely correct -- not just superficially working -- consumed the
  available effort for this sub-phase. Revisit if there's time after 2E.
- **UI auto-refresh via `time.sleep(1.5); st.rerun()`** is a plain-
  Streamlit pattern requiring no extra dependency, decoupled from the
  monitor's own poll interval (the background thread ticks on its own
  schedule; the page just redraws from whatever the shared state currently
  says).

**Status:** Live Sentry demo built, and -- critically -- actually run to
completion multiple times in a real browser with real detection/scheduling/
migration on the full 5000-document benchmark, not just imported
successfully. All 4 discovered issues fixed and regression-tested. Next:
2E (final results package).

---

## 2026-08-25 -- Sub-phase 2E: Final results package

**Built:** `phase2_proposed/results/README.md` -- a comprehensive index of
every chart/table/number from 1C-1D and 2A-2C, plus headline numbers for
quick reference and the reproduction commands for all of it. Copied
Phase 1's key artifacts (`detection_metrics.csv`,
`detection_f1_by_ratio.png`, `remediation_metrics.csv`,
`remediation_quality_by_budget_*.png`, and the `aniso_stress_bge/` stress
test) into `phase2_proposed/results/from_phase1_baseline/`, so everything
needed for Phase 2's paperwork lives in one folder as the brief asks --
`phase1_baseline/results/` remains the canonical copy for Phase 1's own
review and is unaffected.

**Decision:** copied rather than symlinked or merely referenced, since the
brief's stated purpose ("assemble the report quickly... without
re-running experiments under time pressure") implies the paperwork writer
shouldn't need to hop between two folders later. Documented the
regeneration/re-copy step in the new README so this doesn't silently go
stale if Phase 1's results are ever regenerated.

**Status:** Phase 2 (2A-2E) complete. `phase2_proposed/results/` is the
single, indexed entry point for all of Phase 1 and Phase 2's real,
already-computed results. Remaining optional item: the 2D stretch goal
(a third "Database C -- healed via Phase 1 baseline" panel), explicitly
marked optional in the brief and deferred -- see 2D's entry for why.
