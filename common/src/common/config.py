"""Typed configuration system shared by Phase 1 and Phase 2.

Design decision: we use plain dataclasses (not pydantic) so the config layer
has zero extra runtime dependency beyond PyYAML, and so every field's type
and default is visible in one place for a viva/examiner walkthrough. YAML is
chosen over hardcoding because the project explicitly needs contamination
ratios, budgets, and model names to be adjustable per-experiment without
touching source -- e.g. running the full ratio sweep (5/10/20/35/50%) is
just a config edit, not a code edit.

Usage:
    from common.config import load_config
    cfg = load_config("phase1_baseline/configs/default.yaml")
    print(cfg.embedding_models.generation_a)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass
class CorpusConfig:
    # BEIR dataset name, loaded via its HuggingFace mirror (see common/data.py).
    # We avoid the official `beir` pip package because its pinned dependency
    # versions (old transformers/torch) conflict with sentence-transformers;
    # the HF-mirrored datasets (e.g. "BeIR/scifact") give the same documents
    # without that dependency risk.
    name: str = "scifact"
    # Cap corpus size so the whole pipeline stays laptop/free-Colab feasible.
    max_docs: int = 5000


@dataclasses.dataclass
class EmbeddingModelsConfig:
    # "Generation A" -- the original embedding model an index was built with.
    generation_a: str = "sentence-transformers/all-MiniLM-L6-v2"
    # "Generation B" -- a different real model simulating a provider update.
    generation_b: str = "BAAI/bge-small-en-v1.5"


@dataclasses.dataclass
class ContaminationConfig:
    # Fraction of the corpus re-embedded with generation_b, swept across
    # these values so detection/remediation results aren't a single lucky
    # (or unlucky) operating point.
    ratios: list[float] = dataclasses.field(
        default_factory=lambda: [0.05, 0.10, 0.20, 0.35, 0.50]
    )


@dataclasses.dataclass
class TrafficConfig:
    # Zipfian query traffic: a small number of documents receive most of the
    # simulated search volume, matching real production traffic shape and
    # making WHICH documents are contaminated (not just how many) matter --
    # this is exactly the signal Phase 2's prioritizer needs to exploit.
    distribution: str = "zipf"
    # Skew parameter; higher = more concentrated on head queries. 1.0 is the
    # classic Zipf exponent reported for real web/enterprise query logs. The
    # original 1.2 concentrated 37% of all eval weight on ONE query
    # (effective sample size ~6.5 of 300 eval queries), which made the
    # traffic-weighted metric mostly noise -- see DEVLOG.md's "Redesign"
    # entry. 1.0 gives an effective sample size of ~15.
    zipf_s: float = 1.0
    # How a DOCUMENT's traffic is derived from the query log:
    #   "impressions" -- realistic: how often (and how high) each document
    #       was actually SHOWN in the top-`impression_depth` results of the
    #       logged queries, on the pre-incident (clean, generation-A) index,
    #       with a standard 1/log2(rank+2) position discount. This is what a
    #       real search system's impression logs record, and it needs no
    #       relevance labels at all.
    #   "qrels" -- the original Phase 1 construction: a document's traffic is
    #       the summed weight of every query that LABELS it relevant. Kept
    #       for reproducing the old numbers, but it routes evaluation
    #       relevance labels into the scheduler (a label leak) and gives 87%
    #       of the corpus exactly zero traffic.
    doc_signal: str = "impressions"
    impression_depth: int = 10


@dataclasses.dataclass
class IndexConfig:
    backend: str = "faiss"  # "faiss" or "hnswlib"
    metric: str = "cosine"


@dataclasses.dataclass
class BudgetConfig:
    # Budget expressed as a fraction of contaminated documents that could be
    # fully re-embedded, so it stays meaningful across corpus sizes and
    # contamination ratios. Phase 1's FIFO/uniform-patch and Phase 2's MCKP
    # scheduler are all evaluated at the same budget levels for a fair
    # head-to-head comparison in 2C.
    levels: list[float] = dataclasses.field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    )


@dataclasses.dataclass
class PathsConfig:
    results_dir: str = "phase1_baseline/results"
    cache_dir: str = ".cache"


@dataclasses.dataclass
class DetectionConfig:
    # Neighborhood sizes for the two distance/density-based baseline
    # detectors (1C). Isolation Forest needs no neighborhood size -- it
    # splits on random feature thresholds instead.
    knn_k: int = 10  # KNN-OOD: anomaly score = cosine distance to the k-th nearest neighbor
    lof_k: int = 20  # Local Outlier Factor neighborhood size (matches sklearn's own default)


@dataclasses.dataclass
class RemediationConfig:
    # Costs are in the same "full-reembed-equivalent" units BudgetConfig.levels
    # is expressed in, so budget_units = budget_level * n_flagged * cost_full_reembed.
    cost_full_reembed: float = 1.0
    cost_patch: float = 0.1  # patch = one cheap linear transform application, ~10x cheaper
    # Size of the "probe batch" used to fit the patch transform (see
    # common/patch.py) -- independent of any contamination ratio.
    calibration_sample_size: int = 300
    # Which adapter `common.patch.fit_adapter` fits on the probe batch for the
    # "patch" action (shared by Phase 1's uniform patch and Phase 2's MCKP, so
    # neither phase gets a better fix than the other):
    #   "ridge" -- affine least-squares map B -> A with an L2 penalty. Measured
    #       on the probe batch by query-log replay it keeps ~87% of a patched
    #       document's search impressions, vs ~16% for Procrustes.
    #   "procrustes" -- the original orthogonal rotation (norm-preserving, but
    #       a pure rotation cannot model the different centroids of the two
    #       models' spaces).
    adapter: str = "ridge"
    ridge_alpha: float = 1.0
    # Which 1C detector's flagged set feeds remediation. KNN-OOD is the
    # strongest baseline (see DEVLOG.md's sub-phase 1C entry), so it's the
    # one a real operator would actually deploy.
    detector: str = "knn_ood"


@dataclasses.dataclass
class FingerprintConfig:
    # Phase 2's generation-fingerprint detector (phase2/fingerprint.py): each
    # stored vector is compared against the probe batch's vectors from BOTH
    # generations, and P(generation B) is the smoothed fraction of its `k`
    # nearest probe vectors that came from generation B. No assumed
    # contamination ratio is needed -- a document is flagged when that
    # probability exceeds `threshold`.
    k: int = 10
    threshold: float = 0.5


@dataclasses.dataclass
class SchedulerConfig:
    # Phase 2's MCKP benefit function: Benefit(i, action) = severity(i) *
    # traffic(i) * recovery_factor(action). recovery_full_reembed is exactly
    # 1.0 by construction (that action literally substitutes the document's
    # true generation-A embedding). recovery_patch is NOT a literature
    # assumption -- it's the empirically measured average fraction of the
    # oracle-vs-no-remediation nDCG@10 gap that uniform patch actually closed
    # in 1D's real measurements (mean 0.064 across all 5 ratios; see
    # DEVLOG.md's sub-phase 2A entry for the derivation). recovery_defer is
    # 0.0 by construction (leaves the document untouched).
    recovery_full_reembed: float = 1.0
    recovery_patch: float = 0.064
    recovery_defer: float = 0.0
    # Which 1C detector's continuous anomaly score becomes contamination
    # severity(i) -- same detector remediation.detector uses, for
    # consistency (the flagged SET and the severity SCORE come from the
    # same model, not two different oracles of confidence).
    severity_detector: str = "knn_ood"
    # Where Phase 2's per-action effect estimates come from:
    #   "replay" -- estimated label-free from the probe batch by replaying the
    #       query log (phase2/calibration.py): how much of a patched
    #       document's search impressions the adapter restores, and how much
    #       damage patching an already-clean document does.
    #   "config" -- use recovery_patch above as a fixed constant (the original
    #       2A behaviour, kept for the stress-test config and old results).
    recovery_source: str = "replay"
    # Small traffic prior added to every document, as a fraction of the
    # smallest non-zero document traffic. Documents no logged query has shown
    # yet still get fixed once budget allows (future queries may need them),
    # but always after every document with observed traffic.
    traffic_floor_fraction: float = 0.001
    # Size of the subsample used to run the exact MCKP solver (PuLP/CBC) and
    # report its optimality gap against the greedy heuristic -- the exact
    # solver is run on a tractable instance, not the full flagged set (see
    # DEVLOG.md's sub-phase 2A entry for why, and the measured gap).
    exact_solver_sample_size: int = 300


@dataclasses.dataclass
class MigrationConfig:
    # Sub-phase 2B: the "golden query" canary set used to gate a migration
    # cutover -- the highest-traffic eval queries, modeling the real
    # practice of checking a small, curated, high-value query set before a
    # deploy rather than a full (slower) evaluation suite.
    golden_query_count: int = 20
    # How much golden-query nDCG@10 is allowed to drop and still pass
    # verification. 0.0 = strict no-regression (the brief's own wording);
    # kept as a named, adjustable constant rather than hardcoded so a
    # small floating-point-noise allowance can be introduced later without
    # touching migrate.py.
    regression_tolerance: float = 0.0
    # Staged rollout (phase2.migrate.execute_staged_migration): the plan is
    # applied in `n_waves` priority-ordered waves, each verified separately,
    # so one bad wave is rolled back alone instead of discarding the whole
    # plan (the original all-or-nothing gate aborted 10 of 30 sweeps).
    n_waves: int = 5
    # Non-inferiority gate (phase2.migrate.wave_is_unsafe): a wave ships only
    # if a paired bootstrap over the golden queries' per-query nDCG deltas
    # shows, with `gate_confidence`, that the canary is NOT worse than the
    # best state reached so far by more than `regression_tolerance`. Set
    # regression_tolerance to a small positive margin (e.g. 0.01) for staged
    # rollouts: with 0.0 any single query losing a rank blocks a wave.
    gate_confidence: float = 0.95
    bootstrap_samples: int = 2000
    # A rejected wave is split in half and each half retried, this many
    # levels deep, so a few harmful fixes do not sink an otherwise good wave.
    bisect_depth: int = 2
    # Which per-query score gates each wave (phase2.migrate):
    #   "replay" -- shadow-traffic result diffing: the `canary_queries` most
    #       popular logged NON-evaluation queries, scored by how much of their
    #       pre-incident logged top-10 a candidate index still returns. Label
    #       free, so it can be large enough to see harm anywhere in the index.
    #   "labelled" -- nDCG@10 on `golden_query_count` train-split queries
    #       with relevance labels (the original, low-coverage canary).
    canary: str = "replay"
    canary_queries: int = 300
    # After a staged rollout, budget from rolled-back fixes is recycled into
    # a re-plan (rejected patches compete for full re-embeds instead), up to
    # this many rounds in total -- see phase2.pipeline.remediate.
    recycle_rounds: int = 3


@dataclasses.dataclass
class LiveSentryConfig:
    # Sub-phase 2D: the autonomous two-database live monitor.
    poll_interval_seconds: float = 4.0  # how often the background thread re-scores both databases
    injection_batch_fraction: float = 0.10  # fraction of the corpus "re-embedded" per simulated provider update
    # Alert threshold is CALIBRATED, not assumed: the background loop takes
    # Database A's (known-clean) KNN-OOD score distribution as the normal
    # baseline and flags any Database B document scoring above this
    # percentile of it -- a real anomaly-monitoring pattern (calibrate
    # normal, alert on deviation), and a deliberately different design
    # from 1C/2A's ratio-based thresholding, which assumes the operator
    # already has a prevalence estimate -- not available in a live,
    # continuous-monitoring context. See DEVLOG.md's sub-phase 2D entry.
    alert_threshold_percentile: float = 99.0
    # Budget (as a fraction of that tick's flagged count, same units as
    # RemediationConfig) spent on the first remediation attempt for an
    # incident; escalates by this increment on each tick verification
    # fails, up to 1.0 -- modeling a real incident-response escalation
    # (try a modest fix first, throw more budget at it if that's not safe
    # enough to ship) rather than retrying an identical failing plan
    # forever.
    initial_budget_level: float = 0.3
    budget_escalation_step: float = 0.2


@dataclasses.dataclass
class Config:
    seed: int = 42
    corpus: CorpusConfig = dataclasses.field(default_factory=CorpusConfig)
    embedding_models: EmbeddingModelsConfig = dataclasses.field(
        default_factory=EmbeddingModelsConfig
    )
    contamination: ContaminationConfig = dataclasses.field(
        default_factory=ContaminationConfig
    )
    traffic: TrafficConfig = dataclasses.field(default_factory=TrafficConfig)
    index: IndexConfig = dataclasses.field(default_factory=IndexConfig)
    budget: BudgetConfig = dataclasses.field(default_factory=BudgetConfig)
    detection: DetectionConfig = dataclasses.field(default_factory=DetectionConfig)
    remediation: RemediationConfig = dataclasses.field(default_factory=RemediationConfig)
    scheduler: SchedulerConfig = dataclasses.field(default_factory=SchedulerConfig)
    migration: MigrationConfig = dataclasses.field(default_factory=MigrationConfig)
    fingerprint: FingerprintConfig = dataclasses.field(default_factory=FingerprintConfig)
    live_sentry: LiveSentryConfig = dataclasses.field(default_factory=LiveSentryConfig)
    paths: PathsConfig = dataclasses.field(default_factory=PathsConfig)


# Maps each nested dataclass field name to its dataclass type, so
# `_build_dataclass` knows how to recurse into nested YAML mappings.
_NESTED_FIELDS: dict[str, type] = {
    "corpus": CorpusConfig,
    "embedding_models": EmbeddingModelsConfig,
    "contamination": ContaminationConfig,
    "traffic": TrafficConfig,
    "index": IndexConfig,
    "budget": BudgetConfig,
    "detection": DetectionConfig,
    "remediation": RemediationConfig,
    "scheduler": SchedulerConfig,
    "migration": MigrationConfig,
    "fingerprint": FingerprintConfig,
    "live_sentry": LiveSentryConfig,
    "paths": PathsConfig,
}


def _build_dataclass(raw: dict[str, Any]) -> Config:
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _NESTED_FIELDS and isinstance(value, dict):
            kwargs[key] = _NESTED_FIELDS[key](**value)
        else:
            kwargs[key] = value
    return Config(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a validated, typed Config object.

    Any field not present in the YAML falls back to the dataclass default,
    so experiment configs only need to specify what they override.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _build_dataclass(raw)
