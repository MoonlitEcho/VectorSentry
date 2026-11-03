"""VectorSentry on a real vector store: calibrate -> scan -> plan -> (dry-run
| apply through a verified staged rollout) -> record -> rollback.

The decision logic is the research pipeline, reused unchanged:
`phase2.fingerprint` (detection), `phase2.calibration` (patch effect/harm by
query-log replay), `phase2.scheduler` + `phase2.pipeline` (expected-benefit
knapsack, staged rollout, budget recycling). What this module adds is the
production plumbing:

- vectors, texts and IDs come from a `VectorStore`, not a benchmark cache;
- the probe batch is embedded on the spot by the two real models;
- traffic and the safety canary come from a real `QueryLog`;
- a full re-embed calls the serving `Embedder` on the document's text, lazily
  -- only for fixes the rollout actually tries -- and its cost is counted;
- every run is recorded in a `StateStore`, dry-run is the default, and the
  original vectors of every changed document are saved for rollback.

Limitation (stated, not hidden): replay calibration uses the CURRENT index
as its background, which is slightly contaminated during an incident; the
research code had a clean copy to use instead. The estimate is a proxy for
the scheduler's decisions, not a measurement to quote.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence

import numpy as np

from common.config import Config
from common.patch import fit_adapter
from phase2.calibration import estimate_action_effects
from phase2.fingerprint import FingerprintDetector
from phase2.pipeline import Phase2Context, plan_remediation, remediate, traffic_with_floor
from vectorsentry.adapters import Embedder, Snapshot, VectorStore
from vectorsentry.guard import IngestGuard
from vectorsentry.querylog import QueryLog, logged_results_canary
from vectorsentry.state import StateStore

log = logging.getLogger("vectorsentry")


@dataclasses.dataclass
class SentryConfig:
    probe_size: int = 300  # documents dual-embedded to calibrate the detector and the patch
    fingerprint_k: int = 10
    threshold: float = 0.5  # flag when P(suspect model) exceeds this
    adapter: str = "ridge"
    ridge_alpha: float = 1.0
    budget: float = 100.0  # per run, in the serving embedder's cost units
    patch_cost_fraction: float = 0.1  # a patch costs this fraction of a full re-embed
    canary_queries: int = 300
    n_waves: int = 5
    tolerance: float = 0.01  # non-inferiority margin on the canary score
    confidence: float = 0.95
    bootstrap_samples: int = 2000
    bisect_depth: int = 2
    recycle_rounds: int = 3
    traffic_floor_fraction: float = 0.001
    seed: int = 0


@dataclasses.dataclass
class ScanReport:
    snapshot: Snapshot
    p_broken: np.ndarray
    flagged: np.ndarray
    health: float  # canary score of the current index (1.0 = logged results fully reproduced)

    @property
    def n_flagged(self) -> int:
        return int(self.flagged.sum())


@dataclasses.dataclass
class RunReport:
    run_id: int | None
    mode: str
    n_documents: int
    n_flagged: int
    n_full: int
    n_patch: int
    n_rolled_back: int
    cost_planned: float
    embed_cost: float
    health_before: float
    health_after: float
    protected_share: float  # expected share of affected search traffic the plan protects
    actions: list[dict]


class _LazyReembedder:
    """Array-like stand-in for 'the clean vector of every document': indexing
    it with document positions re-embeds those documents' texts with the
    serving model on first access (cached), counting the cost. Lets the
    unchanged rollout code pay only for the fixes it actually tries."""

    def __init__(self, texts: Sequence[str], embedder: Embedder, dim: int):
        self.texts = texts
        self.embedder = embedder
        self.dim = dim
        self.cache: dict[int, np.ndarray] = {}

    def __getitem__(self, idx) -> np.ndarray:
        idx = np.atleast_1d(np.asarray(idx, dtype=int))
        if len(idx) == 0:  # e.g. a re-plan batch with no re-embeds in it
            return np.zeros((0, self.dim), np.float32)
        missing = [int(i) for i in idx if int(i) not in self.cache]
        if missing:
            for i, vec in zip(missing, self.embedder.embed([self.texts[i] for i in missing])):
                self.cache[i] = vec
        return np.stack([self.cache[int(i)] for i in idx])

    @property
    def cost(self) -> float:
        return len(self.cache) * self.embedder.cost_per_document


class VectorSentry:
    def __init__(
        self,
        store: VectorStore,
        serving: Embedder,
        suspect: Embedder,
        query_log: QueryLog,
        state: StateStore | None = None,
        config: SentryConfig | None = None,
        collection: str = "",
    ):
        self.store = store
        self.serving = serving
        self.suspect = suspect
        self.query_log = query_log
        self.state = state or StateStore()
        self.config = config or SentryConfig()
        self.collection = collection
        self._ctx: Phase2Context | None = None
        self._calibrated_ids: list[str] | None = None

    # ---- internal config for the reused research pipeline ----------------
    def _pipeline_config(self) -> Config:
        c, cfg = self.config, Config(seed=self.config.seed)
        full = self.serving.cost_per_document
        cfg.remediation.cost_full_reembed = full
        cfg.remediation.cost_patch = full * c.patch_cost_fraction
        cfg.remediation.adapter, cfg.remediation.ridge_alpha = c.adapter, c.ridge_alpha
        cfg.fingerprint.k, cfg.fingerprint.threshold = c.fingerprint_k, c.threshold
        cfg.migration.n_waves, cfg.migration.regression_tolerance = c.n_waves, c.tolerance
        cfg.migration.gate_confidence, cfg.migration.bootstrap_samples = c.confidence, c.bootstrap_samples
        cfg.migration.bisect_depth, cfg.migration.recycle_rounds = c.bisect_depth, c.recycle_rounds
        return cfg

    def _canary(self, snap: Snapshot):
        top = self.query_log.top(self.config.canary_queries)
        qvecs = self.serving.embed([e.query for e in top])
        return logged_results_canary(qvecs, [e.results for e in top], snap.ids)

    # ---- calibration -------------------------------------------------------
    def calibrate(self, snap: Snapshot | None = None) -> Phase2Context:
        """Embed a probe batch with BOTH models, fit the fingerprint detector
        and the patch adapter on it, and estimate the patch's effect and harm
        by replaying the query log."""
        snap = snap or self.store.fetch_all()
        n = len(snap.ids)
        rng = np.random.default_rng(self.config.seed)
        probe = np.sort(rng.choice(n, size=min(self.config.probe_size, n), replace=False))
        texts = [snap.texts[i] for i in probe]
        probe_a, probe_b = self.serving.embed(texts), self.suspect.embed(texts)
        log.info("Calibrating on %d probe documents (%s vs %s)", len(probe), self.serving.model_id, self.suspect.model_id)

        local = np.arange(len(probe))
        adapter = fit_adapter(probe_a, probe_b, local, self.config.adapter, self.config.ridge_alpha)
        detector = FingerprintDetector.fit(probe_a, probe_b, local, k=self.config.fingerprint_k)

        base = snap.vectors.copy()
        base[probe] = probe_a
        emb_b = np.zeros_like(base)
        emb_b[probe] = probe_b
        top = self.query_log.top(self.config.canary_queries)
        qvecs = self.serving.embed([e.query for e in top])
        weights = np.array([e.count for e in top], dtype=np.float64)
        effects = estimate_action_effects(base, emb_b, probe, qvecs, weights / max(weights.sum(), 1e-12),
                                          self.config.adapter, self.config.ridge_alpha, seed=self.config.seed)
        recovery, harm = (effects.recovery_patch, effects.harm_patch) if effects else (0.5, 0.5)
        traffic = traffic_with_floor(self.query_log.document_traffic(snap.ids), self.config.traffic_floor_fraction)
        self._ctx = Phase2Context(probe, adapter, detector, effects, recovery, harm, traffic)
        self._calibrated_ids = list(snap.ids)
        log.info("Patch effect estimate: recovery %.2f, harm %.2f", recovery, harm)
        return self._ctx

    def _context_for(self, snap: Snapshot) -> Phase2Context:
        if self._ctx is None or self._calibrated_ids != snap.ids:
            return self.calibrate(snap)
        return self._ctx

    def guard(self) -> IngestGuard:
        """A write-path guard sharing this instance's calibrated detector."""
        ctx = self._ctx if self._ctx is not None else self.calibrate()
        return IngestGuard(ctx.detector, self.config.threshold)

    # ---- scan / run / rollback --------------------------------------------
    def scan(self, snap: Snapshot | None = None) -> ScanReport:
        snap = snap or self.store.fetch_all()
        ctx = self._context_for(snap)
        p, flagged = ctx.detector.flag(snap.vectors, self.config.threshold)
        health = float(np.mean(self._canary(snap)(snap.vectors)))
        return ScanReport(snap, p, flagged, health)

    def run(self, apply: bool = False) -> RunReport:
        """Plan a repair under `config.budget`. Dry-run (default) records the
        plan and changes nothing; `apply=True` re-embeds/patches through the
        verified staged rollout, writes what passed, and saves the originals."""
        scan = self.scan()
        snap, ctx, cfg = scan.snapshot, self._context_for(scan.snapshot), self._pipeline_config()
        mode = "apply" if apply else "dry-run"
        run_id = self.state.start_run(mode, self.collection, n_documents=len(snap.ids), n_flagged=scan.n_flagged,
                                      budget=self.config.budget, health_before=scan.health)
        try:
            candidates = np.flatnonzero(scan.flagged)
            p = scan.p_broken[candidates]
            plan = plan_remediation(cfg, ctx, candidates, p, self.config.budget)
            affected = (p * ctx.traffic[candidates]).sum()
            protected = float(plan.plan.total_benefit / affected) if affected > 0 else 1.0
            rank = {int(local): r for r, local in enumerate(plan.rollout_order, start=1)}

            if not apply or len(candidates) == 0:
                actions = [
                    {"doc_id": snap.ids[g], "action": a, "p_broken": float(p[i]), "traffic": float(ctx.traffic[g]),
                     "rollout_rank": rank.get(i), "applied": False}
                    for i, (g, a) in enumerate(zip(candidates, plan.plan.actions))
                ]
                report = RunReport(run_id, mode, len(snap.ids), scan.n_flagged,
                                   int(np.sum(plan.plan.actions == "full_reembed")), int(np.sum(plan.plan.actions == "patch")),
                                   0, float(plan.plan.total_cost), 0.0, scan.health, scan.health, protected, actions)
            else:
                report = self._apply(run_id, snap, scan, ctx, cfg, candidates, p, protected, rank)

            self.state.record_actions(run_id, report.actions)
            self.state.finish_run(run_id, n_full=report.n_full, n_patch=report.n_patch, n_rolled_back=report.n_rolled_back,
                                  cost_planned=report.cost_planned, embed_cost=report.embed_cost,
                                  health_after=report.health_after, protected_share=report.protected_share)
            return report
        except Exception as exc:
            self.state.finish_run(run_id, status="failed", notes=repr(exc))
            raise

    def _apply(self, run_id, snap, scan, ctx, cfg, candidates, p, protected, rank) -> RunReport:
        canary = self._canary(snap)
        reembed = _LazyReembedder(snap.texts, self.serving, snap.vectors.shape[1])
        outcome = remediate(cfg, ctx, snap.vectors, reembed, candidates, p, self.config.budget, canary,
                            seed=self.config.seed)
        changed = candidates[outcome.applied]
        if len(changed):
            # Originals first, then the write: a crash in between leaves a
            # recorded snapshot to restore from, never an unrecorded change.
            self.state.save_snapshot(run_id, [snap.ids[i] for i in changed], snap.vectors[changed])
            for action, tag in (("full_reembed", self.serving.model_id), ("patch", f"{self.serving.model_id}+adapter")):
                idx = candidates[outcome.applied & (outcome.action_applied == action)]
                if len(idx):
                    self.store.upsert([snap.ids[i] for i in idx], outcome.live_embeddings[idx], model_id=tag)
        health_after = float(np.mean(canary(outcome.live_embeddings)))
        log.info("Applied %d re-embeds and %d patches; canary %.3f -> %.3f",
                 outcome.n_full_reembed, outcome.n_patch, scan.health, health_after)
        # The first round planned every candidate, so it records what was
        # intended; `applied` says whether that (or a re-planned exact
        # re-embed) actually went live. A planned fix with applied=False was
        # rolled back by the safety gate.
        planned = outcome.rounds[0][0].plan.actions if outcome.rounds else np.full(len(candidates), "defer", dtype=object)
        actions = [
            {"doc_id": snap.ids[g], "action": outcome.action_applied[i] if outcome.applied[i] else planned[i],
             "p_broken": float(p[i]), "traffic": float(ctx.traffic[g]), "rollout_rank": rank.get(i),
             "applied": bool(outcome.applied[i])}
            for i, g in enumerate(candidates)
        ]
        return RunReport(run_id, "apply", len(snap.ids), scan.n_flagged, outcome.n_full_reembed, outcome.n_patch,
                         outcome.n_rolled_back, outcome.cost_spent, reembed.cost, scan.health, health_after,
                         protected, actions)

    def rollback(self, run_id: int) -> int:
        """Restore every document an `apply` run changed to its pre-run vector.
        Returns the number of documents restored."""
        run = self.state.run(run_id)
        if run is None or run["mode"] != "apply":
            raise ValueError(f"run {run_id} is not an apply run")
        if run["status"] == "rolled-back":
            raise ValueError(f"run {run_id} was already rolled back")
        ids, vectors = self.state.load_snapshot(run_id)
        rb = self.state.start_run("rollback", self.collection, notes=f"rollback of run {run_id}")
        if ids:
            self.store.upsert(ids, vectors, model_id="restored")
        self.state.update_run(run_id, status="rolled-back")
        self.state.finish_run(rb, n_documents=len(ids))
        return len(ids)
