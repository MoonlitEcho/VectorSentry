"""Command-line interface.

    python -m vectorsentry demo-setup                  # build a contaminated Qdrant collection + query log
    python -m vectorsentry scan                        # detect and report, change nothing
    python -m vectorsentry run                         # plan a repair (dry-run, the default)
    python -m vectorsentry run --apply --budget 150    # repair through the verified rollout
    python -m vectorsentry runs                        # run history
    python -m vectorsentry show 3                      # one run's plan
    python -m vectorsentry rollback 3                  # undo an apply run
    python -m vectorsentry guard-check                 # demo: the write-path guard on fresh vectors

Targets Qdrant's embedded local mode under --home by default; pass
--qdrant-url to use a Qdrant server instead.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

DEFAULT_HOME = Path(".vectorsentry")
SERVING = "sentence-transformers/all-MiniLM-L6-v2"
SUSPECT = "sentence-transformers/all-MiniLM-L12-v2"


def _client(args):
    """One Qdrant client per command, closed by `main` when the command ends
    (left to the garbage collector, local mode errors during interpreter
    shutdown)."""
    from qdrant_client import QdrantClient

    if getattr(args, "_client", None) is None:
        if args.qdrant_url:
            args._client = QdrantClient(url=args.qdrant_url)
        else:
            (args.home / "qdrant").mkdir(parents=True, exist_ok=True)
            args._client = QdrantClient(path=str(args.home / "qdrant"))
    return args._client


def _sentry(args, client=None):
    from vectorsentry import QdrantStore, QueryLog, SentenceTransformerEmbedder, SentryConfig, StateStore, VectorSentry

    client = client or _client(args)
    store = QdrantStore(client, args.collection)
    config = SentryConfig(budget=args.budget, tolerance=args.tolerance)
    return VectorSentry(
        store,
        SentenceTransformerEmbedder(args.serving),
        SentenceTransformerEmbedder(args.suspect),
        QueryLog.load(args.home / "query_log.jsonl"),
        StateStore(args.home / "state.db"),
        config,
        collection=args.collection,
    )


def cmd_demo_setup(args) -> None:
    """Builds a realistic test bed from the cached SciFact benchmark: a Qdrant
    collection indexed with the serving model, `--contamination` of it
    silently overwritten with suspect-model vectors, and a pre-incident query
    log (Zipf query frequencies, top-10 results on the clean index)."""
    repo = Path(__file__).resolve().parents[3]
    for src in ("common/src", "phase1_baseline/src", "phase2_proposed/src"):
        sys.path.insert(0, str(repo / src))
    from common.benchmark import build_benchmark
    from common.config import load_config
    from common.data import zipf_query_weights
    from vectorsentry import QdrantStore, QueryLog
    from vectorsentry.querylog import LoggedQuery

    cfg = load_config(repo / "phase2_proposed" / "configs" / "default.yaml")
    bench = build_benchmark(cfg)
    ids = [d.doc_id for d in bench.documents]
    texts = [f"{d.title} {d.text}".strip() for d in bench.documents]
    rng = np.random.default_rng(args.seed)
    hit = rng.random(len(ids)) < args.contamination
    vectors = np.where(hit[:, None], bench.emb_b, bench.emb_a)
    models = [SUSPECT if h else SERVING for h in hit]

    client = _client(args)
    store = QdrantStore.create(client, args.collection, dim=vectors.shape[1])
    store.add_documents(ids, vectors, texts, models)

    weights = zipf_query_weights(len(bench.queries), cfg.traffic.zipf_s, cfg.seed)
    top = np.argsort(-(bench.query_emb_a @ bench.emb_a.T), axis=1)[:, :10]
    QueryLog([
        LoggedQuery(q.text, round(float(w) * 100_000, 1), [ids[j] for j in top[i]])
        for i, (q, w) in enumerate(zip(bench.queries, weights))
    ]).save(args.home / "query_log.jsonl")
    print(f"Collection '{args.collection}': {len(ids):,} documents, {int(hit.sum()):,} silently contaminated "
          f"with {SUSPECT}.\nQuery log: {len(bench.queries):,} queries -> {args.home / 'query_log.jsonl'}")


def cmd_scan(args) -> None:
    scan = _sentry(args).scan()
    print(f"{len(scan.snapshot.ids):,} documents scanned; {scan.n_flagged:,} look like {args.suspect}.")
    print(f"Search health (logged results reproduced): {scan.health:.1%}")


def _print_run(report) -> None:
    print(f"Run #{report.run_id} ({report.mode}): {report.n_flagged:,} flagged of {report.n_documents:,}")
    print(f"  plan: re-embed {report.n_full:,}, patch {report.n_patch:,}"
          + (f", rolled back {report.n_rolled_back:,}" if report.mode == "apply" else ""))
    print(f"  budget used: {report.cost_planned:.1f}" + (f"  (embedding spend {report.embed_cost:.1f})" if report.mode == "apply" else ""))
    print(f"  expected affected traffic protected: {report.protected_share:.0%}")
    print(f"  search health: {report.health_before:.1%} -> {report.health_after:.1%}")
    if report.mode == "dry-run":
        print("  dry-run: nothing was changed. Re-run with --apply to repair.")


def cmd_run(args) -> None:
    _print_run(_sentry(args).run(apply=args.apply))


def cmd_runs(args) -> None:
    from vectorsentry import StateStore

    rows = StateStore(args.home / "state.db").runs(args.limit)
    if not rows:
        print("No runs yet.")
    for r in rows:
        health = f"{r['health_before']:.1%} -> {r['health_after']:.1%}" if r["health_after"] is not None else ""
        print(f"#{r['id']:<4} {r['started']}  {r['mode']:<8} {r['status']:<11} flagged={r['n_flagged'] or 0:<5} "
              f"full={r['n_full'] or 0:<5} patch={r['n_patch'] or 0:<5} {health}")


def cmd_show(args) -> None:
    from vectorsentry import StateStore

    state = StateStore(args.home / "state.db")
    run = state.run(args.run_id)
    if run is None:
        sys.exit(f"No run #{args.run_id}")
    print({k: v for k, v in run.items() if v is not None})
    for a in state.actions(args.run_id)[: args.limit]:
        print(f"  {a['rollout_rank'] or '-':>4}  {a['action']:<12} applied={bool(a['applied'])!s:<5} "
              f"p={a['p_broken']:.2f} traffic={a['traffic']:.3f}  {a['doc_id']}")


def cmd_rollback(args) -> None:
    n = _sentry(args).rollback(args.run_id)
    print(f"Run #{args.run_id} rolled back: {n:,} documents restored to their pre-run vectors.")


def cmd_guard_check(args) -> None:
    """Embeds a sample of document texts with both models and shows the guard
    accepting the serving model's vectors and rejecting the suspect's."""
    sentry = _sentry(args)
    guard = sentry.guard()
    snap = sentry.store.fetch_all()
    texts = snap.texts[: args.sample]
    for label, embedder in (("serving", sentry.serving), ("suspect", sentry.suspect)):
        _, ok = guard.check(embedder.embed(texts))
        print(f"{label:>8} model ({embedder.model_id}): {int(ok.sum())}/{len(ok)} vectors accepted")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="vectorsentry", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME, help="state, query log and local Qdrant data")
    parser.add_argument("--qdrant-url", default=None, help="use a Qdrant server instead of local mode")
    parser.add_argument("--collection", default="scifact")
    parser.add_argument("--serving", default=SERVING, help="the model the index should use")
    parser.add_argument("--suspect", default=SUSPECT, help="the model suspected of contaminating it")
    parser.add_argument("--budget", type=float, default=150.0, help="per run, in full re-embeds")
    parser.add_argument("--tolerance", type=float, default=0.01, help="safety margin on search health")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo-setup", help="build a contaminated demo collection and query log")
    p.add_argument("--contamination", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_demo_setup)
    sub.add_parser("scan", help="detect and report").set_defaults(func=cmd_scan)
    p = sub.add_parser("run", help="plan (dry-run) or apply a repair")
    p.add_argument("--apply", action="store_true", help="actually change the index")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("runs", help="run history")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_runs)
    p = sub.add_parser("show", help="one run's plan")
    p.add_argument("run_id", type=int)
    p.add_argument("--limit", type=int, default=25)
    p.set_defaults(func=cmd_show)
    p = sub.add_parser("rollback", help="undo an apply run")
    p.add_argument("run_id", type=int)
    p.set_defaults(func=cmd_rollback)
    p = sub.add_parser("guard-check", help="demo the write-path guard")
    p.add_argument("--sample", type=int, default=200)
    p.set_defaults(func=cmd_guard_check)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        args.func(args)
    finally:
        if getattr(args, "_client", None) is not None:
            args._client.close()


if __name__ == "__main__":
    main()
