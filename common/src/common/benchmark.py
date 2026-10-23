"""Orchestrates the full contaminated benchmark used by both phases.

Three-layer cache on disk:

  <cache_dir>/benchmark/<corpus>_<max_docs>_<seed>/
      documents.json, queries.json, qrels_eval.json, doc_traffic.npy
          -- model-INDEPENDENT: corpus subsample + traffic weights, built once.
      embeddings/<generation_a_slug>__vs__<generation_b_slug>/
          emb_a.npy, emb_b.npy
              -- built once per (corpus, model-pair).
          ratios/r05/mask.npy, index.faiss  (one such dir per contamination ratio)
              -- built once per (corpus, model-pair, ratio), from cheap numpy
                 indexing into emb_a/emb_b, not re-embedding.

The model-pair layer matters for correctness, not just speed: swapping
`generation_b` (e.g. comparing bge-small vs. MiniLM-L12 as the "Generation
B" model -- see DEVLOG.md's sub-phase 1C entry) while keeping the same
corpus/seed must NOT silently reuse embeddings computed from a different
model. An earlier version of this cache keyed only on corpus/max_docs/seed
and had exactly that bug -- caught before it corrupted a real comparison.

`build_benchmark`/`build_contaminated_ratio` are idempotent and cache-aware:
call them freely from detector code, remediation code, or the dashboard --
the expensive work happens once per (corpus, model-pair[, ratio]).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from common.config import Config
from common.data import (
    Document,
    Query,
    RawBeirData,
    compute_query_traffic,
    load_beir_dataset,
    subsample_corpus,
)
from common.embeddings import embed_documents
from common.index import build_index

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError("faiss-cpu is required (see pyproject.toml).") from exc


@dataclasses.dataclass
class Benchmark:
    """Ratio-independent shared benchmark state."""

    documents: list[Document]
    queries: list[Query]
    qrels_eval: dict[str, set[str]]
    query_traffic: dict[str, float]  # query_id -> Zipfian popularity weight
    doc_traffic: np.ndarray  # (N,) float32, aligned to `documents`
    emb_a: np.ndarray  # (N, d) float32, generation-A embeddings for the full corpus
    emb_b: np.ndarray  # (N, d) float32, generation-B embeddings for the full corpus


@dataclasses.dataclass
class ContaminatedRatio:
    """One contamination-ratio view of a Benchmark: which docs are served by
    generation B, and the resulting mixed-generation FAISS index."""

    ratio: float
    mask: np.ndarray  # (N,) bool; True = document served by generation B (contaminated)
    index: "faiss.Index"


def benchmark_dir(cfg: Config) -> Path:
    """The model-INDEPENDENT cache directory: corpus subsample + traffic
    weights only. Two configs that differ only in embedding models, or in
    fields not part of this path at all (e.g. budget levels), share this."""
    return Path(cfg.paths.cache_dir) / "benchmark" / f"{cfg.corpus.name}_{cfg.corpus.max_docs}_{cfg.seed}"


def _slug(model_name: str) -> str:
    return model_name.replace("/", "__").replace(" ", "_")


def _embedding_dir(cfg: Config) -> Path:
    """The model-pair-specific cache directory: embeddings + every
    contamination ratio's mask/index built from them. See module docstring
    for why this must be keyed on the model names, not just the corpus."""
    pair = f"{_slug(cfg.embedding_models.generation_a)}__vs__{_slug(cfg.embedding_models.generation_b)}"
    return benchmark_dir(cfg) / "embeddings" / pair


def _ratio_dirname(ratio: float) -> str:
    return f"r{round(ratio * 100):02d}"


def build_benchmark(cfg: Config, force: bool = False) -> Benchmark:
    """Builds (or loads from cache) the benchmark state for `cfg`'s corpus
    and embedding-model pair."""
    bench_dir = benchmark_dir(cfg)
    meta_path = bench_dir / "documents.json"
    query_traffic_path = bench_dir / "query_traffic.json"

    if not force and meta_path.exists() and query_traffic_path.exists():
        documents, queries, qrels_eval, query_traffic, doc_traffic = _load_metadata(bench_dir)
    else:
        raw = load_beir_dataset(cfg.corpus.name)
        raw = subsample_corpus(raw, cfg.corpus.max_docs, cfg.seed)
        query_traffic, doc_traffic = compute_query_traffic(raw, cfg.traffic.zipf_s, cfg.seed)
        documents, queries, qrels_eval = raw.documents, raw.queries, raw.qrels_eval
        bench_dir.mkdir(parents=True, exist_ok=True)
        _save_metadata(bench_dir, raw, query_traffic, doc_traffic)

    embed_dir = _embedding_dir(cfg)
    emb_a = embed_documents(documents, cfg.embedding_models.generation_a, embed_dir / "emb_a.npy")
    emb_b = embed_documents(documents, cfg.embedding_models.generation_b, embed_dir / "emb_b.npy")

    if emb_a.shape[1] != emb_b.shape[1]:
        # A dimension mismatch would make a vector DB reject writes outright
        # (a loud error) rather than silently degrade retrieval -- which
        # would defeat the entire premise of this project. Both configured
        # models must share output dimensionality.
        raise ValueError(
            f"generation_a and generation_b must produce same-dimensional embeddings "
            f"for contamination to be silent; got {emb_a.shape[1]} vs {emb_b.shape[1]}."
        )

    return Benchmark(documents, queries, qrels_eval, query_traffic, doc_traffic, emb_a, emb_b)


def build_contaminated_ratio(
    cfg: Config, benchmark: Benchmark, ratio: float, force: bool = False
) -> ContaminatedRatio:
    """Builds (or loads from cache) the mixed-generation index for one
    contamination ratio."""
    ratio_dir = _embedding_dir(cfg) / "ratios" / _ratio_dirname(ratio)
    mask_path = ratio_dir / "mask.npy"
    index_path = ratio_dir / "index.faiss"

    if not force and mask_path.exists() and index_path.exists():
        mask = np.load(mask_path)
        index = faiss.read_index(str(index_path))
        return ContaminatedRatio(ratio, mask, index)

    n = len(benchmark.documents)
    # A distinct RNG per ratio (derived from the global seed) makes each
    # ratio's contamination assignment independent yet fully reproducible.
    # The assignment is uniform random over documents -- deliberately
    # UNCORRELATED with traffic -- because a real embedding-provider
    # migration touches documents in arbitrary/ingestion order, not
    # popularity order. Correlating them would make the benchmark
    # unrealistically easy to fix well by chance.
    rng = np.random.default_rng(cfg.seed * 100_000 + round(ratio * 10_000))
    mask = rng.random(n) < ratio

    mixed = _mixed_embeddings(benchmark, mask)
    index = build_index(mixed)

    ratio_dir.mkdir(parents=True, exist_ok=True)
    np.save(mask_path, mask)
    faiss.write_index(index, str(index_path))

    return ContaminatedRatio(ratio, mask, index)


def _mixed_embeddings(benchmark: Benchmark, mask: np.ndarray) -> np.ndarray:
    return np.where(mask[:, None], benchmark.emb_b, benchmark.emb_a).astype(np.float32)


def mixed_embeddings(benchmark: Benchmark, contaminated: ContaminatedRatio) -> np.ndarray:
    """Reconstructs the raw (N, d) mixed-generation embedding matrix a
    ContaminatedRatio's FAISS index was built from -- e.g. for detectors
    (phase1.detection) that need direct array access rather than kNN search
    through the index."""
    return _mixed_embeddings(benchmark, contaminated.mask)


def _save_metadata(
    bench_dir: Path, raw: RawBeirData, query_traffic: dict[str, float], doc_traffic: np.ndarray
) -> None:
    with (bench_dir / "documents.json").open("w", encoding="utf-8") as f:
        json.dump([dataclasses.asdict(d) for d in raw.documents], f)
    with (bench_dir / "queries.json").open("w", encoding="utf-8") as f:
        json.dump([dataclasses.asdict(q) for q in raw.queries], f)
    with (bench_dir / "qrels_eval.json").open("w", encoding="utf-8") as f:
        json.dump({qid: sorted(docids) for qid, docids in raw.qrels_eval.items()}, f)
    with (bench_dir / "query_traffic.json").open("w", encoding="utf-8") as f:
        json.dump(query_traffic, f)
    np.save(bench_dir / "doc_traffic.npy", doc_traffic)


def _load_metadata(
    bench_dir: Path,
) -> tuple[list[Document], list[Query], dict[str, set[str]], dict[str, float], np.ndarray]:
    with (bench_dir / "documents.json").open("r", encoding="utf-8") as f:
        documents = [Document(**d) for d in json.load(f)]
    with (bench_dir / "queries.json").open("r", encoding="utf-8") as f:
        queries = [Query(**q) for q in json.load(f)]
    with (bench_dir / "qrels_eval.json").open("r", encoding="utf-8") as f:
        qrels_eval = {qid: set(docids) for qid, docids in json.load(f).items()}
    with (bench_dir / "query_traffic.json").open("r", encoding="utf-8") as f:
        query_traffic = json.load(f)
    doc_traffic = np.load(bench_dir / "doc_traffic.npy")
    return documents, queries, qrels_eval, query_traffic, doc_traffic
