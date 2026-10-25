"""Search query logs -- the source of traffic and of the safety canary.

A production search system already records, for each query, how often it is
asked and which documents it returned. VectorSentry needs exactly that, as
JSON Lines, one row per distinct query:

    {"query": "statins and liver damage", "count": 1240, "results": ["d17", "d903", ...]}

`results` must come from BEFORE the incident (a healthy period), because
they double as the reference the canary compares against.

From it come:
- document traffic: position-discounted impressions (`count / log2(rank+2)`
  per appearance), as in the research code but measured, not simulated;
- the replay canary: the most frequent queries re-run against a candidate
  index state, scored by how much of their logged results come back.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from phase2.migrate import _ranked_top_k


@dataclasses.dataclass
class LoggedQuery:
    query: str
    count: float
    results: list[str]


class QueryLog:
    def __init__(self, entries: Sequence[LoggedQuery]):
        self.entries = list(entries)

    @classmethod
    def load(cls, path: str | Path) -> QueryLog:
        entries = []
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    entries.append(LoggedQuery(str(row["query"]), float(row.get("count", 1)), [str(r) for r in row["results"]]))
        return cls(entries)

    def save(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(json.dumps({"query": e.query, "count": e.count, "results": e.results}) + "\n")

    def weights(self) -> np.ndarray:
        """Each query's share of all logged searches."""
        counts = np.array([e.count for e in self.entries], dtype=np.float64)
        return counts / counts.sum() if counts.sum() > 0 else counts

    def document_traffic(self, doc_ids: Sequence[str]) -> np.ndarray:
        """Position-discounted impressions per document, aligned to `doc_ids`."""
        pos = {d: i for i, d in enumerate(doc_ids)}
        traffic = np.zeros(len(doc_ids), dtype=np.float64)
        for e in self.entries:
            for rank, doc_id in enumerate(e.results):
                i = pos.get(doc_id)
                if i is not None:
                    traffic[i] += e.count / np.log2(rank + 2)
        return traffic

    def top(self, n: int) -> list[LoggedQuery]:
        return sorted(self.entries, key=lambda e: -e.count)[:n]


def logged_results_canary(
    query_vectors: np.ndarray, logged_results: Sequence[Sequence[str]], doc_ids: Sequence[str], k: int = 10
) -> Callable[[np.ndarray], np.ndarray]:
    """Per-query score for a candidate index (an (N, d) matrix aligned to
    `doc_ids`): the position-discounted share of each query's logged top-k
    results that the candidate still returns in its top-k.

    Same score as `phase2.migrate.replay_canary_scorer`, but the reference is
    the LOGGED results rather than a recomputed clean index -- in production
    there is no clean copy of the index to recompute from."""
    pos = {d: i for i, d in enumerate(doc_ids)}
    k = min(k, len(doc_ids))
    discount = 1.0 / np.log2(np.arange(k) + 2)
    logged = [[pos[d] for d in results[:k] if d in pos] for results in logged_results]

    def score(embeddings: np.ndarray) -> np.ndarray:
        current = _ranked_top_k(query_vectors, embeddings, k)
        out = np.empty(len(logged))
        for row, ref in enumerate(logged):
            if not ref:
                out[row] = 1.0
                continue
            weights = discount[: len(ref)]
            out[row] = float((np.isin(ref, current[row]) * weights).sum() / weights.sum())
        return out

    return score
