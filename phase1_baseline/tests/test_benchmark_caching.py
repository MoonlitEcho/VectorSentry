"""Fast, network-free regression test for the benchmark cache-path keying.

Model-INDEPENDENT metadata (documents/queries/traffic) must be shared
across embedding-model choices, but model-dependent artifacts (embeddings,
per-ratio mask+index) must NOT be -- an earlier version of this cache keyed
everything on corpus/max_docs/seed only, which would have silently reused
cached embeddings from the WRONG model the moment `generation_b` changed
while corpus/seed stayed the same. Caught while comparing bge-small vs.
MiniLM-L12 as candidate "Generation B" models (see DEVLOG.md's sub-phase 1C
entry) -- before it could corrupt that comparison.
"""

from common.benchmark import _embedding_dir, benchmark_dir
from common.config import Config, EmbeddingModelsConfig


def _cfg(generation_b: str) -> Config:
    return Config(
        seed=1,
        embedding_models=EmbeddingModelsConfig(
            generation_a="sentence-transformers/all-MiniLM-L6-v2",
            generation_b=generation_b,
        ),
    )


def test_metadata_dir_is_shared_across_embedding_model_choices():
    cfg1 = _cfg("BAAI/bge-small-en-v1.5")
    cfg2 = _cfg("sentence-transformers/all-MiniLM-L12-v2")
    assert benchmark_dir(cfg1) == benchmark_dir(cfg2)


def test_embedding_dir_differs_across_embedding_model_choices():
    cfg1 = _cfg("BAAI/bge-small-en-v1.5")
    cfg2 = _cfg("sentence-transformers/all-MiniLM-L12-v2")
    assert _embedding_dir(cfg1) != _embedding_dir(cfg2)
    assert _embedding_dir(cfg1) != benchmark_dir(cfg1)  # nested under it, never equal to it


def test_embedding_dir_is_deterministic():
    cfg = _cfg("BAAI/bge-small-en-v1.5")
    assert _embedding_dir(cfg) == _embedding_dir(cfg)
