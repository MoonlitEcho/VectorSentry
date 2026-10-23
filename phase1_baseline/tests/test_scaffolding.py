"""Smoke test for sub-phase 1A: confirms the repo scaffolding actually works.

This is deliberately not a test of any research logic (there isn't any yet)
-- it just proves that `common`, `phase1`, and `phase2` are all importable
from one editable install, that config loading + validation round-trips
correctly, and that seeding is deterministic. Later sub-phases add real
tests alongside their code.
"""

import random

import common
import phase1
import phase2
from common.config import load_config
from common.seed import set_seed


def test_packages_are_importable():
    # If the editable install / package-dir mapping in pyproject.toml is
    # wrong, these imports fail before we even reach an assertion.
    assert common is not None
    assert phase1 is not None
    assert phase2 is not None


def test_default_config_loads_with_expected_values():
    cfg = load_config("phase1_baseline/configs/default.yaml")
    assert cfg.seed == 42
    assert cfg.corpus.name == "scifact"
    assert cfg.corpus.max_docs == 5000
    assert cfg.embedding_models.generation_a == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.embedding_models.generation_b == "sentence-transformers/all-MiniLM-L12-v2"
    assert cfg.contamination.ratios == [0.05, 0.10, 0.20, 0.35, 0.50]
    assert cfg.budget.levels == [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def test_missing_config_file_raises_clear_error():
    try:
        load_config("phase1_baseline/configs/does_not_exist.yaml")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_set_seed_is_deterministic():
    set_seed(123)
    first = [random.random() for _ in range(5)]
    set_seed(123)
    second = [random.random() for _ in range(5)]
    assert first == second
