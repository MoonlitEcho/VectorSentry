"""Single seeding entry point for reproducibility.

Design decision: every script that touches randomness (traffic sampling,
contamination-ratio subset selection, sklearn detectors, etc.) must call
`set_seed()` once at startup with the seed from the loaded Config, rather
than seeding ad hoc in multiple places. This is what lets us claim the
Phase 1 vs. Phase 2 comparison in 2C is apples-to-apples: both phases run
against a benchmark built from the same seed.

torch is seeded defensively (best-effort import) because sentence-transformers
pulls it in transitively, but this module must not hard-depend on torch just
to seed python/numpy randomness.
"""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python's random module and numpy. Also seeds torch if importable."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
