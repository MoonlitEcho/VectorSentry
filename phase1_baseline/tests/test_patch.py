"""Fast, network-free unit tests for common.patch, using a synthetic
generation-A/B pair related by a KNOWN orthogonal rotation (plus a little
noise) -- lets us check the fitted transform actually recovers the true
relationship, not just that the code runs.
"""

import numpy as np

from common.patch import apply_patch, fit_patch_transform, sample_calibration_indices


def _random_orthogonal(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return q.astype(np.float32)


def _synthetic_pair(n: int, dim: int, seed: int, noise: float = 0.02):
    rng = np.random.default_rng(seed)
    emb_a = rng.normal(size=(n, dim)).astype(np.float32)
    emb_a /= np.linalg.norm(emb_a, axis=1, keepdims=True)

    true_rotation = _random_orthogonal(dim, seed=seed + 1)
    emb_b = emb_a @ true_rotation + rng.normal(scale=noise, size=(n, dim)).astype(np.float32)
    emb_b /= np.linalg.norm(emb_b, axis=1, keepdims=True)
    return emb_a, emb_b


def test_sample_calibration_indices_is_deterministic_and_bounded():
    idx1 = sample_calibration_indices(100, 20, seed=1)
    idx2 = sample_calibration_indices(100, 20, seed=1)
    np.testing.assert_array_equal(idx1, idx2)
    assert len(idx1) == 20
    assert idx1.min() >= 0 and idx1.max() < 100
    assert len(set(idx1)) == 20  # no duplicates


def test_sample_calibration_indices_caps_at_n_documents():
    idx = sample_calibration_indices(10, 50, seed=0)
    assert len(idx) == 10


def test_patch_transform_recovers_generation_a_better_than_raw_generation_b():
    emb_a, emb_b = _synthetic_pair(n=300, dim=16, seed=42)
    calibration = sample_calibration_indices(300, 100, seed=1)
    held_out = np.setdiff1d(np.arange(300), calibration)  # never seen during fitting

    transform = fit_patch_transform(emb_a, emb_b, calibration)
    patched = apply_patch(emb_b, transform)

    raw_sim = np.sum(emb_a[held_out] * emb_b[held_out], axis=1).mean()
    patched_sim = np.sum(emb_a[held_out] * patched[held_out], axis=1).mean()

    assert patched_sim > raw_sim
    assert patched_sim > 0.9  # near-exact recovery: low noise, a true orthogonal relationship


def test_apply_patch_output_is_unit_normalized():
    emb_a, emb_b = _synthetic_pair(n=50, dim=8, seed=3)
    transform = fit_patch_transform(emb_a, emb_b, sample_calibration_indices(50, 20, seed=3))
    patched = apply_patch(emb_b, transform)
    norms = np.linalg.norm(patched, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
