"""Tests for NMF fitting and k-selection."""

import numpy as np
import pytest

from src.nmf_fit import (
    NMFResult,
    fit_nmf_once,
    fit_nmf_with_restarts,
    select_k_clark_dual,
)


def _make_synthetic_V(n_muscles: int = 12, k_true: int = 3, n_time: int = 200) -> np.ndarray:
    """Generate a synthetic non-negative V matrix with known rank."""
    rng = np.random.RandomState(42)
    W = rng.rand(n_muscles, k_true)
    H = rng.rand(k_true, n_time)
    V = W @ H + 0.01 * rng.rand(n_muscles, n_time)
    return np.clip(V, 0, None)


class TestFitNmfOnce:
    def test_shapes(self):
        V = _make_synthetic_V()
        W, H, err = fit_nmf_once(V, k=3, max_iter=200)
        assert W.shape == (12, 3)
        assert H.shape == (3, 200)
        assert err >= 0.0

    def test_non_negative(self):
        V = _make_synthetic_V()
        W, H, _ = fit_nmf_once(V, k=3, max_iter=200)
        assert np.all(W >= 0)
        assert np.all(H >= 0)


class TestFitNmfWithRestarts:
    def test_returns_result(self):
        V = _make_synthetic_V()
        result = fit_nmf_with_restarts(V, k=3, n_random_restarts=2, max_iter=200)
        assert isinstance(result, NMFResult)
        assert result.k == 3
        assert result.global_vaf > 0.5
        assert result.per_muscle_vaf.shape == (12,)

    def test_higher_k_gives_higher_vaf(self):
        V = _make_synthetic_V(k_true=4)
        r2 = fit_nmf_with_restarts(V, k=2, n_random_restarts=2, max_iter=200)
        r4 = fit_nmf_with_restarts(V, k=4, n_random_restarts=2, max_iter=200)
        assert r4.global_vaf >= r2.global_vaf


class TestSelectKClarkDual:
    def _make_results(self, vafs: dict[int, tuple[float, float]]) -> dict[int, NMFResult]:
        """Helper: vafs maps k -> (global_vaf, min_muscle_vaf)."""
        results = {}
        for k, (gv, mv) in vafs.items():
            results[k] = NMFResult(
                k=k,
                W=np.zeros((12, k)),
                H=np.zeros((k, 100)),
                err=0.0,
                init="nndsvda",
                random_state=0,
                global_vaf=gv,
                per_muscle_vaf=np.full(12, mv),
            )
        return results

    def test_both_criteria_met(self):
        results = self._make_results({
            1: (0.50, 0.30),
            2: (0.80, 0.50),
            3: (0.92, 0.80),
            4: (0.96, 0.85),
        })
        assert select_k_clark_dual(results) == 3

    def test_fallback_global_only(self):
        """When per-muscle threshold is never met, fall back to global-only."""
        results = self._make_results({
            1: (0.50, 0.30),
            2: (0.80, 0.50),
            3: (0.92, 0.60),  # global passes, per-muscle doesn't
            4: (0.96, 0.70),
        })
        assert select_k_clark_dual(results) == 3

    def test_last_resort(self):
        """When neither threshold is met, return k with highest global VAF."""
        results = self._make_results({
            1: (0.50, 0.30),
            2: (0.70, 0.40),
            3: (0.85, 0.50),
        })
        assert select_k_clark_dual(results) == 3

    def test_empty_results_raises(self):
        with pytest.raises(ValueError, match="empty"):
            select_k_clark_dual({})

    def test_smallest_k_preferred(self):
        """If multiple k values pass, the smallest should be chosen."""
        results = self._make_results({
            3: (0.92, 0.80),
            4: (0.96, 0.85),
            5: (0.98, 0.90),
        })
        assert select_k_clark_dual(results) == 3
