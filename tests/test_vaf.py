"""Tests for VAF (Variance Accounted For) metrics."""

import numpy as np
import pytest

from src.vaf import global_vaf, per_muscle_vaf


class TestGlobalVAF:
    def test_perfect_reconstruction(self):
        """Identity reconstruction should give VAF = 1.0."""
        V = np.random.RandomState(0).rand(4, 50)
        W = np.eye(4)
        H = V.copy()
        assert global_vaf(V, W, H) == pytest.approx(1.0)

    def test_zero_reconstruction(self):
        """Zero W or H should give VAF = 0.0."""
        V = np.ones((4, 50))
        W = np.zeros((4, 2))
        H = np.zeros((2, 50))
        assert global_vaf(V, W, H) == pytest.approx(0.0)

    def test_zero_input(self):
        """Zero V should return 0.0 (not NaN)."""
        V = np.zeros((4, 50))
        W = np.ones((4, 2))
        H = np.ones((2, 50))
        assert global_vaf(V, W, H) == 0.0

    def test_partial_reconstruction(self):
        """VAF should be between 0 and 1 for a reasonable approximation."""
        rng = np.random.RandomState(42)
        W_true = rng.rand(4, 2)
        H_true = rng.rand(2, 100)
        V = W_true @ H_true + 0.1 * rng.rand(4, 100)
        vaf = global_vaf(V, W_true, H_true)
        assert 0.0 < vaf < 1.0

    def test_negative_vaf_possible(self):
        """A very bad reconstruction can produce negative VAF."""
        V = np.ones((2, 10)) * 0.01
        W = np.ones((2, 1)) * 100
        H = np.ones((1, 10)) * 100
        vaf = global_vaf(V, W, H)
        assert vaf < 0.0


class TestPerMuscleVAF:
    def test_perfect_reconstruction(self):
        V = np.random.RandomState(0).rand(4, 50)
        W = np.eye(4)
        H = V.copy()
        vafs = per_muscle_vaf(V, W, H)
        assert vafs.shape == (4,)
        np.testing.assert_allclose(vafs, 1.0)

    def test_zero_row(self):
        """A zero-amplitude muscle should get VAF = 0.0, not NaN."""
        V = np.ones((3, 20))
        V[1, :] = 0.0
        W = np.eye(3)
        H = V.copy()
        vafs = per_muscle_vaf(V, W, H)
        assert vafs[0] == pytest.approx(1.0)
        assert vafs[1] == 0.0  # zero row

    def test_shape(self):
        V = np.random.RandomState(1).rand(12, 100)
        W = np.random.RandomState(1).rand(12, 4)
        H = np.random.RandomState(1).rand(4, 100)
        vafs = per_muscle_vaf(V, W, H)
        assert vafs.shape == (12,)
