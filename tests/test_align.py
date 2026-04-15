"""Tests for cross-subject synergy alignment."""

import numpy as np
import pytest

from src.align import align_synergies, cross_subject_similarity


class TestAlignSynergies:
    def test_identity_permutation(self):
        """Aligning identical W matrices should return the same W."""
        rng = np.random.RandomState(0)
        W = rng.rand(12, 4)
        permuted, col_ind = align_synergies(W, W.copy())
        np.testing.assert_allclose(permuted, W, atol=1e-10)

    def test_known_permutation(self):
        """A column-swapped W should be correctly realigned."""
        rng = np.random.RandomState(0)
        W_ref = rng.rand(12, 3)
        perm = [2, 0, 1]
        W_target = W_ref[:, perm]
        permuted, col_ind = align_synergies(W_ref, W_target)
        np.testing.assert_allclose(permuted, W_ref, atol=1e-10)

    def test_returns_permutation_indices(self):
        rng = np.random.RandomState(0)
        W_ref = rng.rand(12, 4)
        W_target = W_ref[:, [3, 2, 1, 0]]
        _, col_ind = align_synergies(W_ref, W_target)
        assert col_ind.shape == (4,)
        np.testing.assert_array_equal(W_target[:, col_ind], W_ref)

    def test_different_k_raises(self):
        with pytest.raises(ValueError, match="same number of columns"):
            align_synergies(np.zeros((12, 3)), np.zeros((12, 4)))

    def test_different_row_count_raises(self):
        with pytest.raises(ValueError, match="same number of rows"):
            align_synergies(np.zeros((12, 3)), np.zeros((10, 3)))


class TestCrossSubjectSimilarity:
    def test_self_similarity(self):
        """Diagonal should be 1.0."""
        Ws = [np.random.RandomState(i).rand(12, 4) for i in range(5)]
        sim = cross_subject_similarity(Ws)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-8)

    def test_symmetry(self):
        Ws = [np.random.RandomState(i).rand(12, 4) for i in range(5)]
        sim = cross_subject_similarity(Ws)
        np.testing.assert_allclose(sim, sim.T)

    def test_identical_matrices(self):
        W = np.random.RandomState(0).rand(12, 4)
        sim = cross_subject_similarity([W, W, W])
        np.testing.assert_allclose(sim, 1.0, atol=1e-8)

    def test_shape(self):
        Ws = [np.random.RandomState(i).rand(8, 3) for i in range(7)]
        sim = cross_subject_similarity(Ws)
        assert sim.shape == (7, 7)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="at least one matrix"):
            cross_subject_similarity([])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            cross_subject_similarity([np.zeros((12, 3)), np.zeros((10, 3))])
