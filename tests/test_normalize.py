"""Tests for per-subject amplitude normalization."""

import numpy as np
import pytest

from src.normalize import normalize_by_subject, stack_subjects_to_V


class TestNormalizeBySubject:
    def test_max_normalization_range(self):
        """Max normalization should produce values in (0, ~1]."""
        rng = np.random.RandomState(0)
        cycles = rng.rand(5, 100, 12) + 0.01
        result = normalize_by_subject(cycles, method="max")
        assert result.max() <= 1.0 + 1e-6
        assert result.min() >= 0.0

    def test_mvc_normalization(self):
        rng = np.random.RandomState(0)
        cycles = rng.rand(5, 100, 4)
        mvc = np.array([1.0, 2.0, 0.5, 1.5])
        result = normalize_by_subject(cycles, method="mvc", mvc=mvc)
        assert result.shape == cycles.shape

    def test_mvc_without_array_raises(self):
        cycles = np.zeros((2, 50, 3))
        with pytest.raises(ValueError, match="requires"):
            normalize_by_subject(cycles, method="mvc")

    def test_mvc_wrong_length_raises(self):
        cycles = np.zeros((2, 50, 3))
        with pytest.raises(ValueError, match="mvc length"):
            normalize_by_subject(cycles, method="mvc", mvc=np.ones(5))

    def test_invalid_method_raises(self):
        cycles = np.zeros((2, 50, 3))
        with pytest.raises(ValueError, match="Unknown"):
            normalize_by_subject(cycles, method="zscore")

    def test_wrong_ndim_raises(self):
        cycles = np.zeros((100, 12))
        with pytest.raises(ValueError, match="n_cycles, T, n_muscles"):
            normalize_by_subject(cycles)

    def test_low_mvc_warns(self):
        cycles = np.ones((1, 10, 3))
        mvc = np.array([1.0, 1e-10, 1.0])
        with pytest.warns(RuntimeWarning, match="MVC below"):
            normalize_by_subject(cycles, method="mvc", mvc=mvc)


class TestStackSubjectsToV:
    def test_shape(self):
        all_cycles = [
            np.random.rand(3, 100, 12),
            np.random.rand(5, 100, 12),
        ]
        V = stack_subjects_to_V(all_cycles)
        assert V.shape == (12, 8 * 100)

    def test_nmf_convention(self):
        """V should be (n_muscles, total_time) -- muscles as rows."""
        cycles = [np.random.rand(2, 50, 4)]
        V = stack_subjects_to_V(cycles)
        assert V.shape[0] == 4
        assert V.shape[1] == 2 * 50
