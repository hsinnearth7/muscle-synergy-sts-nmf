"""Tests for cycle extraction and time normalization."""

import numpy as np
import pytest

from src.segment import (
    extract_cycles,
    normalize_cycles,
    resample_cycle_cubic,
    resample_cycle_poly,
)


class TestExtractCycles:
    def test_basic_extraction(self):
        signal = np.arange(100).reshape(50, 2).astype(float)
        cycles = extract_cycles(signal, [(0, 10), (20, 30)])
        assert len(cycles) == 2
        assert cycles[0].shape == (10, 2)
        assert cycles[1].shape == (10, 2)

    def test_invalid_boundary_warns(self):
        signal = np.zeros((50, 2))
        with pytest.warns(RuntimeWarning, match="Skipping invalid"):
            cycles = extract_cycles(signal, [(-1, 10)])
        assert len(cycles) == 0

    def test_end_beyond_signal_warns(self):
        signal = np.zeros((50, 2))
        with pytest.warns(RuntimeWarning, match="Skipping invalid"):
            cycles = extract_cycles(signal, [(0, 100)])
        assert len(cycles) == 0

    def test_empty_cycle_warns(self):
        signal = np.zeros((50, 2))
        with pytest.warns(RuntimeWarning, match="Skipping invalid"):
            cycles = extract_cycles(signal, [(10, 10)])
        assert len(cycles) == 0

    def test_copies_data(self):
        signal = np.ones((50, 2))
        cycles = extract_cycles(signal, [(0, 10)])
        cycles[0][:] = 99
        assert signal[0, 0] == 1.0  # original unchanged


class TestResampleCyclePoly:
    def test_identity_length(self):
        cycle = np.random.rand(100, 4)
        result = resample_cycle_poly(cycle, target_len=100)
        np.testing.assert_array_equal(result, cycle)

    def test_output_shape(self):
        cycle = np.random.rand(200, 12)
        result = resample_cycle_poly(cycle, target_len=100)
        assert result.shape == (100, 12)

    def test_upsample_shape(self):
        cycle = np.random.rand(50, 3)
        result = resample_cycle_poly(cycle, target_len=100)
        assert result.shape == (100, 3)


class TestResampleCycleCubic:
    def test_output_shape(self):
        cycle = np.random.rand(200, 12)
        result = resample_cycle_cubic(cycle, target_len=100)
        assert result.shape == (100, 12)

    def test_endpoints_preserved(self):
        """Cubic spline should exactly hit the first and last points."""
        cycle = np.random.RandomState(0).rand(50, 4)
        result = resample_cycle_cubic(cycle, target_len=100)
        np.testing.assert_allclose(result[0], cycle[0], atol=1e-10)
        np.testing.assert_allclose(result[-1], cycle[-1], atol=1e-10)

    def test_short_cycle_raises(self):
        with pytest.raises(ValueError, match="at least 4 samples"):
            resample_cycle_cubic(np.ones((3, 2)), target_len=100)


class TestNormalizeCycles:
    def test_poly_method(self):
        cycles = [np.random.rand(80, 3), np.random.rand(120, 3)]
        result = normalize_cycles(cycles, method="poly", target_len=50)
        assert result.shape == (2, 50, 3)

    def test_cubic_method(self):
        cycles = [np.random.rand(80, 3)]
        result = normalize_cycles(cycles, method="cubic", target_len=50)
        assert result.shape == (1, 50, 3)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            normalize_cycles([np.zeros((10, 2))], method="linear")

    def test_empty_cycles_raises(self):
        with pytest.raises(ValueError, match="at least one cycle"):
            normalize_cycles([])
