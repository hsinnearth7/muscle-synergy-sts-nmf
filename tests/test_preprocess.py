"""Tests for EMG preprocessing pipeline."""

import numpy as np
import pytest

from src.preprocess import preprocess_emg, rmse
from src.visualize import plot_raw_emg_example


class TestPreprocessEmg:
    def test_output_shape(self):
        """Output should have the same shape as input."""
        raw = np.random.RandomState(0).randn(10000, 12)
        env = preprocess_emg(raw, fs=2000.0)
        assert env.shape == raw.shape

    def test_non_negative(self):
        """Envelope must be non-negative (required for NMF)."""
        raw = np.random.RandomState(0).randn(10000, 12)
        env = preprocess_emg(raw, fs=2000.0)
        assert np.all(env >= 0.0)

    def test_1d_input(self):
        """Single-channel 1D input should be handled gracefully."""
        raw = np.random.RandomState(0).randn(5000)
        env = preprocess_emg(raw, fs=2000.0)
        assert env.shape == (5000, 1)
        assert np.all(env >= 0.0)

    def test_envelope_smoother_than_raw(self):
        """The 4 Hz lowpass should make the envelope much smoother."""
        rng = np.random.RandomState(0)
        raw = rng.randn(10000, 1)
        env = preprocess_emg(raw, fs=2000.0)
        raw_std = np.std(np.diff(np.abs(raw[:, 0])))
        env_std = np.std(np.diff(env[:, 0]))
        assert env_std < raw_std * 0.1

    def test_short_signal_no_padding(self):
        """Very short signals should skip padding and still work."""
        raw = np.random.RandomState(0).randn(100, 4)
        env = preprocess_emg(raw, fs=2000.0, pad_seconds=1.0)
        assert env.shape == (100, 4)
        assert np.all(env >= 0.0)


class TestRmse:
    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert rmse(a, a) == pytest.approx(0.0)

    def test_known_value(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        expected = np.sqrt((9 + 16) / 2)
        assert rmse(a, b) == pytest.approx(expected)


class TestPlotRawEmgExample:
    def test_short_signal_is_clamped(self, monkeypatch):
        raw = np.random.RandomState(0).randn(100, 2)
        saved = {}

        def fake_savefig(self, *args, **kwargs):
            saved["called"] = True

        monkeypatch.setattr("matplotlib.figure.Figure.savefig", fake_savefig)
        out = plot_raw_emg_example(
            raw,
            fs=2000.0,
            seconds=5.0,
            muscle_names=("M1", "M2"),
            out_path="figures/test_raw_emg_example.png",
        )
        assert saved["called"] is True
        assert out.name == "test_raw_emg_example.png"

    def test_1d_input_is_supported(self, monkeypatch):
        raw = np.random.RandomState(0).randn(100)
        saved = {}

        def fake_savefig(self, *args, **kwargs):
            saved["called"] = True

        monkeypatch.setattr("matplotlib.figure.Figure.savefig", fake_savefig)
        out = plot_raw_emg_example(
            raw,
            fs=2000.0,
            out_path="figures/test_raw_emg_example_1d.png",
        )
        assert saved["called"] is True
        assert out.name == "test_raw_emg_example_1d.png"
