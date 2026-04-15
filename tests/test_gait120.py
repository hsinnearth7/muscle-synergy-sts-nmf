"""Tests for Gait120 subject loading against bundled sample data."""

from pathlib import Path

import numpy as np

from src.gait120 import load_subject


ROOT = Path(__file__).resolve().parent.parent
S001_DIR = ROOT / "Gait120_001_to_010" / "S001"


class TestLoadSubject:
    def test_loads_bundled_subject(self):
        subject = load_subject(S001_DIR)

        assert subject.subject_id == "S001"
        assert len(subject.trials) == 5
        assert subject.mvc_per_muscle is not None
        assert subject.mvc_per_muscle.shape == (12,)

        first_trial = subject.trials[0]
        assert first_trial.emg_raw.ndim == 2
        assert first_trial.emg_raw.shape[1] == 12
        assert first_trial.cycle().shape[1] == 12
        assert np.all(subject.mvc_per_muscle >= 0.0)
