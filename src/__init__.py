"""Muscle synergy extraction from sit-to-stand EMG.

Clark et al. (2010) aligned NMF pipeline on the Gait120 dataset.
"""

__version__ = "0.1.0"

MUSCLE_NAMES = [
    "VL", "RF", "VM", "TA", "BF", "ST",
    "GM", "GL", "SM", "SL", "PL", "PB",
]
N_MUSCLES = 12
EMG_FS = 2000         # Gait120 sampling rate (Hz)
NOTCH_FREQ = 60       # KAIST power grid (Hz)
CYCLE_POINTS = 100    # time-normalized length
