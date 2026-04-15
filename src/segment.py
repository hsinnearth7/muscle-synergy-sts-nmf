"""Sit-to-stand cycle extraction and time normalization.

Gait120 stores STS cycles inside per-subject `.mat` files. The exact
marker-based onset / offset convention should be confirmed in Day 2 EDA.
This module provides generic cycle-extraction helpers that work as long
as cycle boundaries (sample indices) are provided.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from math import gcd

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample_poly


def extract_cycles(
    signal: np.ndarray,
    boundaries: Iterable[tuple[int, int]],
) -> list[np.ndarray]:
    """Slice a multi-channel signal into cycles.

    Parameters
    ----------
    signal : ndarray, shape (n_samples, n_channels)
    boundaries : iterable of (start_idx, end_idx) inclusive-exclusive.

    Returns
    -------
    list of ndarray with shape (cycle_len_i, n_channels).
    """
    cycles = []
    for start, end in boundaries:
        if start < 0 or end > signal.shape[0] or end <= start:
            warnings.warn(
                f"Skipping invalid cycle boundary ({start}, {end}) "
                f"for signal shape {signal.shape}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        cycles.append(signal[start:end, :].copy())
    return cycles


def resample_cycle_poly(cycle: np.ndarray, target_len: int = 100) -> np.ndarray:
    """Polyphase-FIR resample to target length.

    Preferred over `scipy.signal.resample` because the FFT-based method
    can introduce ringing on short, non-periodic cycles.
    """
    cycle = np.asarray(cycle, dtype=np.float64)
    if cycle.ndim != 2:
        raise ValueError(f"cycle must be 2D (n_samples, n_channels); got shape {cycle.shape}")
    orig_len = cycle.shape[0]
    if orig_len < 1:
        raise ValueError("cycle must contain at least one sample.")
    if orig_len == target_len:
        return cycle.copy()
    g = gcd(target_len, orig_len)
    up = target_len // g
    down = orig_len // g
    return resample_poly(cycle, up=up, down=down, axis=0)


def resample_cycle_cubic(cycle: np.ndarray, target_len: int = 100) -> np.ndarray:
    """Cubic-spline resample to target length (alternative to polyphase)."""
    cycle = np.asarray(cycle, dtype=np.float64)
    if cycle.ndim != 2:
        raise ValueError(f"cycle must be 2D (n_samples, n_channels); got shape {cycle.shape}")
    if cycle.shape[0] < 4:
        raise ValueError(
            f"cubic resampling requires at least 4 samples; got {cycle.shape[0]}"
        )
    t_old = np.linspace(0.0, 1.0, cycle.shape[0])
    t_new = np.linspace(0.0, 1.0, target_len)
    f = interp1d(t_old, cycle, kind="cubic", axis=0)
    return f(t_new)


def normalize_cycles(
    cycles: list[np.ndarray],
    method: str = "poly",
    target_len: int = 100,
) -> np.ndarray:
    """Stack a list of variable-length cycles into a (n_cycles, T, C) array.

    method : 'poly' | 'cubic'
    """
    if method == "poly":
        fn = resample_cycle_poly
    elif method == "cubic":
        fn = resample_cycle_cubic
    else:
        raise ValueError(f"Unknown resample method: {method}")

    if not cycles:
        raise ValueError("cycles must contain at least one cycle.")
    resampled = [fn(c, target_len=target_len) for c in cycles]
    return np.stack(resampled, axis=0)
