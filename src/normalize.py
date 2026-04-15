"""Per-subject amplitude normalization for EMG cycles.

**Clark 2010 p.845 (right column) reference spec**:
    "the EMG from each muscle was normalized to its peak value from
    self-selected walking and resampled at each 1% of the gait cycle.
    For each subject, leg, and walking speed, the EMGs were combined
    into an m × t matrix (EMGo), where m indicates the number of
    muscles and t is the time base (t = no. of strides × 101)."

Clark 2010's normalization is therefore **peak-from-self-selected-walking**
(functionally a "per-subject max across SS walking cycles"), not MVC.
MVC is not used anywhere in Clark 2010's pipeline.

**Methods in this module**:
    - ``'max'``  corresponds to Clark 2010's approach (normalize by the
                 per-subject max across all input cycles). For an STS
                 pipeline, the input cycles are sit-to-stand cycles
                 rather than self-selected walking cycles, but the
                 normalization principle (peak-from-task) is the same.
    - ``'mvc'``  is a Gait120-specific **deviation** from Clark 2010.
                 Gait120 provides per-muscle MVC recordings as a
                 separate protocol, and for STS-only pipelines MVC is
                 a principled alternative because the subject never
                 performs self-selected walking in the dataset. This
                 is a deliberate adaptation, not a bug.

An earlier revision of this docstring described ``'max'`` as a "Clark
2010 fallback when no MVC". That framing was wrong: Clark 2010's
primary method IS the peak-from-task normalization (what ``'max'``
implements), and MVC is the deviation, not the fallback. Corrected on
2026-04-16 after verification against the Clark 2010 PDF Methods text.
"""

from __future__ import annotations

import warnings

import numpy as np

# Epsilon sized for microvolt-scale EMG envelope amplitudes (Gait120 range
# ~1e-4 V). 1e-9 would be swamped by normal signal floor and provide no
# protection against a degenerate zero-MVC channel.
_EPS = 1e-7
# Threshold under which an MVC amplitude is considered implausibly small
# (a failed MVC recording for that muscle).
_MVC_WARN_THRESHOLD = 1e-8


def normalize_by_subject(
    cycles: np.ndarray,
    method: str = "max",
    mvc: np.ndarray | None = None,
    eps: float = _EPS,
) -> np.ndarray:
    """Per-subject amplitude normalization.

    Parameters
    ----------
    cycles : ndarray, shape (n_cycles, T, n_muscles)
        All STS cycles for one subject, time-normalized.
    method : 'mvc' | 'max'
        'max'  : divide each muscle by its per-subject maximum across
                 all STS cycles. This corresponds to Clark 2010's
                 peak-from-task normalization principle (Clark 2010
                 uses peak from self-selected walking; here the task
                 is sit-to-stand).
        'mvc'  : divide each muscle by that subject's MVC amplitude
                 (requires `mvc` array, length = n_muscles). This is a
                 Gait120-specific **deviation** from Clark 2010: the
                 Gait120 dataset provides per-muscle MVC recordings,
                 and using them is a principled alternative for
                 STS-only pipelines where no self-selected walking
                 data exist in the dataset.
    mvc : ndarray or None, shape (n_muscles,)
        Subject-specific MVC amplitude per muscle. Required if method='mvc'.
    eps : float
        Additive denominator floor. Default sized for microvolt EMG.

    Returns
    -------
    normalized : ndarray, same shape as cycles, in [0, 1] (approximately).
    """
    if cycles.ndim != 3:
        raise ValueError(
            f"cycles must be (n_cycles, T, n_muscles); got {cycles.shape}"
        )

    if method == "mvc":
        if mvc is None:
            raise ValueError("method='mvc' requires an `mvc` array.")
        mvc_arr = np.asarray(mvc, dtype=np.float64).ravel()
        if mvc_arr.size != cycles.shape[-1]:
            raise ValueError(
                f"mvc length {mvc_arr.size} != n_muscles {cycles.shape[-1]}"
            )
        low_channels = np.where(mvc_arr < _MVC_WARN_THRESHOLD)[0]
        if low_channels.size > 0:
            warnings.warn(
                f"MVC below {_MVC_WARN_THRESHOLD:g} for muscle indices "
                f"{low_channels.tolist()}; normalization may be unstable",
                RuntimeWarning,
                stacklevel=2,
            )
        norm_factor = mvc_arr.reshape(1, 1, -1)
    elif method == "max":
        norm_factor = cycles.max(axis=(0, 1), keepdims=True)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return cycles / (norm_factor + eps)


def stack_subjects_to_V(all_cycles: list[np.ndarray]) -> np.ndarray:
    """Concatenate per-subject cycle arrays into NMF input matrix V.

    Parameters
    ----------
    all_cycles : list of ndarray, each shape (n_i, T, n_muscles)

    Returns
    -------
    V : ndarray, shape (n_muscles, total_T)
        NMF convention: rows = muscles, columns = time samples.
    """
    stacked = np.concatenate(all_cycles, axis=0)   # (N_total, T, C)
    n_cycles, t_len, n_muscles = stacked.shape
    # (N*T, C) -> (C, N*T)
    return stacked.reshape(-1, n_muscles).T
