"""Variance-Accounted-For (VAF) metrics for NMF muscle synergy evaluation.

Uses the uncentered / Torres-Oviedo & Ting (2007) convention adopted by
Clark et al. (2010). All formulas are computed on the raw non-negative
matrices without mean subtraction.
"""

from __future__ import annotations

import numpy as np


def global_vaf(V: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    """Global VAF: 1 - ||V - W H||_F^2 / ||V||_F^2."""
    recon = W @ H
    num = float(np.sum((V - recon) ** 2))
    den = float(np.sum(V ** 2))
    if den == 0.0:
        return 0.0
    return 1.0 - num / den


def per_muscle_vaf(V: np.ndarray, W: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Per-muscle VAF -- one value per row of V.

    Returns
    -------
    vafs : ndarray, shape (n_muscles,)
    """
    recon = W @ H
    sse = np.sum((V - recon) ** 2, axis=1)
    sst = np.sum(V ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vafs = np.where(sst > 0, 1.0 - sse / sst, 0.0)
    return vafs
