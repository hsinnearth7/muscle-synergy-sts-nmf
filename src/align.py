"""Cross-subject synergy alignment via the Hungarian algorithm.

Given two sets of synergy weight matrices W_ref (12, k) and W_target (12, k),
match target columns to reference columns by maximizing cosine similarity.
Used for Day 11 (optional) cross-subject W consistency analysis.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _cosine(u: np.ndarray, v: np.ndarray, eps: float = 1e-9) -> float:
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + eps))


def align_synergies(
    W_ref: np.ndarray,
    W_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute columns of W_target to best match W_ref by cosine similarity.

    Returns
    -------
    permuted_W : ndarray, shape (n_muscles, k)
        W_target with columns reordered.
    col_ind : ndarray, shape (k,)
        The permutation indices so that ``permuted_W == W_target[:, col_ind]``.
        Apply the same permutation to H rows: ``H_aligned = H_target[col_ind, :]``.
    """
    W_ref = np.asarray(W_ref, dtype=np.float64)
    W_target = np.asarray(W_target, dtype=np.float64)
    if W_ref.ndim != 2 or W_target.ndim != 2:
        raise ValueError(
            f"W_ref and W_target must both be 2D; got {W_ref.ndim}D and {W_target.ndim}D."
        )
    n_muscles_ref, k_ref = W_ref.shape
    n_muscles_tgt, k_tgt = W_target.shape
    if n_muscles_ref != n_muscles_tgt:
        raise ValueError(
            f"W_ref and W_target must have the same number of rows; "
            f"got {n_muscles_ref} and {n_muscles_tgt}."
        )
    if k_ref != k_tgt:
        raise ValueError(
            f"W_ref and W_target must have the same number of columns; "
            f"got {k_ref} and {k_tgt}."
        )

    cost = np.zeros((k_ref, k_tgt), dtype=np.float64)
    for i in range(k_ref):
        for j in range(k_tgt):
            cost[i, j] = 1.0 - _cosine(W_ref[:, i], W_target[:, j])

    _row_ind, col_ind = linear_sum_assignment(cost)
    permuted = W_target[:, col_ind]
    return permuted, col_ind


def cross_subject_similarity(aligned_Ws: list[np.ndarray]) -> np.ndarray:
    """Mean pairwise cosine similarity across aligned subject W matrices.

    Parameters
    ----------
    aligned_Ws : list of ndarray, each (n_muscles, k), already aligned.

    Returns
    -------
    sim : ndarray, shape (n_subjects, n_subjects)
        sim[i, j] = mean over k of cosine(W_i[:, k], W_j[:, k]).
    """
    if not aligned_Ws:
        raise ValueError("aligned_Ws must contain at least one matrix.")
    matrices = [np.asarray(W, dtype=np.float64) for W in aligned_Ws]
    if any(W.ndim != 2 for W in matrices):
        raise ValueError("Every aligned W matrix must be 2D.")
    ref_shape = matrices[0].shape
    for idx, W in enumerate(matrices[1:], start=1):
        if W.shape != ref_shape:
            raise ValueError(
                f"All aligned W matrices must share the same shape; "
                f"expected {ref_shape}, got {W.shape} at index {idx}."
            )

    n = len(matrices)
    k = ref_shape[1]
    sim = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            per_col = [
                _cosine(matrices[i][:, c], matrices[j][:, c])
                for c in range(k)
            ]
            sim[i, j] = sim[j, i] = float(np.mean(per_col))
    return sim
