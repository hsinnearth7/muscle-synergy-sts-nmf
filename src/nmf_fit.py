"""NMF driver with restart logic and Clark 2010 dual-criterion k-selection.

Restart policy in this module: 1 deterministic nndsvda + 49 random
initializations, keeping the run with the lowest Frobenius
reconstruction error. Uses multiplicative update (``solver='mu'``) to
match the iterative-optimization style used by the original NNMF
algorithm.

**Important attribution note**: Clark 2010 does NOT specify a restart
count. The Clark 2010 Methods text (p.845, right column) cites
*"Lee and Seung 1999; Ting and Macpherson 2005"* for the NNMF
algorithm and simply says *"Within this framework, the NNMF algorithm
performed an iterative optimization until it converged on the muscle
weights and the activation timings of the modules that minimized the
error."* The 50-total (1 nndsvda + 49 random) choice in this module
is a **defensive local practice**, not a Clark 2010 protocol. A
previous revision of this docstring wrongly attributed "50-100
random restarts" to Clark 2010 / Tresch 2006; that attribution has
been corrected on 2026-04-16 after verifying against the Clark 2010
PDF Methods text.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import NMF

from .vaf import global_vaf, per_muscle_vaf

# Defensive local choice: 49 random restarts + 1 nndsvda deterministic run.
# Not specified by Clark 2010 or any single cited paper; chosen to reduce
# the risk of NMF solution non-uniqueness affecting the reported results.
N_RANDOM_RESTARTS = 49


@dataclass
class NMFResult:
    k: int
    W: np.ndarray
    H: np.ndarray
    err: float
    init: str
    random_state: int
    global_vaf: float
    per_muscle_vaf: np.ndarray

    @property
    def min_muscle_vaf(self) -> float:
        return float(self.per_muscle_vaf.min())


def fit_nmf_once(
    V: np.ndarray,
    k: int,
    init: str = "nndsvda",
    random_state: int = 0,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a single NMF model and return (W, H, frobenius_error)."""
    model = NMF(
        n_components=k,
        init=init,
        solver="mu",
        beta_loss="frobenius",
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    W = model.fit_transform(V)
    H = model.components_
    err = float(np.linalg.norm(V - W @ H, ord="fro"))
    return W, H, err


def fit_nmf_with_restarts(
    V: np.ndarray,
    k: int,
    n_random_restarts: int = N_RANDOM_RESTARTS,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> NMFResult:
    """Fit NMF at fixed k with multiple restarts.

    Runs nndsvda (deterministic) + ``n_random_restarts`` random
    initializations, keeps the run with the lowest Frobenius error.
    Default is 1 + 49 = 50 total runs. See module docstring: this
    restart count is a defensive local choice, not a Clark 2010
    protocol (Clark 2010 does not specify a restart count).
    """
    candidates: list[tuple[str, int]] = [("nndsvda", 0)]
    candidates += [("random", i) for i in range(1, n_random_restarts + 1)]

    best: dict | None = None
    for init, rs in candidates:
        W, H, err = fit_nmf_once(
            V, k=k, init=init, random_state=rs,
            max_iter=max_iter, tol=tol,
        )
        if best is None or err < best["err"]:
            best = dict(W=W, H=H, err=err, init=init, random_state=rs)

    if best is None:
        raise RuntimeError(f"NMF fitting produced no result for k={k}")
    gv = global_vaf(V, best["W"], best["H"])
    pv = per_muscle_vaf(V, best["W"], best["H"])
    return NMFResult(
        k=k,
        W=best["W"],
        H=best["H"],
        err=best["err"],
        init=best["init"],
        random_state=best["random_state"],
        global_vaf=gv,
        per_muscle_vaf=pv,
    )


def fit_nmf_sweep(
    V: np.ndarray,
    k_range: range = range(1, 9),
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> dict[int, NMFResult]:
    """Fit NMF for every k in `k_range` with restart logic."""
    out: dict[int, NMFResult] = {}
    for k in k_range:
        out[k] = fit_nmf_with_restarts(V, k, max_iter=max_iter, tol=tol)
    return out


def select_k_clark_dual(
    results: dict[int, NMFResult],
    global_thr: float = 0.90,
    per_muscle_thr: float = 0.90,
) -> int:
    """Clark et al. (2010) dual VAF criterion for k selection.

    Clark 2010 page 3: "If VAF was 90% for each of the eight muscles and
    six regions, it was concluded that additional modules were not needed."
    So both thresholds default to 0.90, matching the original paper.

    Primary:  smallest k with global_vaf >= global_thr AND
              min(per_muscle_vaf) >= per_muscle_thr.
    Fallback: if the per-muscle threshold is never met (a common
              real-world outcome on higher-channel-count datasets),
              fall back to the smallest k meeting the primary global
              threshold alone. The caller is expected to report the
              per-muscle gap honestly.
    Last resort: if neither threshold is met, return the k with the
              highest global VAF.
    """
    if not results:
        raise ValueError("results dict is empty; NMF sweep produced no output")
    ks = sorted(results.keys())
    for k in ks:
        r = results[k]
        if r.global_vaf >= global_thr and r.min_muscle_vaf >= per_muscle_thr:
            return k
    for k in ks:
        if results[k].global_vaf >= global_thr:
            return k
    return max(ks, key=lambda k: results[k].global_vaf)
