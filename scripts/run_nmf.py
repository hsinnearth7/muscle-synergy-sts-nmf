"""Day 6-7 pipeline driver: fit NMF k=1..8, apply Clark 2010 dual VAF,
save results + k selection, write vaf_summary.txt.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import MUSCLE_NAMES
from src.nmf_fit import fit_nmf_sweep, select_k_clark_dual


def main() -> None:
    processed = ROOT / "data" / "processed"
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    V = np.load(processed / "V_matrix.npy")
    metadata = pd.read_csv(processed / "cycle_metadata.csv")
    n_subjects = metadata["subject_id"].nunique()
    print(f"V_matrix loaded: shape={V.shape}  range=[{V.min():.4f}, {V.max():.4f}]")

    results = fit_nmf_sweep(V, k_range=range(1, 9))

    print("\n{:>4s}  {:>10s}  {:>12s}  {:>12s}  {:>10s}  init".format(
        "k", "global VAF", "min muscle", "frob err", "rank(W)"))
    for k in sorted(results.keys()):
        r = results[k]
        print(
            f"{k:>4d}  {r.global_vaf:>10.4f}  {r.min_muscle_vaf:>12.4f}  "
            f"{r.err:>12.4f}  {np.linalg.matrix_rank(r.W):>10d}  {r.init}"
        )

    chosen_k = select_k_clark_dual(results)
    chosen = results[chosen_k]
    print(f"\nClark 2010 dual criterion -> chosen k = {chosen_k}")
    print(f"  global VAF         = {chosen.global_vaf:.4f}")
    print(f"  min per-muscle VAF = {chosen.min_muscle_vaf:.4f}")

    # Save results + final W, H
    np.save(results_dir / "W_final.npy", chosen.W)
    np.save(results_dir / "H_final.npy", chosen.H)

    with open(results_dir / "nmf_k1_to_8.pkl", "wb") as f:
        pickle.dump(results, f)

    # Write human-readable summary
    summary = []
    summary.append(f"Subjects: {n_subjects}")
    summary.append(f"Cycles  : {V.shape[1] // 100}")
    summary.append(f"V matrix: {V.shape}")
    summary.append("")
    summary.append("Clark 2010 dual VAF criterion (page 3)")
    summary.append("  primary: smallest k with global VAF >= 0.90")
    summary.append("           AND min per-muscle VAF >= 0.90")
    summary.append("")
    summary.append(f"Chosen k          : {chosen_k}")
    summary.append(f"Global VAF        : {chosen.global_vaf:.4f}")
    summary.append(f"Min per-muscle VAF: {chosen.min_muscle_vaf:.4f}")
    summary.append(f"Frobenius err     : {chosen.err:.4f}")
    summary.append(f"Init              : {chosen.init} (seed={chosen.random_state})")
    summary.append("")
    summary.append("Per-muscle VAF at chosen k:")
    for name, v in zip(MUSCLE_NAMES, chosen.per_muscle_vaf):
        summary.append(f"  {name:4s}: {v:.4f}")
    summary.append("")
    summary.append("Sweep table (k | global VAF | min per-muscle VAF | Frob err):")
    for k in sorted(results.keys()):
        r = results[k]
        summary.append(
            f"  k={k}  {r.global_vaf:.4f}  {r.min_muscle_vaf:.4f}  {r.err:.4f}"
        )

    out_path = results_dir / "vaf_summary.txt"
    out_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
