"""Day 8 pipeline driver: render the 4-panel Figure 1 composite."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nmf_fit import select_k_clark_dual
from src.visualize import make_figure1


def main() -> None:
    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    processed = ROOT / "data" / "processed"

    with open(results_dir / "nmf_k1_to_8.pkl", "rb") as f:
        results = pickle.load(f)

    chosen_k = select_k_clark_dual(results)
    print(f"Chosen k = {chosen_k}")

    V_cycles = np.load(processed / "V_cycles.npy")
    print(f"V_cycles shape = {V_cycles.shape}")

    out = make_figure1(
        V_stacked_cycles=V_cycles,
        results=results,
        chosen_k=chosen_k,
        out_dir=figures_dir,
        example_cycle_idx=0,
        repr_muscles=(0, 3, 4, 6),  # VL, TA, BF, GM
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
