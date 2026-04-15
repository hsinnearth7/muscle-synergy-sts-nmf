# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 04. Figure 1 Main Composite (Day 8)
#
# Produce the 4-panel Figure 1 for README and report.pdf.

# %%
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path("..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualize import make_figure1
from src.nmf_fit import select_k_clark_dual

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
PROCESSED_DIR = ROOT / "data" / "processed"

# %% [markdown]
# ## Load results

# %%
# with open(RESULTS_DIR / "nmf_k1_to_8.pkl", "rb") as f:
#     results = pickle.load(f)
#
# chosen_k = select_k_clark_dual(results)
# print(f"Chosen k = {chosen_k}")

# %% [markdown]
# ## Load per-cycle EMG tensor
# Shape expected: (n_cycles, 100, 12)

# %%
# V_cycles = np.load(PROCESSED_DIR / "V_cycles.npy")
# print("V_cycles shape:", V_cycles.shape)

# %% [markdown]
# ## Render Figure 1

# %%
# out_png = make_figure1(
#     V_stacked_cycles=V_cycles,
#     results=results,
#     chosen_k=chosen_k,
#     out_dir=FIGURES_DIR,
#     example_cycle_idx=0,
#     repr_muscles=(0, 3, 4, 6),  # VL, TA, BF, GM
# )
# print(f"Saved {out_png}")
