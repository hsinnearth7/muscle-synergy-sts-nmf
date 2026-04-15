# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 03. NMF Fit + Dual VAF (Day 6-7)
#
# Fit NMF for k = 1..8 with the restart policy, apply Clark 2010 dual
# criterion, save W, H and VAF summary.

# %%
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path("..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nmf_fit import fit_nmf_sweep, select_k_clark_dual
from src.vaf import global_vaf, per_muscle_vaf

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = ROOT / "data" / "processed"

# %% [markdown]
# ## Load stacked V matrix from Day 5

# %%
V_path = PROCESSED_DIR / "V_matrix.npy"
print(f"V_matrix path: {V_path}  exists: {V_path.exists()}")
# V = np.load(V_path)     # expected shape: (12, total_T) where total_T = n_cycles * 100
# print("V shape:", V.shape)

# %% [markdown]
# ## Fit the sweep

# %%
# results = fit_nmf_sweep(V, k_range=range(1, 9))
# for k, r in results.items():
#     print(f"k={k}  global VAF={r.global_vaf:.3f}  "
#           f"min muscle VAF={r.min_muscle_vaf:.3f}  "
#           f"err={r.err:.3f}  init={r.init}")

# %% [markdown]
# ## Select k via Clark 2010 dual criterion

# %%
# chosen_k = select_k_clark_dual(results)
# print(f"Chosen k = {chosen_k}")
# chosen = results[chosen_k]
# print(f"Global VAF         = {chosen.global_vaf:.3f}")
# print(f"Min per-muscle VAF = {chosen.min_muscle_vaf:.3f}")

# %% [markdown]
# ## Save W, H, and the complete results pickle

# %%
# np.save(RESULTS_DIR / "W_final.npy", chosen.W)
# np.save(RESULTS_DIR / "H_final.npy", chosen.H)
#
# with open(RESULTS_DIR / "nmf_k1_to_8.pkl", "wb") as f:
#     pickle.dump(results, f)
#
# summary_path = RESULTS_DIR / "vaf_summary.txt"
# with open(summary_path, "w", encoding="utf-8") as f:
#     f.write(f"Chosen k          : {chosen_k}\n")
#     f.write(f"Global VAF        : {chosen.global_vaf:.4f}\n")
#     f.write(f"Min per-muscle VAF: {chosen.min_muscle_vaf:.4f}\n")
#     f.write("\nper-muscle VAF:\n")
#     for i, v in enumerate(chosen.per_muscle_vaf):
#         f.write(f"  muscle[{i:2d}]: {v:.4f}\n")
# print(f"Saved {summary_path}")
