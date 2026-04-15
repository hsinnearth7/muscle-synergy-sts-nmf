# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 05. Cross-Subject W Consistency (Day 11, OPTIONAL)
#
# For each subject, fit NMF at the chosen k and align the resulting W
# matrices to a common reference via the Hungarian algorithm. Then report
# the mean pairwise cosine similarity.

# %%
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path("..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nmf_fit import fit_nmf_with_restarts
from src.align import align_synergies, cross_subject_similarity

# %% [markdown]
# ## Per-subject fit

# %%
# subject_Vs: dict[str, np.ndarray]   # load from data/processed/per_subject/*.npy
# chosen_k = 4
# subject_Ws = {sid: fit_nmf_with_restarts(V, k=chosen_k).W
#               for sid, V in subject_Vs.items()}
#
# ref_sid = sorted(subject_Ws.keys())[0]
# W_ref = subject_Ws[ref_sid]
# aligned = [W_ref] + [
#     align_synergies(W_ref, W)[0]  # [0] = permuted W, [1] = col_ind
#     for sid, W in sorted(subject_Ws.items()) if sid != ref_sid
# ]
#
# sim = cross_subject_similarity(aligned)
# print(f"Mean off-diagonal cosine similarity = "
#       f"{(sim.sum() - np.trace(sim)) / (sim.size - sim.shape[0]):.3f}")
#
# fig, ax = plt.subplots(figsize=(6, 5))
# sns.heatmap(sim, vmin=0, vmax=1, cmap="viridis", ax=ax, cbar_kws={"label": "cosine"})
# ax.set_title("Cross-subject W similarity")
# fig.tight_layout()
# fig.savefig(ROOT / "figures" / "figure2_cross_subject.png", dpi=300, bbox_inches="tight")
