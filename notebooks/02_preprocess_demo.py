# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 02. Preprocess Demo (Day 3)
#
# Run `src.preprocess.preprocess_emg` on one subject and (if available)
# compare the resulting envelope against Gait120's own `ProcessedData.mat`
# via RMSE. A small RMSE indicates our Clark 2010-aligned pipeline is
# consistent with the provider's processing.

# %%
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import MUSCLE_NAMES, EMG_FS, NOTCH_FREQ
from src.preprocess import preprocess_emg, rmse

DATA_ROOT = ROOT / "data" / "raw"

# %% [markdown]
# ## Load one subject's raw EMG
# The helper below is a placeholder -- adapt after Day 2 confirms the
# actual key names inside RawData.mat.

# %%
def load_raw_emg_s001() -> np.ndarray:
    """Return raw_emg shape (n_samples, 12). Adapt after Day 2 EDA."""
    raise NotImplementedError(
        "Fill in after Day 2 EDA confirms Gait120 .mat structure."
    )


# raw = load_raw_emg_s001()
# envelope = preprocess_emg(raw, fs=EMG_FS, notch_freq=NOTCH_FREQ)
# print("Envelope shape:", envelope.shape)

# %% [markdown]
# ## Sanity plot: compare raw vs envelope for one channel

# %%
def _quick_compare(raw_ch: np.ndarray, env_ch: np.ndarray, fs: float,
                   muscle: str = "VL", seconds: float = 5.0):
    n = int(seconds * fs)
    t = np.arange(n) / fs
    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    axes[0].plot(t, raw_ch[:n], linewidth=0.6)
    axes[0].set_title(f"Raw EMG -- {muscle}")
    axes[0].set_ylabel("raw")
    axes[1].plot(t, env_ch[:n], color="#E76F51", linewidth=1.2)
    axes[1].set_title(f"Envelope (40-450 bp, 4 Hz lp) -- {muscle}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("envelope")
    fig.tight_layout()
    plt.show()


# _quick_compare(raw[:, 0], envelope[:, 0], EMG_FS, muscle="VL")

# %% [markdown]
# ## Optional: RMSE vs Gait120 ProcessedData.mat
#
# If the provider's `ProcessedData.mat` is compatible (same convention),
# compute `rmse(our_envelope, provider_envelope)` per channel. A value
# near the per-channel dynamic range is fine; order-of-magnitude lower
# means the pipelines match.
