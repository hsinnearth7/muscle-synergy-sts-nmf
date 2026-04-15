# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 01. Exploratory Data Analysis (Day 2)
#
# Goal: open one Gait120 subject (S001), list every key in `RawData.mat`
# and `ProcessedData.mat`, identify the 12 EMG channels, confirm sampling
# rate, and find the sit-to-stand (STS) cycle metadata.
#
# **Day 2 decision point** (see `docs/data_structure.md`): does the
# provided `ProcessedData.mat` already match Clark et al. (2010) filter
# specs? If yes, we can use it directly and save time.

# %%
from pathlib import Path
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Make the `src/` package importable when running from notebooks/.
ROOT = Path("..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import MUSCLE_NAMES, EMG_FS, NOTCH_FREQ
from src.visualize import plot_raw_emg_example

DATA_ROOT = ROOT / "data" / "raw"
print(f"DATA_ROOT: {DATA_ROOT}")
print(f"Exists: {DATA_ROOT.exists()}")

# %% [markdown]
# ## List subjects currently present

# %%
if DATA_ROOT.exists():
    subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
    print(f"Found {len(subjects)} subject dirs: {subjects[:5]} ...")
else:
    print("data/raw/ is empty -- download Gait120 zips from Figshare first.")


# %% [markdown]
# ## Helper: open a MATLAB v7.3 (HDF5) .mat file recursively

# %%
def try_load_mat(path: Path):
    """Attempt both scipy (v<7.3) and h5py (v7.3+)."""
    try:
        data = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        return ("scipy", data)
    except NotImplementedError:
        return ("h5py", h5py.File(str(path), "r"))
    except Exception as e:
        print(f"loadmat failed: {e}")
        return ("h5py", h5py.File(str(path), "r"))


def describe_h5(node, indent: int = 0, max_depth: int = 4):
    prefix = "  " * indent
    if indent > max_depth:
        print(prefix + "...")
        return
    if hasattr(node, "items"):
        for key, value in node.items():
            if hasattr(value, "shape"):
                print(f"{prefix}{key}  shape={value.shape}  dtype={value.dtype}")
            else:
                print(f"{prefix}{key}/")
                describe_h5(value, indent + 1, max_depth)


# %% [markdown]
# ## Open S001 and explore structure

# %%
s001_dir = DATA_ROOT / "S001"
raw_mat = s001_dir / "EMG" / "RawData.mat"
proc_mat = s001_dir / "EMG" / "ProcessedData.mat"

print(f"RawData.mat present: {raw_mat.exists()}")
print(f"ProcessedData.mat present: {proc_mat.exists()}")

# %%
if raw_mat.exists():
    loader, obj = try_load_mat(raw_mat)
    print(f"Loader used: {loader}")
    if loader == "scipy":
        print("Top-level keys:", [k for k in obj.keys() if not k.startswith("__")])
    else:
        describe_h5(obj)

# %% [markdown]
# ## Day 2 deliverables
#
# 1. Fill in `docs/data_structure.md` with:
#    - Channel order of the 12 EMG muscles
#    - Sampling rate confirmation (expected: 2000 Hz)
#    - `SitToStand/` folder / marker convention
#    - Whether `ProcessedData.mat` matches Clark 2010 filter parameters
# 2. Save `figures/raw_emg_example.png` via `plot_raw_emg_example()`.
# 3. Commit with:  `feat(eda): add Gait120 structure exploration notebook`
