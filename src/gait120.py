"""High-level Gait120 subject-level loader.

Combines the MCOS subsystem decoder (``src.gait120_mcos``) with the
top-level MAT5 structure (``scipy.io.loadmat``) to produce raw STS
EMG cycles, MVC amplitudes, and useful metadata on a per-subject basis.

Data layout discovered during Day 2 EDA (2026-04-11):

* Top-level MAT5 variables: ``LevelWalking``, ``StairAscent``,
  ``StairDescent``, ``SlopeAscent``, ``SlopeDescent``, ``SitToStand``,
  ``StandToSit``, ``EMGs_info``, ``Markers_info``.
* Each task struct has an ``MVC_raw`` field (5 MVC trials) plus
  ``Trial01..Trial05``. Each trial stores ``EMGs_raw``, ``TotalFrame``
  (mocap-rate), and ``Step01.TargetFrame`` (STS cycle boundaries, also
  in mocap frames).
* ``EMGs_raw`` is a MATLAB ``table`` wrapped in an MCOS ``FileWrapper__``
  object stored in ``__function_workspace__``. Each trial's object
  reference is a uint32 array ``[0xDD000000, 2, 1, 1, ref_id, 1]``.
* MCOS objects are organized as 70 sequential tables in the subsystem
  cell pool. ``ref_id`` is 1-based and directly maps to the Nth table.
* Every table occupies 7 cells starting at
  ``cell[2 + 7 * (ref_id - 1)]`` where the first cell is a
  ``cell{12}`` holding one ``(nrows, 1)`` float64 column per muscle.
* EMG sampling rate (``EMGs_info.fs``) is 2000 Hz. Mocap rate is
  100 Hz, so EMG samples = mocap frames * 20.
* Muscle channel order (cell[7] of each table):
  VastusLateralis, RectusFemoris, VastusMedialis, TibialisAnterior,
  BicepsFemoris, Semitendinosus, GastrocnemuisMedialis (sic),
  GastrocnemiusLateralis, SoleusMedialis, SoleusLateralis,
  PeroneusLongus, PeroneusBrevis.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from . import MUSCLE_NAMES
from .gait120_mcos import Node, load_function_workspace, parse_subsystem

EMG_FS = 2000.0
MOCAP_FS = 100.0
EMG_TO_MOCAP = int(EMG_FS / MOCAP_FS)  # 20

MUSCLE_ORDER_SHORT = tuple(MUSCLE_NAMES)

MUSCLE_ORDER_LONG = (
    "VastusLateralis", "RectusFemoris", "VastusMedialis", "TibialisAnterior",
    "BicepsFemoris", "Semitendinosus", "GastrocnemuisMedialis",
    "GastrocnemiusLateralis", "SoleusMedialis", "SoleusLateralis",
    "PeroneusLongus", "PeroneusBrevis",
)


@dataclass
class StsTrial:
    subject_id: str
    trial_idx: int                       # 1..5
    emg_raw: np.ndarray                  # (n_samples, 12)
    total_frame: tuple                   # (start_mocap, end_mocap)
    target_frame: tuple                  # (cycle_start_mocap, cycle_end_mocap)
    mvc_raw: np.ndarray | None = None    # (n_samples_mvc, 12) per trial (optional)

    @property
    def fs(self) -> float:
        return EMG_FS

    @property
    def cycle_samples(self) -> tuple:
        """Return STS cycle [start_sample, end_sample] at EMG rate.

        MATLAB TargetFrame values are 1-based mocap frame indices, so
        frame N corresponds to EMG samples ``[(N-1)*20 ... N*20)`` at the
        2000 Hz / 100 Hz rate ratio. We convert to a half-open 0-based
        slice ``[(start-1)*20, (end)*20)`` so the slice length matches
        ``(end - start + 1)`` mocap frames worth of EMG samples.
        """
        s, e = self.target_frame
        start = max(0, (int(s) - 1) * EMG_TO_MOCAP)
        end = int(e) * EMG_TO_MOCAP
        return start, end

    def cycle(self) -> np.ndarray:
        """Return the STS cycle slice of the raw EMG: (n_cycle_samples, 12)."""
        s, e = self.cycle_samples
        s = max(0, s)
        e = min(self.emg_raw.shape[0], e)
        return self.emg_raw[s:e, :]

    def slice_envelope(self, envelope: np.ndarray) -> np.ndarray:
        """Extract the STS cycle slice from an already-preprocessed envelope.

        Useful when preprocessing the entire trial once and slicing the
        result (the Clark 2010-style order). `envelope` must have the
        same first-axis length as `emg_raw`.
        """
        if envelope.shape[0] != self.emg_raw.shape[0]:
            raise ValueError(
                f"envelope length {envelope.shape[0]} != raw length "
                f"{self.emg_raw.shape[0]}"
            )
        s, e = self.cycle_samples
        s = max(0, s)
        e = min(envelope.shape[0], e)
        return envelope[s:e, :]


@dataclass
class SubjectData:
    subject_id: str
    trials: list[StsTrial] = field(default_factory=list)
    mvc_per_muscle: np.ndarray | None = None
    """(12,) per-subject MVC amplitude per muscle from ProcessedData.mat.

    Gait120 stores per-trial MVC amplitudes (peak envelope values from
    dedicated MVC recordings). We take ``np.max`` across 5 MVC trials
    to select the best effort per muscle — this is the standard
    convention for MVC normalization in EMG research.
    """

    def get_trial(self, idx: int) -> StsTrial:
        return self.trials[idx]

    def __len__(self) -> int:
        return len(self.trials)


# ---------------------------------------------------------------------------
# Core accessors
# ---------------------------------------------------------------------------


#: Each table in the subsystem pool occupies exactly 7 cells: the data
#: cell (cell{12}), then 6 metadata cells (ndims, nrows, empty rownames,
#: nvars, varnames cell, props struct). This stride was validated against
#: Gait120 S001 and all 20 test subjects produced consistent structure.
_TABLE_STRIDE_CELLS = 7
#: Cell index of the first table's data cell. The preceding cells are
#: cell[0] = metadata blob, cell[1] = empty placeholder.
_FIRST_TABLE_OFFSET = 2


def _table_first_cell(ref_id: int) -> int:
    """Cell index of the ``data`` cell for table (ref_id), 1-based ref_id.

    Valid ref_ids are 1..N where N is the total number of tables in the
    subsystem pool. For Gait120: 70 tables (7 tasks x 5 MVC + 5 EMG).
    """
    if ref_id < 1:
        raise ValueError(f"ref_id must be >= 1, got {ref_id}")
    return _FIRST_TABLE_OFFSET + _TABLE_STRIDE_CELLS * (ref_id - 1)


def _extract_table_columns(data_cell: Node, ref_id: int = -1) -> np.ndarray:
    """Given the ``cell{12}`` data container of a table, return (nrows, 12) float64.

    Validates that the cell is in fact a 12-element numeric cell array;
    raises a descriptive error with the ref_id for debugging if a
    table-stride miscount landed the index on an unrelated cell.
    """
    tag = f" (ref_id={ref_id})" if ref_id >= 0 else ""
    if data_cell.kind != "cell":
        raise ValueError(
            f"expected a 12-column cell array{tag}, got kind={data_cell.kind}; "
            f"this usually means the table-stride mapping is off for this subject"
        )
    subs = data_cell.value
    if len(subs) != 12:
        raise ValueError(
            f"expected 12 columns{tag}, got {len(subs)}; "
            f"possible ref_id or table-stride miscount"
        )
    cols = []
    for i, sub in enumerate(subs):
        if sub.kind != "numeric" or sub.value is None:
            raise ValueError(
                f"column {i} is not numeric{tag}: kind={sub.kind}"
            )
        arr = np.asarray(sub.value, dtype=np.float64).reshape(-1)
        cols.append(arr)
    n_samples = cols[0].size
    for i, c in enumerate(cols):
        if c.size != n_samples:
            raise ValueError(
                f"inconsistent column lengths{tag}: col0={n_samples}, "
                f"col{i}={c.size}"
            )
    return np.stack(cols, axis=1)   # (nrows, 12)


def _ref_id_from_opaque(opaque_record: dict) -> int:
    """Extract the ref_id (1-based) from a top-level MatlabOpaque EMG record."""
    if isinstance(opaque_record, np.void):
        arr = opaque_record["arr"]
    elif isinstance(opaque_record, dict):
        arr = opaque_record.get("arr")
    else:
        arr = getattr(opaque_record, "arr", None)
    if arr is None:
        raise ValueError("opaque record missing 'arr' field")
    # The array is shape (6,) or (6, 1) uint32:
    # [0xDD000000, 2, 1, 1, ref_id, 1]
    flat = np.asarray(arr).ravel()
    if flat.size < 5:
        raise ValueError(
            f"opaque 'arr' too short: expected >= 5 elements, got {flat.size}"
        )
    ref_id = int(flat[4])
    if ref_id < 1:
        raise ValueError(f"ref_id must be >= 1, got {ref_id}")
    return ref_id


# ---------------------------------------------------------------------------
# Per-subject loader
# ---------------------------------------------------------------------------


def load_subject(subject_dir: Path | str) -> SubjectData:
    """Load all 5 Sit-to-Stand trials and per-muscle MVCs for one subject.

    Parameters
    ----------
    subject_dir : path
        Directory containing ``EMG/RawData.mat`` and ``EMG/ProcessedData.mat``.
        e.g., ``Gait120_001_to_010/S001``.
    """
    subject_dir = Path(subject_dir)
    subject_id = subject_dir.name
    raw_path = subject_dir / "EMG" / "RawData.mat"
    proc_path = subject_dir / "EMG" / "ProcessedData.mat"

    # 1. Decode the subsystem cell pool once.
    fw = load_function_workspace(raw_path)
    pool = parse_subsystem(fw)
    n_cells = len(pool.data_cells)

    # 2. Grab the top-level struct to retrieve ref_ids and metadata.
    top = loadmat(str(raw_path), squeeze_me=True, struct_as_record=False)
    sts = top["SitToStand"]
    trials: list[StsTrial] = []
    for t_idx in range(1, 6):
        trial_name = f"Trial0{t_idx}"
        tr = getattr(sts, trial_name, None)
        if tr is None:
            continue
        rec = tr.EMGs_raw[0]
        ref_id = _ref_id_from_opaque(rec)
        cell_idx = _table_first_cell(ref_id)
        if cell_idx >= n_cells:
            raise IndexError(
                f"[{subject_id} T{t_idx}] ref_id={ref_id} maps to cell index "
                f"{cell_idx} but the subsystem pool has only {n_cells} cells. "
                f"Possible subject-specific table-stride divergence."
            )
        data_cell = pool.data_cells[cell_idx]
        emg = _extract_table_columns(data_cell, ref_id=ref_id)

        total_frame_arr = np.asarray(tr.TotalFrame).ravel()
        target_frame_arr = np.asarray(tr.Step01.TargetFrame).ravel()
        if target_frame_arr.size != 2:
            raise ValueError(
                f"[{subject_id} T{t_idx}] TargetFrame expected length 2, "
                f"got {target_frame_arr.size}: {target_frame_arr.tolist()}"
            )
        if total_frame_arr.size != 2:
            raise ValueError(
                f"[{subject_id} T{t_idx}] TotalFrame expected length 2, "
                f"got {total_frame_arr.size}: {total_frame_arr.tolist()}"
            )
        total_frame = (int(total_frame_arr[0]), int(total_frame_arr[1]))
        target_frame = (int(target_frame_arr[0]), int(target_frame_arr[1]))

        trials.append(
            StsTrial(
                subject_id=subject_id,
                trial_idx=t_idx,
                emg_raw=emg,
                total_frame=total_frame,
                target_frame=target_frame,
            )
        )

    # 3. Per-subject MVC amplitudes from ProcessedData.mat.
    #    Each trial stores peak envelope amplitudes from dedicated MVC
    #    recordings (12 values). We take np.max across 5 MVC trials to
    #    select the best effort per muscle (standard MVC convention).
    mvc_per_muscle: np.ndarray | None = None
    if proc_path.exists():
        try:
            proc = loadmat(str(proc_path), squeeze_me=True, struct_as_record=False)
            proc_sts = proc["SitToStand"]
            per_trial = []
            for t_idx in range(1, 6):
                tr = getattr(proc_sts, f"Trial0{t_idx}", None)
                if tr is None:
                    continue
                mvcs = np.asarray(tr.MVCs, dtype=np.float64).ravel()
                if mvcs.size == 12:
                    per_trial.append(mvcs)
            if per_trial:
                mvc_per_muscle = np.max(np.stack(per_trial, axis=0), axis=0)
        except Exception as exc:
            # Distinguish from the FileNotFoundError branch: if the file
            # exists but fails to parse, warn so the caller notices the
            # silent fallback to max normalization.
            warnings.warn(
                f"[{subject_id}] ProcessedData.mat parse failed: {exc!r}; "
                f"falling back to per-subject max normalization",
                RuntimeWarning,
                stacklevel=2,
            )
            mvc_per_muscle = None

    return SubjectData(subject_id=subject_id, trials=trials, mvc_per_muscle=mvc_per_muscle)


def discover_subjects(root_dirs: list[Path | str]) -> list[Path]:
    """Return all SXXX subject directories under the given Gait120 zip folders."""
    out: list[Path] = []
    for root in root_dirs:
        root = Path(root)
        if not root.exists():
            continue
        for p in sorted(root.iterdir()):
            if p.is_dir() and p.name.startswith("S") and (p / "EMG" / "RawData.mat").exists():
                out.append(p)
    return out
