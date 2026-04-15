"""Day 5 pipeline driver: build the NMF input V matrix from 20 subjects.

Reads all SXXX subject directories under the two downloaded Gait120 zips,
extracts the STS cycle from each trial, applies the Clark 2010 aligned
preprocessing pipeline, time-normalizes to 100 points, normalizes by MVC
(fallback: subject max), and stacks into the final V matrix.

Outputs:
    data/processed/V_matrix.npy        shape (12, n_cycles * 100)
    data/processed/V_cycles.npy        shape (n_cycles, 100, 12)
    data/processed/cycle_metadata.csv  one row per cycle
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make src importable when run from repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gait120 import discover_subjects, load_subject, EMG_FS
from src.preprocess import preprocess_emg
from src.segment import resample_cycle_poly
from src.normalize import normalize_by_subject


DATA_ROOTS = [ROOT / "Gait120_001_to_010", ROOT / "Gait120_011_to_020"]
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LEN = 100  # time-normalized length per cycle


def main() -> None:
    subject_dirs = discover_subjects(DATA_ROOTS)
    print(f"Found {len(subject_dirs)} subject dirs")

    all_cycles = []      # list of (n_cycles_i, 100, 12) per subject
    metadata = []
    skipped = []

    for sdir in subject_dirs:
        try:
            subj = load_subject(sdir)
        except Exception as exc:
            skipped.append((sdir.name, f"load: {exc}"))
            continue

        per_subject_cycles = []
        for tr in subj.trials:
            # Clark et al. (2010) protocol: filter the full trial first,
            # THEN extract the marker-bounded STS cycle. This avoids the
            # edge artefact that would otherwise contaminate the short slice.
            if tr.emg_raw.shape[0] < 400:
                skipped.append(
                    (f"{subj.subject_id}/T{tr.trial_idx}", "trial too short")
                )
                continue
            try:
                trial_envelope = preprocess_emg(
                    tr.emg_raw, fs=EMG_FS, notch_freq=60.0
                )
            except Exception as exc:
                skipped.append(
                    (f"{subj.subject_id}/T{tr.trial_idx}", f"preprocess: {exc}")
                )
                continue
            cycle_env = tr.slice_envelope(trial_envelope)
            if cycle_env.shape[0] < 200:
                skipped.append(
                    (f"{subj.subject_id}/T{tr.trial_idx}", "cycle slice too short")
                )
                continue
            resampled = resample_cycle_poly(cycle_env, target_len=TARGET_LEN)
            per_subject_cycles.append(resampled)
            metadata.append(
                dict(
                    subject_id=subj.subject_id,
                    trial_idx=tr.trial_idx,
                    trial_nsamples=tr.emg_raw.shape[0],
                    cycle_nsamples=cycle_env.shape[0],
                    cycle_seconds=cycle_env.shape[0] / EMG_FS,
                )
            )

        if not per_subject_cycles:
            continue

        stacked = np.stack(per_subject_cycles, axis=0)   # (n, 100, 12)
        if subj.mvc_per_muscle is not None:
            normed = normalize_by_subject(stacked, method="mvc", mvc=subj.mvc_per_muscle)
        else:
            normed = normalize_by_subject(stacked, method="max")
        all_cycles.append(normed)
        print(
            f"  {subj.subject_id}: {stacked.shape[0]} cycles  "
            f"mvc={'yes' if subj.mvc_per_muscle is not None else 'no'}  "
            f"envelope max={normed.max():.2f}"
        )

    if not all_cycles:
        if skipped:
            print(f"\nSkipped {len(skipped)} subjects/trials:")
            for name, why in skipped[:10]:
                print(f"  {name}: {why}")
        print("No cycles collected; aborting.")
        raise SystemExit(1)

    V_cycles = np.concatenate(all_cycles, axis=0)   # (n_total, 100, 12)
    # resample_poly can reintroduce tiny numerical negatives after our
    # envelope clip; NMF requires strictly non-negative inputs.
    V_cycles = np.clip(V_cycles, 0.0, None)
    n_total = V_cycles.shape[0]
    print(f"\nTotal cycles collected: {n_total}")
    print(f"V_cycles shape: {V_cycles.shape}  range [{V_cycles.min():.4f}, {V_cycles.max():.4f}]")

    # NMF convention: (n_muscles, total_time). Flatten (n_cycles*T, 12) then transpose.
    V_flat = V_cycles.reshape(-1, 12)              # (n_total * 100, 12)
    V = V_flat.T                                    # (12, n_total * 100)

    np.save(OUT_DIR / "V_cycles.npy", V_cycles)
    np.save(OUT_DIR / "V_matrix.npy", V)
    pd.DataFrame(metadata).to_csv(OUT_DIR / "cycle_metadata.csv", index=False)

    print(f"\nSaved:")
    print(f"  {OUT_DIR / 'V_cycles.npy'}  shape={V_cycles.shape}")
    print(f"  {OUT_DIR / 'V_matrix.npy'}  shape={V.shape}")
    print(f"  {OUT_DIR / 'cycle_metadata.csv'}  ({len(metadata)} rows)")

    if skipped:
        print(f"\nSkipped {len(skipped)} trials:")
        for name, why in skipped[:10]:
            print(f"  {name}: {why}")


if __name__ == "__main__":
    main()
