"""Figure 1 main composite (4 panels) for the project deliverable."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from . import MUSCLE_NAMES
from .nmf_fit import NMFResult


def make_figure1(
    V_stacked_cycles: np.ndarray,   # (n_cycles, T, n_muscles)
    results: dict[int, NMFResult],
    chosen_k: int,
    out_dir: Path | str = "figures",
    muscle_names: Sequence[str] = MUSCLE_NAMES,
    example_cycle_idx: int = 0,
    repr_muscles: Sequence[int] = (0, 3, 4, 6),   # VL, TA, BF, GM
    title_suffix: str | None = None,
) -> Path:
    """Produce the 4-panel main figure and save PNG + PDF.

    Panel A: W heatmap (n_muscles x k) with annotated weights.
    Panel B: Mean +/- SD temporal activation H across cycles.
    Panel C: Dual VAF curves (global + min per-muscle) vs k.
    Panel D: Reconstruction overlay for 4 representative muscles.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = results[chosen_k]
    W = chosen.W                       # (n_muscles, k)
    H = chosen.H                       # (k, n_cycles * T)

    n_cycles, t_len, n_muscles = V_stacked_cycles.shape
    H_per_cycle = H.reshape(chosen_k, n_cycles, t_len)
    H_mean_cycle = H_per_cycle.mean(axis=1)
    H_std_cycle = H_per_cycle.std(axis=1)

    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

    # Layout:
    #   A A B B
    #   A A B B
    #   C C D E
    #   C C F G
    # where D=VL, E=TA, F=BF, G=GM
    mosaic = [
        ["A", "A", "B", "B"],
        ["A", "A", "B", "B"],
        ["C", "C", "D0", "D1"],
        ["C", "C", "D2", "D3"],
    ]
    fig, axd = plt.subplot_mosaic(
        mosaic, figsize=(14, 11), constrained_layout=True,
    )

    # Panel A -- W heatmap
    sns.heatmap(
        W,
        annot=True, fmt=".2f", cmap="viridis",
        xticklabels=[f"Syn {i + 1}" for i in range(chosen_k)],
        yticklabels=list(muscle_names),
        ax=axd["A"],
        cbar_kws={"label": "Weight"},
    )
    axd["A"].set_title(f"A. Muscle Synergy Weights (W), k = {chosen_k}")

    # Panel B -- H mean +/- SD
    colors = plt.cm.tab10(np.arange(chosen_k))
    x = np.linspace(0, 100, t_len)
    for i in range(chosen_k):
        axd["B"].plot(
            x, H_mean_cycle[i],
            color=colors[i], linewidth=2,
            label=f"Syn {i + 1}",
        )
        axd["B"].fill_between(
            x,
            H_mean_cycle[i] - H_std_cycle[i],
            H_mean_cycle[i] + H_std_cycle[i],
            color=colors[i], alpha=0.15,
        )
    axd["B"].set_xlabel("STS cycle (%)")
    axd["B"].set_ylabel("Activation")
    axd["B"].set_title("B. Synergy Activation over STS Cycle (mean +/- SD)")
    axd["B"].legend(loc="upper right", framealpha=0.9)
    axd["B"].grid(alpha=0.3)

    # Panel C -- VAF vs k
    ks = sorted(results.keys())
    global_vafs = [results[k].global_vaf for k in ks]
    min_muscle_vafs = [results[k].min_muscle_vaf for k in ks]
    axd["C"].plot(
        ks, global_vafs, "o-", color="#264653",
        linewidth=2, markersize=8, label="Global VAF",
    )
    axd["C"].plot(
        ks, min_muscle_vafs, "s--", color="#E76F51",
        linewidth=2, markersize=8, label="Min per-muscle VAF",
    )
    axd["C"].axhline(0.90, ls=":", color="gray", alpha=0.7,
                     label="Clark 2010 threshold (0.90)")
    axd["C"].axvline(chosen_k, ls=":", color="red", alpha=0.5)
    axd["C"].set_xlabel("Number of synergies (k)")
    axd["C"].set_ylabel("VAF")
    axd["C"].set_title("C. Clark 2010 Dual-Criterion VAF")
    axd["C"].set_ylim(0, 1.02)
    axd["C"].set_xticks(ks)
    axd["C"].legend(loc="lower right", fontsize=9)
    axd["C"].grid(alpha=0.3)

    # Panel D -- Reconstruction of 4 representative muscles (2x2 proper subplots)
    recon_example = W @ H_per_cycle[:, example_cycle_idx, :]   # (n_muscles, t_len)
    for pos, m in enumerate(repr_muscles):
        key = f"D{pos}"
        ax = axd[key]
        original = V_stacked_cycles[example_cycle_idx, :, m]
        recon_m = recon_example[m]
        ax.plot(x, original, "k", linewidth=1.8, label="Original")
        ax.plot(x, recon_m, "r--", linewidth=1.8, label="NMF reconstruction")
        muscle_vaf = chosen.per_muscle_vaf[m]
        ax.set_title(f"{muscle_names[m]}  (VAF={muscle_vaf:.2f})",
                     fontsize=10)
        ax.set_xlabel("cycle (%)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        if pos == 0:
            ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # Section label for Panel D (matplotlib subplot_mosaic doesn't have a
    # group title, so stick one on the top-left D cell's top).
    axd["D0"].text(
        -0.18, 1.35,
        "D. Reconstruction Examples  (cycle 0, 4 representative muscles)",
        transform=axd["D0"].transAxes,
        fontsize=11, fontweight="regular", ha="left",
    )

    if title_suffix:
        fig.suptitle(title_suffix, fontsize=10, y=1.02)

    png_path = out_dir / "figure1_main.png"
    pdf_path = out_dir / "figure1_main.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def plot_raw_emg_example(
    raw: np.ndarray,
    fs: float,
    muscle_names: Sequence[str] = MUSCLE_NAMES,
    seconds: float = 5.0,
    out_path: Path | str = "figures/raw_emg_example.png",
) -> Path:
    """Quick 12-channel x N-second overview plot for Day 2 EDA."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[:, None]
    elif raw.ndim != 2:
        raise ValueError(f"raw must be 1D or 2D; got shape {raw.shape}")

    n_samples = min(raw.shape[0], int(round(seconds * fs)))
    snippet = raw[:n_samples]
    t = np.arange(snippet.shape[0]) / fs
    n_channels = snippet.shape[1]

    fig, ax = plt.subplots(n_channels, 1, figsize=(10, 12), sharex=True)
    ax = np.atleast_1d(ax)
    for i in range(n_channels):
        ax[i].plot(t, snippet[:, i], linewidth=0.8)
        ax[i].set_ylabel(muscle_names[i], rotation=0, labelpad=20, ha="right")
        ax[i].grid(alpha=0.2)
    ax[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Raw EMG example ({seconds:.0f} s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
