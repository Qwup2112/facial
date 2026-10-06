"""Per-epoch CSV logging, training-curve plots and the confusion-matrix figure."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

# Validated categorical slots 1-2 (light surface) and neutral inks.
TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Sequential blue ramp (steps 100-700) for magnitude; 0 recedes to the white surface.
SEQ_BLUE = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def set_report_style() -> None:
    """Report figure style: Times New Roman (serif fallback), recessive axes, 300 dpi."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


class CSVLogger:
    """Append one dict per epoch to a CSV file; the header comes from the first row."""

    def __init__(self, path):
        self.path = Path(path)
        self.fields = None

    def log(self, row: dict) -> None:
        """Write one row (every row must have the same keys)."""
        first = self.fields is None
        if first:
            self.fields = list(row)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            if first:
                writer.writeheader()
            writer.writerow(row)


def plot_curves(log_csv, out_png, title: str | None = None) -> None:
    """Loss and macro-F1 vs. epoch (train and val); vertical lines mark where a stage starts."""
    set_report_style()
    df = pd.read_csv(log_csv)
    stage_starts = df.loc[df["stage"] != df["stage"].shift()].iloc[1:]
    best = df.loc[df["val_macro_f1"].idxmax()]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    for ax, key, label in [(axes[0], "loss", "Loss"), (axes[1], "macro_f1", "Macro-F1")]:
        ax.plot(df["epoch"], df[f"train_{key}"], color=TRAIN_COLOR, lw=1.5, label="Train")
        ax.plot(df["epoch"], df[f"val_{key}"], color=VAL_COLOR, lw=1.5, label="Validation")
        for _, row in stage_starts.iterrows():
            x = row["epoch"] - 0.5
            ax.axvline(x, color=INK_MUTED, lw=0.8)
            ax.text(x, 0.98, f" {row['stage']}", transform=ax.get_xaxis_transform(),
                    va="top", ha="left", fontsize=8, color=INK_MUTED)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)

    # Selective direct label: the epoch kept as best.pt.
    axes[1].plot(best["epoch"], best["val_macro_f1"], "o", ms=6, color=VAL_COLOR,
                 mec="white", mew=1.5, zorder=3)
    axes[1].annotate(f"best {best['val_macro_f1']:.3f} (epoch {int(best['epoch'])})",
                     (best["epoch"], best["val_macro_f1"]), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=8, color=INK)
    axes[0].legend(loc="upper right")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def _relative_luminance(rgb) -> float:
    """WCAG relative luminance of an sRGB color given as floats in [0, 1]."""
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def plot_confusion_matrix(cm, classes: list[str], out_png, title: str | None = None) -> None:
    """Row-normalized confusion matrix: each cell is the % of the true class (row) predicted as
    the column class, so the diagonal is the per-class recall."""
    set_report_style()
    cm = np.asarray(cm, dtype=float)
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    k = len(classes)
    ink_lum = _relative_luminance(to_rgb(INK))

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    mesh = ax.pcolormesh(share, cmap=cmap, vmin=0, vmax=1, edgecolors="white", linewidth=1.5)
    ax.set_xlim(0, k)
    ax.set_ylim(k, 0)                      # first class on top
    ax.set_aspect("equal")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(k):
        for j in range(k):
            lum = _relative_luminance(cmap(share[i, j])[:3])
            # whichever text color has the higher WCAG contrast on this cell
            white = 1.05 / (lum + 0.05) > (lum + 0.05) / (ink_lum + 0.05)
            ax.text(j + 0.5, i + 0.5, f"{100 * share[i, j]:.0f}", ha="center", va="center",
                    fontsize=8, color="white" if white else INK)
    ax.set_xticks(np.arange(k) + 0.5, classes, rotation=45, ha="right")
    ax.set_yticks(np.arange(k) + 0.5, classes)
    ax.tick_params(length=0)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0], labels=["0", "25", "50", "75", "100"])
    cbar.set_label("Share of the true class (%)")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)
    if title:
        ax.set_title(title, fontsize=10)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
