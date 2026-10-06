"""Label the images of a split with a trained run and draw a labeled image grid.

Writes ``{run}/predictions_{split}.csv`` (true label, predicted label, confidence and
class probabilities per image) and ``reports/figures/predictions_{model}_s{seed}_{split}.png``
(one row per true class: correct and misclassified examples with the predicted label).
Reporting only: never use test predictions to choose models, epochs or checkpoints.

Example:
    python -m src.predict --run runs/resnet18_s42_20261004-1400
"""
import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.dataset import FERDataset, class_names, read_split
from src.data.transforms import build_transforms, resize_batch
from src.models import build_model
from src.utils.logger import INK, set_report_style

GOOD, CRITICAL = "#0ca30c", "#d03b3b"   # status colors: correct / misclassified


def load_run(run_dir: Path, device):
    """Rebuild a run's model from ``best.pt``; returns (model, cfg, class names)."""
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    cfg = ckpt["cfg"]
    build_cfg = copy.deepcopy(cfg)
    build_cfg["model"]["pretrained"] = False   # weights come from the checkpoint
    model = build_model(ckpt["model_name"], len(ckpt["classes"]), build_cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, ckpt["classes"]


@torch.inference_mode()
def predict_probs(model, cfg: dict, df: pd.DataFrame, device) -> np.ndarray:
    """Softmax probabilities (N, num_classes) with the val/test transform."""
    ds = FERDataset(df, build_transforms(cfg, train=False), cfg["data"]["grayscale"])
    loader = DataLoader(ds, batch_size=2 * cfg["train"]["batch_size"], shuffle=False)
    probs = [model(resize_batch(x.to(device), cfg)).float().softmax(dim=1).cpu() for x, _ in loader]
    return torch.cat(probs).numpy()


def label_table(df: pd.DataFrame, probs: np.ndarray, classes: list[str]) -> pd.DataFrame:
    """One row per image: true label, predicted label, confidence and every class probability."""
    out = df[["path", "label", "label_idx"]].copy()
    out["pred_idx"] = probs.argmax(axis=1)
    out["pred"] = [classes[i] for i in out["pred_idx"]]
    out["confidence"] = probs.max(axis=1)
    out["correct"] = out["pred_idx"] == out["label_idx"]
    for i, name in enumerate(classes):
        out[f"p_{name}"] = probs[:, i]
    return out


def plot_grid(table: pd.DataFrame, classes: list[str], per_class: int, seed: int, title: str, out_png) -> None:
    """Rows = true class; left block correct, right block misclassified (seeded random picks)."""
    set_report_style()
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(2 * per_class * 0.95 + 1.3, len(classes) * 1.1 + 0.9), layout="constrained")
    fig.suptitle(title, fontsize=11)
    left, right = fig.subfigures(1, 2, wspace=0.03)
    left.supylabel("True label", fontsize=10)
    for sub, correct, header in [(left, True, "Correct"), (right, False, "Misclassified (×)")]:
        sub.suptitle(header, fontsize=10)
        axes = sub.subplots(len(classes), per_class, squeeze=False)
        for r, name in enumerate(classes):
            pool = table[(table["label"] == name) & (table["correct"] == correct)]
            picks = pool.iloc[rng.permutation(len(pool))[:per_class]]
            for c in range(per_class):
                ax = axes[r, c]
                ax.set_xticks([])
                ax.set_yticks([])
                if c < len(picks):
                    row = picks.iloc[c]
                    with Image.open(row["path"]) as im:
                        ax.imshow(im.convert("L"), cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
                    mark = "" if correct else "× "
                    ax.set_title(f"{mark}{row['pred']} {row['confidence']:.2f}", fontsize=7.5, color=INK, pad=2)
                    for spine in ax.spines.values():  # full frame (report style hides top/right)
                        spine.set_visible(True)
                        spine.set_edgecolor(GOOD if correct else CRITICAL)
                        spine.set_linewidth(1.0 if correct else 1.6)
                else:
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                if c == 0 and correct:
                    ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9.5, labelpad=4)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Label a split with a trained run and draw a labeled grid.")
    ap.add_argument("--run", required=True, help="run directory containing best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--per-class", type=int, default=4, help="correct and misclassified examples per class")
    ap.add_argument("--out", help="figure path (default: reports/figures/predictions_{model}_s{seed}_{split}.png)")
    ap.add_argument("--cpu", action="store_true", help="use the CPU even if CUDA is available")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    run_dir = Path(args.run)
    model, cfg, classes = load_run(run_dir, device)
    df = read_split(cfg, args.split)
    assert class_names(df, len(classes)) == classes, "class names differ from the checkpoint"

    table = label_table(df, predict_probs(model, cfg, df, device), classes)
    csv_path = run_dir / f"predictions_{args.split}.csv"
    table.to_csv(csv_path, index=False)

    name, seed = cfg["model"]["name"], cfg["seed"]
    out_png = Path(args.out or f"reports/figures/predictions_{name}_s{seed}_{args.split}.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    title = f"{name} (seed {seed}): predicted label and confidence on {args.split} images"
    plot_grid(table, classes, args.per_class, seed, title, out_png)
    print(f"labeled {len(table):,} {args.split} images -> {csv_path}")
    print(f"figure -> {out_png}")


if __name__ == "__main__":
    main()
