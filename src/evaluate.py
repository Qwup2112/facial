"""Evaluate a finished run on the test split (CLAUDE.md 5).

Writes ``{run}/test_metrics.json`` and ``{run}/confusion_matrix.png``: accuracy, macro-F1,
weighted-F1, per-class precision / recall / F1, a bootstrap 95% CI for accuracy and macro-F1,
parameter count, checkpoint size and inference time (ms/image, batch size 1). The softmax
outputs go to ``{run}/test_probs.npz`` (with ``--tta`` also the flip-averaged ones) for src.ensemble.
Settings come from the ``eval:`` block of ``configs/base.yaml``.
Reporting only: run it once per final model and never use test results to choose models,
epochs or checkpoints (those are chosen on val).

Example:
    python -m src.evaluate --run runs/resnet18_s42_20261004-1443
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from src.data.dataset import FERDataset, class_names, read_split
from src.data.transforms import build_transforms, resize_batch
from src.predict import load_run, predict_probs
from src.utils.config import load_config
from src.utils.logger import plot_confusion_matrix
from src.utils.metrics import bootstrap_ci, classification_metrics, confusion_counts


@torch.inference_mode()
def predict_probs_hflip(model, cfg: dict, df, device):
    """Softmax probabilities averaged over the image and its horizontal flip (TTA)."""
    ds = FERDataset(df, build_transforms(cfg, train=False), cfg["data"]["grayscale"])
    loader = DataLoader(ds, batch_size=2 * cfg["train"]["batch_size"], shuffle=False)
    probs = []
    for x, _ in loader:
        x = resize_batch(x.to(device), cfg)
        p = model(x).float().softmax(dim=1) + model(torch.flip(x, dims=[3])).float().softmax(dim=1)
        probs.append((p / 2).cpu())
    return torch.cat(probs).numpy()


@torch.inference_mode()
def time_inference(model, cfg: dict, x, device, warmup: int, runs: int) -> float:
    """Mean ms per image at batch size 1, including the upsampling to the model input size."""
    model = model.to(device).eval()
    x = x.to(device)
    for _ in range(warmup):
        model(resize_batch(x, cfg))
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        model(resize_batch(x, cfg))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / runs


def main():
    ap = argparse.ArgumentParser(description="Evaluate a finished run on the test split.")
    ap.add_argument("--run", required=True, help="run directory containing best.pt")
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="test for the report (default); val writes val_metrics.json for checks")
    ap.add_argument("--base-config", default="configs/base.yaml", help="file with the eval: block")
    ap.add_argument("--tta", action="store_true", help="also report horizontal-flip TTA, separately")
    ap.add_argument("--no-timing", action="store_true", help="skip the inference-time measurement")
    ap.add_argument("--cpu", action="store_true", help="use the CPU even if CUDA is available")
    args = ap.parse_args()

    run_dir = Path(args.run)
    # finished, non-dry runs only: train.py writes curves.png when training ends
    assert (run_dir / "curves.png").exists(), f"{run_dir} has no curves.png: training is not finished"
    ev = load_config(args.base_config)["eval"]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model, cfg, classes = load_run(run_dir, device)
    assert not cfg.get("run", {}).get("dry_run", False), f"{run_dir} is a dry run"
    info = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    monitor = cfg["train"]["monitor"]
    k = len(classes)
    df = read_split(cfg, args.split)
    assert class_names(df, k) == classes, "class names differ from the checkpoint"

    y_true = df["label_idx"].to_numpy()
    probs = predict_probs(model, cfg, df, device)
    y_pred = probs.argmax(axis=1)
    metrics = classification_metrics(y_true, y_pred, k)
    ci = bootstrap_ci(y_true, y_pred, k, int(ev["bootstrap_samples"]), int(ev["bootstrap_seed"]))
    report = classification_report(y_true, y_pred, labels=list(range(k)), target_names=classes,
                                   output_dict=True, zero_division=0)
    per_class = {c: {"precision": report[c]["precision"], "recall": report[c]["recall"],
                     "f1": report[c]["f1-score"], "support": int(report[c]["support"])} for c in classes}
    cm = confusion_counts(y_true, y_pred, k)
    # accuracy per image source (local / rafdb / affectnet); sources do not cover the same classes
    per_source = {}
    for src in sorted(df["source"].unique()) if "source" in df else []:
        pos = (df["source"] == src).to_numpy()
        per_source[src] = {"n_images": int(pos.sum()), "accuracy": float((y_pred[pos] == y_true[pos]).mean()),
                           "classes": sorted(df.loc[pos, "label"].unique().tolist())}

    tta = None
    # softmax outputs for src.ensemble, in the row order of data/splits/{split}.csv
    arrays = {"probs": probs.astype(np.float32), "y_true": y_true, "paths": df["path"].to_numpy(dtype=str),
              "classes": np.array(classes)}
    if args.tta:
        arrays["probs_tta"] = predict_probs_hflip(model, cfg, df, device).astype(np.float32)
        m = classification_metrics(y_true, arrays["probs_tta"].argmax(axis=1), k)
        tta = {"accuracy": m["acc"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"]}
    np.savez_compressed(run_dir / f"{args.split}_probs.npz", **arrays)

    timing = {}
    if not args.no_timing:
        x = FERDataset(df.iloc[:1], build_transforms(cfg, train=False), cfg["data"]["grayscale"])[0][0][None]
        warmup, runs = int(ev["timing_warmup"]), int(ev["timing_runs"])
        timing["cpu"] = time_inference(model, cfg, x, torch.device("cpu"), warmup, runs)
        if device.type == "cuda":
            timing["gpu"] = time_inference(model, cfg, x, device, warmup, runs)

    name, seed = cfg["model"]["name"], cfg["seed"]
    out = {
        "run": run_dir.name, "model": name, "seed": seed, "split": args.split, "n_images": len(df),
        "classes": classes, "best_epoch": info.get("epoch"), f"best_{monitor}": info.get(monitor),
        "accuracy": metrics["acc"], "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"],
        "ci95": ci, "bootstrap_samples": int(ev["bootstrap_samples"]),
        "per_class": per_class, "per_source": per_source, "confusion_matrix": cm.tolist(),
        "params": sum(p.numel() for p in model.parameters()),
        "checkpoint_mb": (run_dir / "best.pt").stat().st_size / 2**20,
        "ms_per_image": timing, "timing": {"batch_size": 1, "precision": "fp32",
                                           "cpu_threads": torch.get_num_threads(),
                                           "cpu": platform.processor(),
                                           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "torch": torch.__version__,
    }
    if tta:
        out["tta_hflip"] = tta
    prefix = "" if args.split == "test" else f"_{args.split}"
    json_path = run_dir / f"{args.split}_metrics.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    cm_path = run_dir / f"confusion_matrix{prefix}.png"
    plot_confusion_matrix(cm, classes, cm_path, title=f"{name} (seed {seed}), {args.split} set")

    (acc_lo, acc_hi), (f1_lo, f1_hi) = ci["accuracy"], ci["macro_f1"]
    print(f"{name} (seed {seed}) on {args.split} ({len(df):,} images)")
    print(f"  accuracy {metrics['acc']:.4f} [95% CI {acc_lo:.4f}, {acc_hi:.4f}] | "
          f"macro-F1 {metrics['macro_f1']:.4f} [{f1_lo:.4f}, {f1_hi:.4f}] | "
          f"weighted-F1 {metrics['weighted_f1']:.4f}")
    print("  per-class F1: " + " | ".join(f"{c} {per_class[c]['f1']:.3f}" for c in classes))
    if per_source:
        print("  accuracy by source: " + " | ".join(
            f"{s} {v['accuracy']:.3f} ({v['n_images']} images, {len(v['classes'])} classes)"
            for s, v in per_source.items()))
    if tta:
        print(f"  hflip TTA (reported separately): accuracy {tta['accuracy']:.4f} | macro-F1 {tta['macro_f1']:.4f}")
    print(f"  params {out['params']:,} | checkpoint {out['checkpoint_mb']:.1f} MB"
          + "".join(f" | {d.upper()} {ms:.2f} ms/image" for d, ms in timing.items()))
    print(f"-> {json_path}, {cm_path}, {run_dir / f'{args.split}_probs.npz'}")


if __name__ == "__main__":
    main()
