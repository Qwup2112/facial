"""Ensembles of the final runs: average their softmax outputs (reported separately, CLAUDE.md 5).

Reads ``{run}/{split}_probs.npz`` written by src.evaluate (with ``--tta`` it also holds the
horizontal-flip outputs) for the final runs chosen by src.aggregate (latest config per model).
Reports the average of all final runs and, per model, the average of its seeds, with and
without TTA. Like a single model, an ensemble is fixed on val first and run on test once.

Writes ``reports/ensemble_{split}.json``, ``reports/ensemble_{split}.md`` and the confusion
matrix of the all-runs ensemble to ``reports/figures/confusion_matrix_ensemble_{split}.png``.

Examples:
    python -m src.ensemble --split val
    python -m src.ensemble --split test
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src.aggregate import MODEL_ORDER, finished_runs, select
from src.utils.config import load_config
from src.utils.logger import plot_confusion_matrix
from src.utils.metrics import bootstrap_ci, classification_metrics, confusion_counts

ALL = "all final runs"


def load_probs(run_dir: Path, split: str) -> dict:
    """Arrays saved by src.evaluate for one run and split."""
    path = run_dir / f"{split}_probs.npz"
    assert path.exists(), f"{path} is missing: python -m src.evaluate --run {run_dir.as_posix()} --split {split} --tta"
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def score(y_true, probs, classes: list[str], sources: np.ndarray, ev: dict) -> dict:
    """Metrics of averaged probabilities, in the layout of src.evaluate."""
    k = len(classes)
    y_pred = probs.argmax(axis=1)
    m = classification_metrics(y_true, y_pred, k)
    report = classification_report(y_true, y_pred, labels=list(range(k)), target_names=classes,
                                   output_dict=True, zero_division=0)
    return {
        "accuracy": m["acc"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
        "ci95": bootstrap_ci(y_true, y_pred, k, int(ev["bootstrap_samples"]), int(ev["bootstrap_seed"])),
        "per_class_f1": {c: report[c]["f1-score"] for c in classes},
        "per_source_accuracy": {s: float((y_pred[sources == s] == y_true[sources == s]).mean())
                                for s in sorted(set(sources))},
        "confusion_matrix": confusion_counts(y_true, y_pred, k).tolist(),
    }


def to_markdown(split: str, results: list[dict], n_images: int) -> str:
    """Table of every ensemble, with and without TTA."""
    lines = [f"# Ensembles on {split} ({n_images:,} images)", "",
             "Average of the softmax outputs of the final runs (src.aggregate). TTA: each model also sees "
             "the horizontally flipped image. Reported separately from the single models.", "",
             "| Ensemble | Runs | TTA | Accuracy (%) | Macro-F1 | Macro-F1 95% CI |", "|---|---|---|---|---|---|"]
    for r in results:
        lo, hi = r["ci95"]["macro_f1"]
        lines.append(f"| {r['ensemble']} | {len(r['runs'])} | {'yes' if r['tta'] else 'no'} "
                     f"| {100 * r['accuracy']:.2f} | {r['macro_f1']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines += ["", "Runs: " + ", ".join(results[0]["runs"])]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Average the softmax outputs of the final runs.")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--runs", default="runs", help="directory with the run folders")
    ap.add_argument("--base-config", default="configs/base.yaml", help="file with the data: and eval: blocks")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--force", action="store_true", help="overwrite an existing test result")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_json = out_dir / f"ensemble_{args.split}.json"
    if args.split == "test" and out_json.exists() and not args.force:
        raise SystemExit(f"{out_json} exists: the test set is used once per final ensemble (--force to redo)")
    cfg = load_config(args.base_config)
    sel, _ = select(finished_runs(Path(args.runs)))
    sel = sel.assign(rank=sel["model"].map(MODEL_ORDER.index)).sort_values(["rank", "seed"])
    data = {r.run: load_probs(r.dir, args.split) for r in sel.itertuples()}

    first = next(iter(data.values()))
    y_true, paths, classes = first["y_true"], first["paths"], first["classes"].tolist()
    for run, d in data.items():
        assert (d["paths"] == paths).all() and d["classes"].tolist() == classes, f"{run}: rows or classes differ"
    split_csv = pd.read_csv(Path(cfg["data"]["splits_dir"]) / f"{args.split}.csv")
    sources = pd.Series(paths).map(dict(zip(split_csv["path"], split_csv["source"]))).to_numpy()
    assert not pd.isna(sources).any(), f"some rows are not in {args.split}.csv"

    groups = {ALL: list(data)}
    for model in MODEL_ORDER:
        runs = sel.loc[sel["model"] == model, "run"].tolist()
        if len(runs) > 1:
            groups[f"{model} (its seeds)"] = runs
    variants = ["probs"] + (["probs_tta"] if all("probs_tta" in d for d in data.values()) else [])
    results = []
    for name, runs in groups.items():
        for v in variants:
            probs = np.mean([data[r][v] for r in runs], axis=0)
            results.append({"ensemble": name, "tta": v == "probs_tta", "runs": runs,
                            **score(y_true, probs, classes, sources, cfg["eval"])})

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"split": args.split, "n_images": len(y_true), "classes": classes,
                                    "results": results}, indent=2), encoding="utf-8")
    md = out_json.with_suffix(".md")
    md.write_text(to_markdown(args.split, results, len(y_true)), encoding="utf-8")
    fig = out_dir / "figures" / f"confusion_matrix_ensemble_{args.split}.png"
    fig.parent.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(np.array(results[0]["confusion_matrix"]), classes, fig,
                          title=f"Ensemble of {len(data)} runs, {args.split} set")

    for r in results:
        print(f"{r['ensemble']:28s} TTA {'yes' if r['tta'] else 'no ':3s} | accuracy {r['accuracy']:.4f} "
              f"| macro-F1 {r['macro_f1']:.4f} [{r['ci95']['macro_f1'][0]:.4f}, {r['ci95']['macro_f1'][1]:.4f}]")
    print(f"-> {out_json}, {md}, {fig}")


if __name__ == "__main__":
    main()
