"""Subject-independent k-fold cross-validation on CK+: the evidence for the CK+ accuracy.

Fold k: test = fold k, val = fold k+1 (checkpoint choice, as everywhere), train = the other folds
(src.data.ckplus_folds puts all frames of a person in one fold). After training, the run's best.pt
labels its test fold once (``test_probs.npz``). Pooled over the folds, every CK+ image is tested
exactly once by a model that never saw that person. The summary reports, per model, the pooled
accuracy and macro-F1 (mean ± std over seeds, bootstrap 95% CI), the accuracy per fold
(mean ± std) and the pooled confusion matrix. Finished runs are skipped, so an interrupted job
resumes where it stopped.

Examples:
    python -m src.ckplus_cv --models cnn_scratch --seeds 42 --folds 0 --dry-run
    python -m src.ckplus_cv                      # every model, seed and fold, then the summary
    python -m src.ckplus_cv --summary-only
"""
import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataset import read_split
from src.models import MODELS
from src.predict import load_run, predict_probs
from src.train import train
from src.utils.config import load_config
from src.utils.logger import plot_confusion_matrix
from src.utils.metrics import bootstrap_ci, classification_metrics, confusion_counts

CONFIG = "configs/ckplus/{}.yaml"


def fold_config(model: str, fold: int, ck: dict) -> dict:
    """The model's CK+ config with fold ``fold``'s split CSVs and the CV runs folder."""
    cfg = load_config(CONFIG.format(model))
    cfg["data"]["splits_dir"] = (Path(ck["root"]) / "splits" / f"fold{fold}").as_posix()
    cfg["runs_dir"] = ck["cv_runs_dir"]
    return cfg


def find_run(ck: dict, model: str, fold: int, seed: int, marker: str):
    """Latest run directory of (model, fold, seed) that contains ``marker``, or None."""
    hits = sorted(Path(ck["cv_runs_dir"]).glob(f"{model}_f{fold}_s{seed}_*/{marker}"))
    return hits[-1].parent if hits else None


def test_fold(run_dir: Path, device) -> float:
    """Label the run's test fold with best.pt; writes test_probs.npz and returns the accuracy."""
    model, cfg, classes = load_run(run_dir, device)
    df = read_split(cfg, "test")
    probs = predict_probs(model, cfg, df, device)
    y = df["label_idx"].to_numpy()
    np.savez_compressed(run_dir / "test_probs.npz", probs=probs.astype(np.float32), y_true=y,
                        paths=df["path"].to_numpy(dtype=str), classes=np.array(classes))
    return float((probs.argmax(axis=1) == y).mean())


def summarize(ck: dict, ev: dict, models, seeds, out_dir: Path, fig_dir: Path) -> None:
    """Pool every finished model/seed over the folds; write the markdown and CSV summaries."""
    k = int(ck["folds"])
    n_images = len(pd.read_csv(Path(ck["root"]) / "index.csv"))
    runs, table, per_class, notes = [], [], [], []
    for model in models:
        done, cm, classes, n_params = [], 0, None, None
        for seed in seeds:
            found = [find_run(ck, model, f, seed, "test_probs.npz") for f in range(k)]
            if None in found:
                notes.append(f"{model} seed {seed}: {found.count(None)} of {k} folds not finished, left out")
                continue
            y, pred, paths = [], [], []
            for fold, run in enumerate(found):
                z = np.load(run / "test_probs.npz")
                cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
                log = pd.read_csv(run / "log.csv")
                best = log.loc[log["val_macro_f1"].idxmax()]
                p = z["probs"].argmax(axis=1)
                classes, n_params = z["classes"].tolist(), cfg["run"]["n_params"]
                m = classification_metrics(z["y_true"], p, len(classes))
                runs.append({"model": model, "seed": seed, "fold": fold, "run": run.name,
                             "n_test": len(p), "accuracy": m["acc"], "macro_f1": m["macro_f1"],
                             "best_epoch": int(best["epoch"]), "best_val_macro_f1": best["val_macro_f1"],
                             "epochs": len(log)})
                y.append(z["y_true"])
                pred.append(p)
                paths += z["paths"].tolist()
            assert len(paths) == len(set(paths)) == n_images, "the test folds must cover CK+ exactly once"
            y, pred = np.concatenate(y), np.concatenate(pred)
            m = classification_metrics(y, pred, len(classes))
            ci = bootstrap_ci(y, pred, len(classes), int(ev["bootstrap_samples"]), int(ev["bootstrap_seed"]))
            c = confusion_counts(y, pred, len(classes))
            tp = np.diag(c)
            f1 = 2 * tp / np.maximum(c.sum(axis=0) + c.sum(axis=1), 1)
            per_class.append({"model": model, "seed": seed, **dict(zip(classes, f1))})
            done.append({"seed": seed, **m, "ci": ci})
            cm = cm + c
        if not done:
            continue
        fold_acc = np.array([r["accuracy"] for r in runs if r["model"] == model])
        acc = np.array([d["acc"] for d in done])
        f1 = np.array([d["macro_f1"] for d in done])
        table.append({
            "model": model, "seeds": " ".join(str(d["seed"]) for d in done), "params": n_params,
            "acc_mean": acc.mean(), "acc_std": acc.std(ddof=1) if len(acc) > 1 else 0.0,
            "f1_mean": f1.mean(), "f1_std": f1.std(ddof=1) if len(f1) > 1 else 0.0,
            "acc_ci_low": np.mean([d["ci"]["accuracy"][0] for d in done]),
            "acc_ci_high": np.mean([d["ci"]["accuracy"][1] for d in done]),
            "f1_ci_low": np.mean([d["ci"]["macro_f1"][0] for d in done]),
            "f1_ci_high": np.mean([d["ci"]["macro_f1"][1] for d in done]),
            "fold_acc_mean": fold_acc.mean(), "fold_acc_std": fold_acc.std(ddof=1),
            "fold_acc_min": fold_acc.min(), "fold_acc_max": fold_acc.max(),
        })
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(cm, classes, fig_dir / f"ckplus_cv_confusion_{model}.png",
                              title=f"{model}, CK+ {k}-fold CV ({len(done)} seed(s))")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(runs).to_csv(out_dir / "ckplus_cv_runs.csv", index=False)
    pd.DataFrame(table).to_csv(out_dir / "ckplus_cv_summary.csv", index=False)
    lines = [
        f"# CK+ subject-independent {k}-fold cross-validation",
        "",
        f"{n_images} CK+ images, 6 classes. Fold k: test = fold k, val = fold k+1 (best.pt), train = the "
        "other folds; all frames of one person are in one fold. Pooled = the test folds together, so every "
        "image is tested once by a model that never saw its person.",
        "",
        "| Model | Params | Pooled accuracy (%) | Pooled macro-F1 | Macro-F1 95% CI* | Fold accuracy (%) | Worst / best fold (%) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in table:
        lines.append(
            f"| {r['model']} | {r['params'] / 1e6:.2f} M | {100 * r['acc_mean']:.2f} ± {100 * r['acc_std']:.2f} "
            f"| {r['f1_mean']:.4f} ± {r['f1_std']:.4f} | [{r['f1_ci_low']:.4f}, {r['f1_ci_high']:.4f}] "
            f"| {100 * r['fold_acc_mean']:.2f} ± {100 * r['fold_acc_std']:.2f} "
            f"| {100 * r['fold_acc_min']:.1f} / {100 * r['fold_acc_max']:.1f} |")
    lines += ["", "Mean ± std over seeds (pooled columns) or over all folds of all seeds (fold columns). "
              "*Bootstrap 95% CI of each seed's pooled macro-F1 (images resampled), averaged over seeds.", ""]
    if per_class:
        pc = pd.DataFrame(per_class).drop(columns="seed").groupby("model", sort=False).mean()
        lines += ["## Per-class F1 (pooled, mean over seeds)", "",
                  "| Model | " + " | ".join(pc.columns) + " |", "|---" * (len(pc.columns) + 1) + "|"]
        lines += [f"| {m} | " + " | ".join(f"{v:.3f}" for v in row) + " |" for m, row in pc.iterrows()]
        lines.append("")
    lines += [f"Confusion matrices: `{fig_dir.as_posix()}/ckplus_cv_confusion_<model>.png` (all seeds pooled). "
              "Per-run results: `ckplus_cv_runs.csv`."]
    if notes:
        lines += ["", "## Not included", ""] + [f"- {n}" for n in notes]
    (out_dir / "ckplus_cv.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Subject-independent k-fold cross-validation on CK+.")
    ap.add_argument("--base-config", default="configs/base.yaml", help="file with the ckplus: and eval: blocks")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, help="default: seeds of the base config")
    ap.add_argument("--folds", nargs="+", type=int, help="default: all folds")
    ap.add_argument("--dry-run", action="store_true", help="2 batches per stage, no summary")
    ap.add_argument("--summary-only", action="store_true", help="only pool the finished runs")
    ap.add_argument("--out", default="reports", help="folder for ckplus_cv.md and the CSVs")
    ap.add_argument("--cpu", action="store_true", help="use the CPU even if CUDA is available")
    args = ap.parse_args()

    base = load_config(args.base_config)
    ck, ev = base["ckplus"], base["eval"]
    seeds = args.seeds or base["seeds"]
    folds = args.folds if args.folds is not None else list(range(int(ck["folds"])))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    if not args.summary_only:
        total = len(args.models) * len(seeds) * len(folds)
        for i, (model, seed, fold) in enumerate(
                ((m, s, f) for m in args.models for s in seeds for f in folds), start=1):
            head = f"[{i}/{total}] {model} seed {seed} fold {fold}"
            if not args.dry_run and find_run(ck, model, fold, seed, "test_probs.npz"):
                print(f"{head}: done, skipped")
                continue
            tic = time.time()
            # a run whose training finished but whose test fold was not labelled yet (and that still
            # has its best.pt: the Colab notebook does not copy fold checkpoints to Drive)
            run_dir = None if args.dry_run else find_run(ck, model, fold, seed, "curves.png")
            if run_dir is None or not (run_dir / "best.pt").exists():
                run_dir = train(fold_config(model, fold, ck), seed, dry_run=args.dry_run,
                                tag=f"f{fold}", cpu=args.cpu)
            acc = test_fold(run_dir, device)
            print(f"{head}: test fold accuracy {acc:.4f} ({time.time() - tic:.0f}s) -> {run_dir}", flush=True)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if args.dry_run:
            return
    summarize(ck, ev, args.models, seeds, Path(args.out), Path(args.out) / "figures")


if __name__ == "__main__":
    main()
