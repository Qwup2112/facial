"""Collect every final run's test_metrics.json into reports/results_summary.csv and a markdown table.

Final runs: per model, the finished runs trained with the same config as the model's most recent
run (the latest run per seed), so a superseded config, such as a first try before tuning on val,
is never mixed into the mean ± std. A run that trained every planned epoch counts as a run with
early stopping off, whatever its patience (see config_signature). Dropped runs are listed in the
markdown file.
``--list-final`` prints the final run directories, i.e. the runs to pass to src.evaluate, so
the test set is touched only for final models.

Examples:
    python -m src.aggregate --list-final
    python -m src.aggregate --runs runs --out reports/results_summary.csv
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml

# Same order as src.models.MODELS (not imported here: it would load torch).
MODEL_ORDER = ("cnn_scratch", "resnet18", "mobilenet_v2")
STAMP = re.compile(r"_(\d{8}-\d{4})(?:-(\d+))?$")   # run dir suffix written by train.py
METRIC_COLS = ["accuracy", "macro_f1", "weighted_f1", "acc_ci_low", "acc_ci_high",
               "f1_ci_low", "f1_ci_high", "ms_cpu", "ms_gpu"]


def config_signature(cfg: dict, full_schedule: bool = False) -> str:
    """The training-relevant part of a run config (drops `seed`, `run:`, `eval:` and paths).

    A run that trained every planned epoch never triggered early stopping, so its patience did not
    change the procedure: it matches a run with early stopping off, and the patience is left out.
    """
    sig = {k: cfg.get(k) for k in ("model", "data", "augment", "train")}
    if full_schedule:
        sig["train"] = {k: v for k, v in sig["train"].items() if k != "early_stop_patience"}
    return json.dumps(sig, sort_keys=True)


def finished_runs(runs_dir: Path) -> pd.DataFrame:
    """Finished, non-dry runs (train.py writes curves.png at the end) with their config signature."""
    rows = []
    for cfg_path in sorted(runs_dir.glob("*/config.yaml")):
        run = cfg_path.parent
        stamp = STAMP.search(run.name)
        if run.name.startswith("dryrun_") or not stamp or not (run / "curves.png").exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        planned = sum(int(s["epochs"]) for s in cfg["train"]["stages"])
        trained = len(pd.read_csv(run / "log.csv"))
        rows.append({"model": cfg["model"]["name"], "seed": int(cfg["seed"]), "run": run.name, "dir": run,
                     "signature": config_signature(cfg, trained == planned), "full": trained == planned,
                     "epochs": f"{trained}/{planned}", "order": (stamp.group(1), int(stamp.group(2) or 1))})
    return pd.DataFrame(rows)


def select(df: pd.DataFrame):
    """Per model, keep the runs with its latest config (latest run per seed); return (kept, dropped)."""
    kept, dropped = [], []
    for _, g in df.groupby("model"):
        g = g.sort_values("order")
        final = g["signature"].iloc[-1]
        latest = g[g["signature"] == final].groupby("seed").tail(1)
        rest = g.drop(latest.index)
        kept.append(latest)
        dropped.append(rest.assign(reason=["older run, same seed" if s == final else
                                           f"stopped early ({e} epochs)" if not f else "older config"
                                           for s, f, e in zip(rest["signature"], rest["full"], rest["epochs"])]))
    return pd.concat(kept), pd.concat(dropped)


def add_metrics(sel: pd.DataFrame) -> pd.DataFrame:
    """Join each run's test_metrics.json (runs without one are skipped by the caller)."""
    rows = []
    for r in sel.itertuples():
        m = json.loads((r.dir / "test_metrics.json").read_text(encoding="utf-8"))
        ms = m.get("ms_per_image", {})
        rows.append({
            "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
            "acc_ci_low": m["ci95"]["accuracy"][0], "acc_ci_high": m["ci95"]["accuracy"][1],
            "f1_ci_low": m["ci95"]["macro_f1"][0], "f1_ci_high": m["ci95"]["macro_f1"][1],
            "params": m["params"], "checkpoint_mb": m["checkpoint_mb"],
            "ms_cpu": ms.get("cpu"), "ms_gpu": ms.get("gpu"), "best_epoch": m["best_epoch"],
        })
    return pd.concat([sel.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def summarize(sel: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample std over seeds per model (std is NaN for a single seed)."""
    rows = []
    for model in [m for m in MODEL_ORDER if m in set(sel["model"])]:
        g = sel[sel["model"] == model].sort_values("seed")
        row = {"model": model, "n_seeds": len(g), "seeds": " ".join(map(str, g["seed"])),
               "params": int(g["params"].iloc[0]), "checkpoint_mb": g["checkpoint_mb"].iloc[0]}
        for col in METRIC_COLS:
            row[f"{col}_mean"] = g[col].mean()
            if col in ("accuracy", "macro_f1", "weighted_f1"):
                row[f"{col}_std"] = g[col].std(ddof=1)
        row["runs"] = " ".join(g["run"])
        rows.append(row)
    return pd.DataFrame(rows)


def pm(mean: float, std: float, scale: float = 1.0, digits: int = 2) -> str:
    """'mean ± std' (std omitted for a single seed)."""
    s = f"{mean * scale:.{digits}f}"
    return s if pd.isna(std) else f"{s} ± {std * scale:.{digits}f}"


def to_markdown(summary: pd.DataFrame, sel: pd.DataFrame, dropped: pd.DataFrame) -> str:
    """Comparison table, per-run table and the list of runs left out."""
    lines = ["# Test results (mean ± std over seeds)", "",
             "| Model | Seeds | Params | Accuracy (%) | Macro-F1 | Macro-F1 95% CI* | ms/image (CPU) | ms/image (GPU) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in summary.itertuples():
        cpu = "-" if pd.isna(r.ms_cpu_mean) else f"{r.ms_cpu_mean:.2f}"
        gpu = "-" if pd.isna(r.ms_gpu_mean) else f"{r.ms_gpu_mean:.2f}"
        lines.append(f"| {r.model} | {r.seeds} | {r.params / 1e6:.2f} M | {pm(r.accuracy_mean, r.accuracy_std, 100)} "
                     f"| {pm(r.macro_f1_mean, r.macro_f1_std, 1, 4)} "
                     f"| [{r.f1_ci_low_mean:.4f}, {r.f1_ci_high_mean:.4f}] | {cpu} | {gpu} |")
    lines += ["", "*Bootstrap 95% CI of each seed's test macro-F1 (1,000 resamples), averaged over seeds.", "",
              "## Runs included", "",
              "| Model | Seed | Run | Accuracy | Macro-F1 | Acc. 95% CI | Macro-F1 95% CI | Best epoch | Epochs |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in sel.sort_values(["model", "seed"]).itertuples():
        lines.append(f"| {r.model} | {r.seed} | {r.run} | {r.accuracy:.4f} | {r.macro_f1:.4f} "
                     f"| [{r.acc_ci_low:.4f}, {r.acc_ci_high:.4f}] | [{r.f1_ci_low:.4f}, {r.f1_ci_high:.4f}] "
                     f"| {r.best_epoch} | {r.epochs} |")
    if len(dropped):
        lines += ["", "## Runs left out", "", "| Model | Seed | Run | Reason |", "|---|---|---|---|"]
        for r in dropped.sort_values(["model", "seed"]).itertuples():
            lines.append(f"| {r.model} | {r.seed} | {r.run} | {r.reason} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Aggregate test_metrics.json files into a comparison table.")
    ap.add_argument("--runs", default="runs", help="directory with the run folders")
    ap.add_argument("--out", default="reports/results_summary.csv", help="summary CSV (markdown next to it)")
    ap.add_argument("--list-final", action="store_true", help="only print the final run directories")
    args = ap.parse_args()

    runs = finished_runs(Path(args.runs))
    assert len(runs), f"no finished run under {args.runs}"
    sel, dropped = select(runs)
    if args.list_final:
        for r in sel.sort_values(["model", "seed"]).itertuples():
            print(r.dir.as_posix())
        return

    evaluated = sel["dir"].map(lambda d: (d / "test_metrics.json").exists())
    for r in sel[~evaluated].itertuples():
        print(f"not evaluated yet (left out): python -m src.evaluate --run {r.dir.as_posix()}")
    dropped = pd.concat([dropped, sel[~evaluated].assign(reason="no test_metrics.json yet")])
    sel = sel[evaluated]
    assert len(sel), "no final run has test_metrics.json yet"
    sel = add_metrics(sel)
    summary = summarize(sel)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    sel.drop(columns=["dir", "signature", "full", "order"]).to_csv(out.with_name("results_runs.csv"), index=False)
    md = out.with_suffix(".md")
    md.write_text(to_markdown(summary, sel, dropped), encoding="utf-8")

    print(summary[["model", "n_seeds", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std",
                   "ms_cpu_mean"]].round(4).to_string(index=False))
    for r in summary.itertuples():
        if r.n_seeds < 3:
            print(f"warning: {r.model} has {r.n_seeds} seed(s) ({r.seeds}); CLAUDE.md asks for 3")
    for r in dropped.itertuples():
        print(f"left out: {r.run} ({r.reason})")
    print(f"-> {out}, {out.with_name('results_runs.csv')}, {md}")


if __name__ == "__main__":
    main()
