"""Head-to-head of two transfer run sets (two label versions of the VTV dataset) on the test faces they share.

Each label version is a separate build, split by person on its own, so the two test sets differ and their
agreement numbers cannot be compared directly; the old models cannot be scored on the new test set either
(most new test people were in the old train set). A face that is in both test sets belongs to people that
neither model set trained on, so both are scored there, against the newer labels (run set B).

Reads each run's test_probs.npz (probs, y_true, paths, classes, written by src.evaluate). With --moves
(data/vtv/claude_suggest_moves.csv) it also splits run set A's own test agreement into faces Claude's pre-screen
kept and faces it later moved out (_reject/_unsure). Labels are suggestions (not human-checked): every number
here is agreement with them, not accuracy.

  python transfer_compare.py --a runs/transfer_vtv_suggested --b runs/transfer_vtv_claude \
      --moves claude_suggest_moves.csv --out reports/transfer_compare_vtv_suggested_vs_vtv_claude.md
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

RUN = re.compile(r"^(?P<model>.+)_s(?P<seed>\d+)_\d{8}-\d{4}$")
MODELS = ["cnn_scratch", "resnet18", "mobilenet_v2"]


def load(runs_dir: Path) -> dict:
    """(model, seed) -> DataFrame(id, label, pred) of the newest evaluated run."""
    out = {}
    for npz in sorted(runs_dir.glob("*/test_probs.npz")):
        m = RUN.match(npz.parent.name)
        if not m:
            continue  # dryrun_* and other non-final dirs
        z = np.load(npz, allow_pickle=False)
        cls = [str(c) for c in z["classes"]]
        out[(m["model"], int(m["seed"]))] = pd.DataFrame({
            "id": [Path(str(p)).stem for p in z["paths"]],
            "label": [cls[i] for i in z["y_true"]],
            "pred": [cls[i] for i in z["probs"].argmax(1)],
        }).set_index("id")
    assert out, f"no evaluated runs in {runs_dir}"
    return out


def macro_f1(y, p, classes) -> float:
    f = []
    for c in classes:
        tp = ((y == c) & (p == c)).sum()
        prec = tp / max((p == c).sum(), 1)
        rec = tp / max((y == c).sum(), 1)
        f.append(0.0 if tp == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f))


def ms(v) -> str:
    v = np.asarray(v, float)
    return f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else f"{v.mean():.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", default="runs/transfer_vtv_suggested")
    ap.add_argument("--b", default="runs/transfer_vtv_claude")
    ap.add_argument("--moves", default=str(Path(__file__).resolve().parent / "claude_suggest_moves.csv"))
    ap.add_argument("--out", default="reports/transfer_compare_vtv_suggested_vs_vtv_claude.md")
    args = ap.parse_args()
    na, nb = Path(args.a).name.removeprefix("transfer_"), Path(args.b).name.removeprefix("transfer_")
    A, B = load(Path(args.a)), load(Path(args.b))
    ref_b = next(iter(B.values()))
    classes = sorted(ref_b["label"].unique())
    shared = set(next(iter(A.values())).index) & set(ref_b.index)
    for d in list(A.values()) + list(B.values()):
        shared &= set(d.index)
    shared = sorted(shared)
    y = ref_b.loc[shared, "label"]
    y_a = next(iter(A.values())).loc[shared, "label"]
    L = [f"# Transfer: `{na}` vs `{nb}` on the test faces both test sets share", "",
         f"Labels are model suggestions (not human-checked): every number is **agreement** with the `{nb}` labels, "
         "not accuracy. Each label version was split by person separately, so the test sets differ; a face in "
         "both test sets belongs to people neither model set trained on.", "",
         f"Runs: `{na}` {len(A)}, `{nb}` {len(B)}. Shared test faces: **{len(shared)}** "
         f"(`{na}` test {len(next(iter(A.values())))}, `{nb}` test {len(ref_b)}); "
         f"label differs between the versions on {(y != y_a).sum()} of them.", "",
         "| Class | " + " | ".join(classes) + " |", "|---" * (len(classes) + 1) + "|",
         "| shared faces | " + " | ".join(str((y == c).sum()) for c in classes) + " |", "",
         "## Agreement on the shared faces (mean ± std over seeds, %)", "",
         f"| Model | `{na}` agreement | `{nb}` agreement | `{na}` macro-F1 | `{nb}` macro-F1 |", "|---|---|---|---|---|"]
    rec = {}
    for m in MODELS:
        row = []
        for name, R in ((na, A), (nb, B)):
            accs, f1s = [], []
            for (mm, s), d in R.items():
                if mm != m:
                    continue
                p = d.loc[shared, "pred"]
                accs.append(100 * (p == y).mean())
                f1s.append(macro_f1(y, p, classes))
                for c in classes:
                    rec.setdefault((m, name, c), []).append(100 * (p[y == c] == c).mean() if (y == c).any() else np.nan)
            row.append((accs, f1s))
        if row[0][0] and row[1][0]:
            L.append(f"| {m} | {ms(row[0][0])} | {ms(row[1][0])} | {ms(row[0][1])} | {ms(row[1][1])} |")
    L += ["", "## Per-class agreement (recall) on the shared faces (mean over seeds, %)", "",
          "| Class | n | " + " | ".join(f"{m} `{n}`" for m in MODELS for n in (na, nb)) + " |",
          "|---|---" + "|---" * (2 * len(MODELS)) + "|"]
    for c in classes:
        cells = [f"{np.nanmean(rec[(m, n, c)]):.0f}" if (m, n, c) in rec else "–" for m in MODELS for n in (na, nb)]
        L.append(f"| {c} | {(y == c).sum()} | " + " | ".join(cells) + " |")
    if args.moves and Path(args.moves).exists():
        mv = pd.read_csv(args.moves).drop_duplicates("id", keep="last").set_index("id")["to"]
        L += ["", f"## `{na}` test set: faces Claude's pre-screen later moved out", "",
              f"Own test set of `{na}`, all its runs pooled. \"Moved\" = moved to `_reject` (blurred, dark, occluded, "
              "turned) or `_unsure` (Claude disagreed with the suggested emotion) on 2026-11-08.", "",
              "| Suggested label | test faces | moved out | agreement on kept (%) | agreement on moved (%) |",
              "|---|---|---|---|---|"]
        for c in classes:
            ka, ma, n_c, n_m = [], [], 0, 0
            for d in A.values():
                dc = d[d["label"] == c]
                moved = dc.index.isin(mv.index)
                n_c, n_m = len(dc), int(moved.sum())
                ok = dc["pred"] == c
                ka.append(100 * ok[~moved].mean() if (~moved).any() else np.nan)
                ma.append(100 * ok[moved].mean() if moved.any() else np.nan)
            L.append(f"| {c} | {n_c} | {n_m} | {np.nanmean(ka):.0f} | "
                     + (f"{np.nanmean(ma):.0f}" if n_m else "–") + " |")
    text = "\n".join(L) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
