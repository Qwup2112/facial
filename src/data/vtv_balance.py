"""VTV step 2b: offline-augmented TRAIN copies so that every class has at least ``min_per_class`` faces.

The user asked for >= 1000 faces per emotion (2026-11-07); the real VTV faces of angry, disgust, fear and surprise
are fewer. For such a class, copies of its train faces are added as ``dataset/train/<class>/<id>_aug<k>.png`` with
the offline augmentation of src.data.add_external (flip, rotation, scale, shift, brightness, contrast; ranges
measured on the main dataset's augmented copies). The sources are the class's train faces in random order, each
used at most ceil(need / n) times, so every person keeps its share. val/test stay 100% real.

In manifest.csv the copies have ``source = augmented`` (the project convention: src.data.index_splits passes it
on and ``data.use_offline_aug: false`` drops them in src.data.dataset.read_split), ``split = train`` and
``origin`` = the real file. Called at the end of src.data.vtv_build when ``build.balance.min_per_class`` is set;
alone it replaces the copies of an existing build and rewrites the zip:
    python -m src.data.vtv_balance
"""
import argparse
import math
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.add_external import augment
from src.data.vtv_faces import imread, imwrite
from src.utils.config import load_config


def balance(ds: Path, min_per_class: int, seed: int) -> str:
    """Add augmented train copies in ``ds`` (a vtv_build dataset folder); returns the report section (Markdown)."""
    man = pd.read_csv(ds / "manifest.csv", dtype=str, keep_default_na=False)
    if "origin" not in man:
        man["origin"] = ""
    old = man[man["source"] == "augmented"]
    for f in old["file"]:
        (ds / "train" / f).unlink(missing_ok=True)
    man = man[man["source"] != "augmented"]
    rng = np.random.default_rng(seed)
    rows, table = [], []
    for c, d in man.groupby("label"):
        total, tr = len(d), d[d["split"] == "train"]
        need = max(0, min_per_class - total)
        if need and len(tr):
            src = tr.iloc[rng.permutation(len(tr))]
            for k in range(math.ceil(need / len(tr))):
                for r in src.iloc[:need - k * len(tr)].itertuples():
                    gray = imread(ds / "train" / r.file, cv2.IMREAD_GRAYSCALE)
                    aid = f"{r.id}_aug{k}"
                    imwrite(ds / "train" / c / f"{aid}.png", augment(gray, rng))
                    rows.append({**r._asdict(), "id": aid, "file": f"{c}/{aid}.png", "source": "augmented",
                                 "origin": f"train/{r.file}"})
        added = sum(1 for x in rows if x["label"] == c)
        table.append((c, total, int((d["split"] == "train").sum()), added, total + added))
    aug = pd.DataFrame(rows).drop(columns="Index", errors="ignore")
    out = pd.concat([man, aug], ignore_index=True) if len(aug) else man
    out.to_csv(ds / "manifest.csv", index=False)
    lines = [f"## Class balance (src.data.vtv_balance, min_per_class = {min_per_class})", "",
             "Offline-augmented copies of train faces (flip, rotation, scale, shift, brightness, contrast), only in "
             "train, `source = augmented` in manifest.csv (`data.use_offline_aug: false` drops them). val/test: real "
             "faces only.", "", "| Class | real faces | real train | + augmented (train) | total |", "|---|---|---|---|---|"]
    lines += [f"| {c} | {t} | {tr} | {a} | {s} |" for c, t, tr, a, s in table]
    return "\n".join(lines) + "\n"


def write_zip(ds: Path, zpath: Path) -> None:
    """The dataset under dataset/ for Colab (same rule as src.data.vtv_build)."""
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(ds.rglob("*")):
            if f.is_file() and f.suffix in (".png", ".csv") and f.name not in ("person_check.png", "leak_check.png"):
                z.write(f, "dataset/" + f.relative_to(ds).as_posix())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/data/vtv_faces.yaml")
    ap.add_argument("--min-per-class", type=int, help="override build.balance.min_per_class")
    args = ap.parse_args()
    b = load_config(args.config)["build"]
    n = args.min_per_class or (b.get("balance") or {}).get("min_per_class")
    assert n, "no min_per_class (build.balance.min_per_class or --min-per-class)"
    ds = Path(b["dataset_dir"])
    text = balance(ds, int(n), int(b["seed"]))
    rep = ds / "report.md"
    body = rep.read_text(encoding="utf-8").split("\n## Class balance")[0].rstrip("\n")
    rep.write_text(body + "\n\n" + text, encoding="utf-8")
    write_zip(ds, Path(b["zip"]))
    print(text)
    print(f"zip: {Path(b['zip']).resolve()} ({Path(b['zip']).stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
