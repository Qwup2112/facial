"""Index the existing train/val/test split into CSV files.

Walks ``{root}/{train,val,test}/<class>/`` and writes ``{out}/{split}.csv`` with
columns ``path,label,label_idx,source``. ``source`` comes from ``manifest.csv``
(local / rafdb / affectnet / augmented) when available, else ``unknown``.
The split itself is never changed. ``root`` and ``out`` default to
``data.root`` and ``data.splits_dir`` of the config.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.utils.config import load_config

SPLITS = ("train", "val", "test")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def read_class_names(root: Path, num_classes: int) -> list[str]:
    """Return sorted class folder names of the train split and check all splits agree."""
    classes = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    assert len(classes) == num_classes, f"Expected {num_classes} classes, got {classes}"
    for split in SPLITS[1:]:
        other = sorted(p.name for p in (root / split).iterdir() if p.is_dir())
        assert other == classes, f"Class folders differ in '{split}': {other}"
    return classes


def index_split(root: Path, split: str, classes: list[str], sources: dict) -> pd.DataFrame:
    """List every image of one split as a DataFrame."""
    rows = []
    for idx, name in enumerate(classes):
        for f in sorted((root / split / name).iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                rel = f"{name}/{f.name}"
                rows.append({
                    "path": f.as_posix(),
                    "label": name,
                    "label_idx": idx,
                    "source": sources.get((split, rel), "unknown"),
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--root", help="folder containing train/val/test (default: data.root)")
    ap.add_argument("--out", help="output folder for the CSVs (default: data.splits_dir)")
    args = ap.parse_args()

    data_cfg = load_config(args.config)["data"]
    root = Path(args.root or data_cfg["root"])
    out = Path(args.out or data_cfg["splits_dir"])
    out.mkdir(parents=True, exist_ok=True)
    classes = read_class_names(root, data_cfg["num_classes"])

    sources = {}
    manifest = root / "manifest.csv"
    if manifest.exists():
        m = pd.read_csv(manifest, usecols=["file", "source", "split"])
        sources = {(s, f): src for f, src, s in m.itertuples(index=False)}

    for split in SPLITS:
        df = index_split(root, split, classes, sources)
        df.to_csv(out / f"{split}.csv", index=False)
        counts = df.groupby("label").size().to_dict()
        print(f"{split}: {len(df)} images, per class {counts}")
    print("classes:", classes)


if __name__ == "__main__":
    main()
