"""Remove likely copies of photos from val/test, then re-balance val and test (train is not touched).

Reads reports/near_duplicates.csv written by src.data.inspect (column `likely_copy`). For each
likely-copy pair the held-out member leaves its split: the val or test image of a val-train or
test-train pair, and the val image of a test-val pair, so the test set stays as large as possible.
Each class of val and test is then cut to the smallest class count of that split (seeded random
choice), so every split stays exactly balanced.

Files are MOVED, never deleted, to {out}/<split>/<class>/; {out}/removed.csv lists every moved
file with the reason, and manifest.csv is backed up to {out} before its rows are dropped.
Afterwards run src.data.index_splits and src.data.inspect again.

Examples:
    python -m src.data.dedup_split --dry-run
    python -m src.data.dedup_split
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.index_splits import IMG_EXTS, read_class_names
from src.utils.config import load_config

HELD_OUT = ("val", "test")


def copies_to_remove(near: pd.DataFrame) -> pd.DataFrame:
    """Held-out member of every likely-copy pair (one row per image, closest pair kept as the reason)."""
    rows = []
    for r in near[near["likely_copy"]].sort_values(["distance", "ncc"], ascending=[True, False]).itertuples():
        if (r.split_a, r.split_b) == ("test", "val"):           # keep the test image, drop the val one
            rows.append((r.split_b, r.file_b, r.label_b, f"copy of test/{r.file_a}", r.distance, r.ncc))
        else:
            rows.append((r.split_a, r.file_a, r.label_a, f"copy of {r.split_b}/{r.file_b}", r.distance, r.ncc))
    out = pd.DataFrame(rows, columns=["split", "file", "label", "reason", "distance", "ncc"])
    return out.drop_duplicates(["split", "file"])


def balance_cuts(root: Path, classes: list[str], removed: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Extra images to set aside so every class of a held-out split has the same count."""
    rng = np.random.default_rng(seed)
    gone = set(zip(removed["split"], removed["file"]))
    rows = []
    for split in HELD_OUT:
        left = {c: sorted(f"{c}/{p.name}" for p in (root / split / c).iterdir()
                          if p.suffix.lower() in IMG_EXTS and (split, f"{c}/{p.name}") not in gone) for c in classes}
        n_min = min(len(v) for v in left.values())
        for c, files in left.items():
            for f in rng.choice(files, size=len(files) - n_min, replace=False):
                rows.append((split, f, c, f"balance: {split} keeps {n_min} per class", None, None))
    return pd.DataFrame(rows, columns=removed.columns)


def main():
    ap = argparse.ArgumentParser(description="Move likely copies out of val/test and re-balance them.")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--raw", help="dataset folder (default: data.root)")
    ap.add_argument("--near", default="reports/near_duplicates.csv", help="output of src.data.inspect")
    ap.add_argument("--out", default="dataset_removed", help="where moved files go (outside the dataset)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the random balancing cut")
    ap.add_argument("--dry-run", action="store_true", help="only print what would be moved")
    args = ap.parse_args()

    data_cfg = load_config(args.config)["data"]
    root, out = Path(args.raw or data_cfg["root"]), Path(args.out)
    classes = read_class_names(root, data_cfg["num_classes"])
    near = pd.read_csv(args.near)
    assert "likely_copy" in near, f"{args.near} has no likely_copy column; re-run src.data.inspect"

    copies = copies_to_remove(near)
    plan = pd.concat([copies, balance_cuts(root, classes, copies, args.seed)], ignore_index=True)
    plan["kind"] = np.where(plan["reason"].str.startswith("copy"), "copy", "balance")
    summary = plan.groupby(["label", "split", "kind"]).size().unstack(["split", "kind"], fill_value=0)
    summary.columns = [f"{s} {k}" for s, k in summary.columns]
    print("images to move out, per class:")
    print(summary.to_string())
    for split in HELD_OUT:
        n_now = sum(1 for c in classes for p in (root / split / c).iterdir() if p.suffix.lower() in IMG_EXTS)
        n_out = int((plan["split"] == split).sum())
        print(f"{split}: {n_now:,} -> {n_now - n_out:,} images ({(n_now - n_out) // len(classes):,} per class)")
    if args.dry_run:
        print("dry run: nothing moved")
        return

    out.mkdir(parents=True, exist_ok=True)
    backup = out / "manifest_before_dedup.csv"
    if not backup.exists():                                    # keep the first, untouched manifest
        shutil.copy2(root / "manifest.csv", backup)
    for r in plan.itertuples():
        dst = out / r.split / r.file
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(root / r.split / r.file, dst)
    log = out / "removed.csv"
    plan.to_csv(log, mode="a", header=not log.exists(), index=False)

    m = pd.read_csv(root / "manifest.csv", dtype=str)
    moved = set(zip(plan["split"], plan["file"]))
    keep = [(s, f) not in moved for s, f in zip(m["split"], m["file"])]
    m[keep].to_csv(root / "manifest.csv", index=False)
    print(f"moved {len(plan):,} files to {out}/ (list: {log}); manifest.csv: {len(m):,} -> {sum(keep):,} rows")
    print("next: python -m src.data.index_splits && python -m src.data.inspect --raw", root.as_posix())


if __name__ == "__main__":
    main()
