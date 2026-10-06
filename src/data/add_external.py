"""Merge an external, subject-labelled face set into dataset/ (CK+ for now), split by person.

Every person goes to one split only (seeded), with about the same train/val/test share as the
current originals, so the near-identical frames of one person never cross splits. Each split keeps
exactly the same number of originals per class: every new image takes the place of a random
`local` image of the same class and split, which is MOVED (with its augmented copy in train) to
{out}/<split>/<class>/ and logged in {out}/removed.csv, never deleted. New train images get one
offline-augmented copy each (flip, rotation, scale, shift, brightness, contrast), with ranges
measured on the existing copies. manifest.csv is backed up to {out} before it is changed.
Afterwards run src.data.index_splits and src.data.inspect.

CK+ source: the 48x48 grayscale version (last 3 frames of the 327 labelled sequences), file names
S<subject>_<sequence>_<frame>; `contempt` is not one of the 7 classes and is dropped.

Examples:
    python -m src.data.add_external --source ckplus --dry-run
    python -m src.data.add_external --source ckplus
"""
import argparse
import hashlib
import io
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.data.index_splits import read_class_names
from src.data.inspect import dhash
from src.utils.config import load_config

SPLITS = ("train", "val", "test")
SOURCES = {
    "ckplus": {
        "file": "data/external/ckplus/train-00000-of-00001.parquet",
        "labels": {"anger": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy",
                   "sadness": "sad", "surprise": "surprise"},          # contempt: not a class here
        "subject": r"^(S\d{3})_",
    },
}
# offline augmentation, ranges measured on 400 existing (original, copy) pairs
AUG = {"hflip": 0.5, "angle": 10.0, "scale": (0.92, 1.08), "shift": 0.11, "brightness": 33.0, "contrast": 0.2}


def load_ckplus(cfg: dict, size: int) -> pd.DataFrame:
    """Images (uint8 arrays), class labels and subject ids of the CK+ parquet."""
    d = pd.read_parquet(cfg["file"])
    d = d[d["label"].isin(cfg["labels"])].reset_index(drop=True)
    rows = []
    for i, r in d.iterrows():
        gray = np.asarray(Image.open(io.BytesIO(r["image"]["bytes"])).convert("L"))
        if gray.shape != (size, size):
            gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        rows.append({"img": gray, "label": cfg["labels"][r["label"]], "name": r["file"],
                     "origin": f"{Path(cfg['file']).name}#{i} {r['file']}"})
    out = pd.DataFrame(rows)
    out["subject"] = out["name"].str.extract(cfg["subject"])[0]
    assert out["subject"].notna().all(), "file names without a subject id"
    return out


def split_subjects(new: pd.DataFrame, shares: dict, seed: int) -> dict:
    """Subject -> split: subjects in random order, each to the split furthest below its share."""
    rng = np.random.default_rng(seed)
    counts = new.groupby("subject").size()
    subjects = rng.permutation(counts.index.to_numpy())
    total, got, out = len(new), dict.fromkeys(SPLITS, 0), {}
    for s in subjects:
        split = min(SPLITS, key=lambda k: got[k] / total - shares[k])
        out[s], got[split] = split, got[split] + counts[s]
    return out


def augment(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One offline-augmented copy of a 48x48 grayscale face."""
    img = gray[:, ::-1] if rng.random() < AUG["hflip"] else gray
    n = img.shape[0]
    m = cv2.getRotationMatrix2D((n / 2, n / 2), rng.uniform(-AUG["angle"], AUG["angle"]), rng.uniform(*AUG["scale"]))
    m[:, 2] += rng.uniform(-AUG["shift"], AUG["shift"], 2) * n
    img = cv2.warpAffine(np.ascontiguousarray(img), m, (n, n), borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)
    mean = img.mean()
    img = (img - mean) * rng.uniform(1 - AUG["contrast"], 1 + AUG["contrast"]) + mean
    return np.clip(img + rng.uniform(-AUG["brightness"], AUG["brightness"]), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Merge an external face set into dataset/, split by person.")
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="dataset_removed", help="where replaced files go (outside the dataset)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="only print the plan")
    args = ap.parse_args()

    data_cfg = load_config(args.config)["data"]
    root, out, size = Path(data_cfg["root"]), Path(args.out), int(data_cfg["source_size"])
    classes = read_class_names(root, data_cfg["num_classes"])
    m = pd.read_csv(root / "manifest.csv", dtype=str)
    assert not (m["source"] == args.source).any(), f"{args.source} is already in manifest.csv"
    new = load_ckplus(SOURCES[args.source], size)
    assert set(new["label"]) <= set(classes)

    orig = m[m["source"] != "augmented"]
    shares = (orig["split"].value_counts() / len(orig)).to_dict()
    new["split"] = new["subject"].map(split_subjects(new, shares, args.seed))
    new["md5"] = [hashlib.md5(g.tobytes()).hexdigest() for g in new["img"]]
    new["file"] = [f"{c}/{args.source}_{h[:16]}.png" for c, h in zip(new["label"], new["md5"])]
    assert new["file"].is_unique, "two new images share a file name"

    # local images that make room, per split and class
    rng = np.random.default_rng(args.seed)
    gone = []
    for (split, label), g in new.groupby(["split", "label"]):
        pool = orig[(orig["split"] == split) & (orig["label"] == label) & (orig["source"] == "local")]
        assert len(pool) >= len(g), f"not enough local images in {split}/{label}"
        gone += pool["file"].iloc[rng.choice(len(pool), size=len(g), replace=False)].map(lambda f: (split, f)).tolist()
    gone_set = set(gone)
    aug = m[m["source"] == "augmented"]
    gone_aug = [("train", a) for a, o in zip(aug["file"], aug["origin"]) if ("train", o.split("/", 1)[1]) in gone_set]
    assert len(gone_aug) == sum(s == "train" for s, _ in gone), "a replaced train image has no augmented copy"
    dup = new["md5"].isin(m["md5"].dropna())
    assert not dup.any(), f"{dup.sum()} new images are pixel copies of images already in the dataset"

    print(f"{args.source}: {len(new):,} images of {new['subject'].nunique()} people")
    print(pd.crosstab(new["label"], new["split"]).reindex(columns=list(SPLITS), fill_value=0).to_string())
    print(f"people per split: {new.groupby('split')['subject'].nunique().to_dict()}")
    print(f"local images moved to {out}/ to keep the class counts: {len(gone):,} originals + {len(gone_aug):,} augmented copies")
    if args.dry_run:
        print("dry run: nothing changed")
        return

    out.mkdir(parents=True, exist_ok=True)
    backup = out / f"manifest_before_{args.source}.csv"
    if not backup.exists():
        shutil.copy2(root / "manifest.csv", backup)
    reason = f"replaced by {args.source} (same class and split)"
    for split, f in gone + gone_aug:
        dst = out / split / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(root / split / f, dst)
    log = out / "removed.csv"
    pd.DataFrame([{"split": s, "file": f, "label": f.split("/")[0], "reason": reason, "distance": None,
                   "ncc": None, "kind": "replace"} for s, f in gone + gone_aug]).to_csv(
        log, mode="a", header=not log.exists(), index=False)

    rows = []
    aug_rng = np.random.default_rng(args.seed)
    for r in new.itertuples():
        Image.fromarray(r.img).save(root / r.split / r.file)
        mirror_inv = min(dhash(r.img), dhash(np.ascontiguousarray(r.img[:, ::-1])))   # manifest convention
        rows.append({"file": r.file, "label": r.label, "source": args.source, "origin": r.origin, "md5": r.md5,
                     "dhash": f"{mirror_inv:016x}", "status": "ok", "group": "external", "split": r.split})
        if r.split == "train":
            aug_file = f"{r.label}/aug_{Path(r.file).name}"
            Image.fromarray(augment(r.img, aug_rng)).save(root / "train" / aug_file)
            rows.append({"file": aug_file, "label": r.label, "source": "augmented", "origin": f"train/{r.file}",
                         "group": "augmented", "split": "train"})
    moved = gone_set | set(gone_aug)
    keep = [(s, f) not in moved for s, f in zip(m["split"], m["file"])]
    merged = pd.concat([m[keep], pd.DataFrame(rows)], ignore_index=True)
    merged.to_csv(root / "manifest.csv", index=False)
    print(f"added {len(new):,} {args.source} images (+{(new['split'] == 'train').sum():,} augmented copies); "
          f"manifest.csv: {len(m):,} -> {len(merged):,} rows; log: {log}")
    print("next: python -m src.data.index_splits && python -m src.data.inspect")


if __name__ == "__main__":
    main()
