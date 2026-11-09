"""Index CK+ for the CK+ stage: images, subject-independent folds and the transfer split.

Reads the CK+ parquet (48x48 grayscale, the last 3 frames of the labelled sequences; CK+ has no
neutral and contempt is dropped, so 6 classes), writes the images to ``{root}/images/<class>/`` and
``{root}/index.csv`` (path,label,label_idx,source,subject,sequence,fold,main_split), then the split
CSVs read by src.train (same columns as data/splits/):
- ``{root}/splits/fold{k}/``: test = fold k, val = fold k+1, train = the other folds (src.ckplus_cv);
- ``{root}/splits/transfer/``: the CK+ people of each split of dataset/ stay in that split, so a
  model pretrained here never saw a val or test person of dataset/. If dataset/ has no CK+ images
  (no ckplus rows in manifest.csv): val = fold 0, train = the other folds, test empty.
Folds come from StratifiedGroupKFold over people: all frames of one person are in one fold.
dataset/ is only read. Settings: the ``ckplus:`` block of configs/base.yaml.

Example:
    python -m src.data.ckplus_folds
"""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from src.data.add_external import SOURCES, load_ckplus
from src.utils.config import load_config

SPLITS = ("train", "val", "test")
COLS = ["path", "label", "label_idx", "source"]


def write_split(out: Path, parts: dict) -> None:
    """Write ``{out}/{split}.csv`` for each DataFrame in ``parts``."""
    out.mkdir(parents=True, exist_ok=True)
    for split, df in parts.items():
        df[COLS].to_csv(out / f"{split}.csv", index=False)


def main_splits(df: pd.DataFrame, manifest: Path) -> pd.Series:
    """Split of each CK+ person in dataset/, or '' if CK+ is not in it.

    src.data.add_external split CK+ by person; a frame that de-duplication later moved out of
    dataset/ takes the split of its person.
    """
    if not manifest.exists():
        return pd.Series("", index=df.index)
    m = pd.read_csv(manifest, usecols=["source", "origin", "split"], dtype=str)
    m = m[m["source"] == "ckplus"]
    if m.empty:
        return pd.Series("", index=df.index)
    split = df["origin"].map(dict(zip(m["origin"], m["split"])))
    per_person = split.groupby(df["subject"]).unique()
    per_person = per_person.map(lambda s: [x for x in s if isinstance(x, str)])
    assert per_person.map(len).eq(1).all(), "a CK+ person has no split, or more than one, in dataset/"
    return df["subject"].map(per_person.str[0])


def main():
    ap = argparse.ArgumentParser(description="Index CK+ into subject-independent folds and the transfer split.")
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ck, data_cfg = cfg["ckplus"], cfg["data"]
    root, k = Path(ck["root"]), int(ck["folds"])

    df = load_ckplus(dict(SOURCES["ckplus"], file=ck["parquet"]), int(data_cfg["source_size"]))
    classes = sorted(df["label"].unique())
    assert len(classes) == ck["num_classes"], f"Expected {ck['num_classes']} CK+ classes, got {classes}"
    df["label_idx"] = df["label"].map({c: i for i, c in enumerate(classes)})
    df["source"] = "ckplus"
    df["sequence"] = df["name"].str.extract(r"^(S\d{3}_\d{3})_")[0]
    df["path"] = [(root / "images" / c / f"{Path(n).stem}.png").as_posix() for c, n in zip(df["label"], df["name"])]
    assert df["path"].is_unique, "two CK+ images share a file name"
    for path, img in zip(df["path"], df["img"]):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(path)

    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=int(ck["fold_seed"]))
    for f, (_, idx) in enumerate(sgkf.split(df, df["label_idx"], groups=df["subject"])):
        df.loc[idx, "fold"] = f
    assert (df.groupby("subject")["fold"].nunique() == 1).all(), "a person is in more than one fold"
    df["main_split"] = main_splits(df, Path(data_cfg["root"]) / "manifest.csv")
    df.drop(columns=["img", "name"]).to_csv(root / "index.csv", index=False)

    for f in range(k):
        val = (f + 1) % k
        write_split(root / "splits" / f"fold{f}", {
            "train": df[~df["fold"].isin([f, val])], "val": df[df["fold"] == val], "test": df[df["fold"] == f]})
    if (df["main_split"] != "").all():
        transfer = {s: df[df["main_split"] == s] for s in SPLITS}
    else:
        transfer = {"train": df[df["fold"] != 0], "val": df[df["fold"] == 0], "test": df.iloc[:0]}
    write_split(root / "splits" / "transfer", transfer)

    print(f"CK+: {len(df)} images, {df['subject'].nunique()} people, {df['sequence'].nunique()} sequences, "
          f"classes {classes}")
    table = pd.crosstab(df["fold"], df["label"])
    table["people"] = df.groupby("fold")["subject"].nunique()
    table["total"] = df.groupby("fold").size()
    print(table.to_string())
    print("transfer split: " + ", ".join(
        f"{s} {len(d)} images / {d['subject'].nunique()} people" for s, d in transfer.items()))
    print(f"-> {root / 'index.csv'}, {root / 'splits'}/fold0..fold{k - 1}/, {root / 'splits' / 'transfer'}/")


if __name__ == "__main__":
    main()
