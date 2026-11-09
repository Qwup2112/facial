"""VTV step 1d: add the faces that a re-run with other gates found to review/, without touching what is there.

Used for the 360p frame images (VTV_Emotion_Dataset_Final/final_dataset/images) and then the videos: a re-run
with a lower eye-distance gate (configs/data/vtv_faces_eye16.yaml, 16 px) finds more of their small faces. The re-run goes to a temporary folder:
    python -m src.data.vtv_faces --config <cfg with other gates> --out <tmp> --only final_dataset
It finds the old faces again (same ids) plus new ones. This keeps only the new ones:
  - id not in any data/vtv/faces*.csv;
  - not a repeat of a face already in review/ (any folder, also _reject/), with the rules of vtv_faces.Dedup:
    NCC >= group_ncc with any face of the same group, or, within recent_s seconds of video (recent_frames frames
    for frame images), NCC >= recent_ncc_any,
    or NCC >= recent_ncc with the same FER+ suggestion.
They are copied to review/<FER+ folder>/ (an existing file is never replaced) and listed in
data/vtv/faces_<tag>.csv (+ summary_<tag>.md). Then:
    python -m src.data.vtv_relabel --csv faces_<tag>.csv      # ensemble re-sort of the new faces only
    python -m src.data.vtv_build

Example:
    python -m src.data.vtv_supplement --run <tmp> --tag images_eye16 --dry-run
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.vtv_faces import imread, ncc_vector
from src.utils.config import load_config


def vector(png: Path) -> np.ndarray:
    g = imread(png, cv2.IMREAD_GRAYSCALE)
    return ncc_vector(cv2.resize(g, (48, 48), interpolation=cv2.INTER_AREA))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/data/vtv_faces.yaml")
    ap.add_argument("--run", required=True, help="output folder of the vtv_faces re-run")
    ap.add_argument("--tag", required=True, help="name of the new faces_<tag>.csv")
    ap.add_argument("--dry-run", action="store_true", help="only count, copy nothing")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dd = cfg["dedup"]
    out, run = Path(cfg["out_dir"]), Path(args.run)
    review = out / "review"
    dst_csv = out / f"faces_{args.tag}.csv"
    assert not dst_csv.exists(), f"{dst_csv} exists: pick another --tag"

    old = pd.concat([pd.read_csv(p, usecols=["id", "group", "frame", "time_s", "suggested"]) for p in out.glob("faces*.csv")])
    new = pd.read_csv(run / "faces.csv")
    found_again = new["id"].isin(old["id"])
    new = new[~found_again].reset_index(drop=True)
    where = {p.stem: p for p in review.glob("*/*.png")}
    old = old[old["group"].isin(set(new["group"])) & old["id"].isin(where.keys())]
    print(f"re-run: {len(found_again)} faces, {int(found_again.sum())} already in review/, {len(new)} new candidates; "
          f"comparing with {len(old)} faces of the same groups in review/", flush=True)

    # position in the group: video time in s (window recent_s) or frame number of frame images (recent_frames)
    old["pos"] = old["time_s"].fillna(old["frame"])
    ref = {g: (d["pos"].to_numpy(float), np.stack([vector(where[i]) for i in d["id"]]), d["suggested"].to_numpy())
           for g, d in old.groupby("group")}
    keep, why = [], []
    for r in new.itertuples():
        v = vector(run / "review" / r.review_folder / f"{r.id}.png")
        reason = ""
        if r.group in ref:
            pos, vecs, labels = ref[r.group]
            s = vecs @ v
            video = not pd.isna(r.time_s)
            near = np.abs(pos - (r.time_s if video else r.frame)) <= (dd["recent_s"] if video else dd["recent_frames"])
            if (s >= dd["group_ncc"]).any():
                reason = "repeat (group_ncc)"
            elif (near & ((s >= dd["recent_ncc_any"]) | ((s >= dd["recent_ncc"]) & (labels == r.suggested)))).any():
                reason = "repeat (recent)"
        keep.append(not reason)
        why.append(reason or "new")
    new["supplement"] = why
    print(new["supplement"].value_counts().to_string())
    add = new[keep].drop(columns="supplement")
    print("new faces by FER+ folder:", add["review_folder"].value_counts().to_dict())
    if args.dry_run:
        return
    for r in add.itertuples():
        dst = review / r.review_folder / f"{r.id}.png"
        assert not dst.exists() and r.id not in where, dst
        dst.parent.mkdir(exist_ok=True)
        shutil.copy2(run / "review" / r.review_folder / f"{r.id}.png", dst)
    add.to_csv(dst_csv, index=False)
    lines = [f"# Supplement `{args.tag}`\n", f"Re-run: `{run}` (its gates: `{run / 'summary.md'}`).\n",
             f"{len(found_again)} faces found, {int(found_again.sum())} already in review/, {len(new)} new ids, "
             f"{len(add)} added (the rest repeat a face already in review/).\n",
             "| Review folder (FER+) | Faces |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in add["review_folder"].value_counts().items()]
    if (run / "summary.md").exists():
        lines += ["", "## Re-run summary", "", (run / "summary.md").read_text(encoding="utf-8")]
    (out / f"summary_{args.tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"added {len(add)} faces to {review}; {dst_csv}")


if __name__ == "__main__":
    main()
