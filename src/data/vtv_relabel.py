"""VTV target dataset, step 1b: re-sort the review folders with an ensemble of label suggesters.

FER+ alone (used by src.data.vtv_faces) hardly ever predicts angry, fear or disgust, so most faces
landed in neutral/. This step adds two HSEmotion models (Savchenko, EfficientNet-B2 trained on
AffectNet; ONNX from github.com/HSE-asavchenko/face-emotion-recognition; not our own models) and
averages the three softmax outputs (contempt dropped, renormalised).

Measured on labelled faces prepared like the VTV crops (48x48 -> 192 -> YuNet -> CK+ alignment):
                  CK+ (927, posed)        RAF-DB val+test (1860, in the wild)
  FER+ alone      acc 0.659, mF1 0.584    acc 0.548, mF1 0.413
  ensemble        acc 0.887, mF1 0.866    acc 0.670, mF1 0.576
  ensemble, top probability >= 0.4 (min_conf): CK+ 97% of faces at 90% accuracy, RAF-DB 84% at 74%
  (>= 0.5: CK+ 87% at 95%, RAF-DB 65% at 80%, but only 45% of VTV faces sorted).

A face goes to the ensemble's class folder if its top probability >= ``min_conf``, else to
``_unsure/``. Only faces still in the folder the pipeline put them in are moved: a face the user
moved by hand (current folder != ``review_folder`` in faces*.csv), put in ``_reject/`` or checked in the
review app (``human_review.csv``, src.data.vtv_review_app) is never touched. faces*.csv get ``e_<class>`` (ensemble probabilities), ``ensemble``, ``ensemble_conf``,
and ``review_folder`` is updated to the new folder (so src.data.vtv_build still sees which labels
the human review changed). Idempotent: faces that already have ``ensemble`` are not recomputed.

Example:
    python -m src.data.vtv_relabel                     # all faces*.csv
    python -m src.data.vtv_relabel --csv faces.csv     # only the main run (e.g. while an --append run writes)
"""
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.vtv_faces import CLASSES, imread
from src.data.vtv_review_app import human_checked
from src.utils.config import load_config

FERPLUS = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]
HSE_CLASSES = {7: ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
               8: ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]}
KEEP = set(CLASSES) | {"_unsure", "_unlabeled"}  # folders the pipeline may move faces out of


class HSEmotion:
    """HSEmotion EfficientNet ONNX: gray crop -> probabilities in CLASSES order (mirror averaged)."""

    def __init__(self, path: Path, threads: int):
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = threads
        o.enable_cpu_mem_arena = False  # the arena kept ~4.5 GB after batches of 260x260 images
        o.enable_mem_pattern = False
        self.sess = ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.name, self.size = inp.name, inp.shape[-1]
        n = self.sess.get_outputs()[0].shape[-1]
        self.idx = [HSE_CLASSES[n].index(c) for c in CLASSES]
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
        self.std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]

    def __call__(self, grays: list) -> np.ndarray:
        x = np.stack([cv2.resize(g, (self.size, self.size), interpolation=cv2.INTER_LINEAR) for g in grays])
        x = (np.repeat(x[:, None].astype(np.float32) / 255, 3, 1) - self.mean) / self.std
        x = np.concatenate([x, x[:, :, :, ::-1]]).astype(np.float32)
        z = self.sess.run(None, {self.name: x})[0]
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        p = (p[:len(grays)] + p[len(grays):]) / 2
        p = p[:, self.idx]
        return p / p.sum(1, keepdims=True)


def ferplus_probs(df: pd.DataFrame) -> np.ndarray:
    """FER+ probabilities already in faces*.csv, in CLASSES order without contempt."""
    p = df[[f"p_{c}" for c in CLASSES]].to_numpy(float)
    return p / p.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/data/vtv_faces.yaml")
    ap.add_argument("--csv", nargs="+", help="faces*.csv names in out_dir (default: all)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rl = cfg["relabel"]
    out = Path(cfg["out_dir"])
    review = out / "review"
    csvs = [out / c for c in args.csv] if args.csv else sorted(out.glob("faces*.csv"))
    now = {p.stem: p.parent.name for p in review.glob("*/*.png")}
    checked = human_checked(out)  # faces a human checked in the review app: never moved
    models = [HSEmotion(Path(m), rl["threads"]) for m in rl["hsemotion"]]
    t0 = time.time()

    for csv in csvs:
        df = pd.read_csv(csv)
        if df.empty:
            continue
        for c in ["ensemble", "ensemble_conf"] + [f"e_{c}" for c in CLASSES]:
            if c not in df:
                df[c] = np.nan if c != "ensemble" else ""
        df["ensemble"] = df["ensemble"].fillna("").astype(str)
        todo = df.index[(df["ensemble"] == "") & df["id"].isin(now.keys())]
        for s in range(0, len(todo), rl["batch"]):
            b = todo[s:s + rl["batch"]]
            grays = [imread(review / now[i] / f"{i}.png", cv2.IMREAD_GRAYSCALE) for i in df.loc[b, "id"]]
            p = (ferplus_probs(df.loc[b]) + sum(m(grays) for m in models)) / (1 + len(models))
            df.loc[b, [f"e_{c}" for c in CLASSES]] = p
            df.loc[b, "ensemble"] = [CLASSES[k] for k in p.argmax(1)]
            df.loc[b, "ensemble_conf"] = p.max(1)
            if (s // rl["batch"]) % 20 == 0:
                print(f"{csv.name}: {s + len(b)}/{len(todo)} faces, {time.time() - t0:.0f} s", flush=True)

        moves, kept_by_user = [], 0
        for i in df.index[df["ensemble"] != ""]:
            fid, old = df.at[i, "id"], df.at[i, "review_folder"]
            cur = now.get(fid)
            if cur is None or cur != old or cur not in KEEP or fid in checked:
                kept_by_user += cur is not None and (cur != old or fid in checked)
                continue  # moved or rejected by the user (or deleted): leave it
            new = df.at[i, "ensemble"] if df.at[i, "ensemble_conf"] >= rl["min_conf"] else "_unsure"
            if new != cur:
                (review / new).mkdir(exist_ok=True)
                os.replace(review / cur / f"{fid}.png", review / new / f"{fid}.png")
                df.at[i, "review_folder"] = new
                now[fid] = new
                moves.append((cur, new))
        df.to_csv(csv, index=False)
        print(f"{csv.name}: {len(df)} faces, moved {len(moves)}, left alone (moved or checked by the user) {kept_by_user}")
        if moves:
            print(pd.Series([f"{a} -> {b}" for a, b in moves]).value_counts().head(25).to_string())

    counts = pd.Series(now).value_counts()
    print("review folders now:\n" + counts.to_string())


if __name__ == "__main__":
    main()
