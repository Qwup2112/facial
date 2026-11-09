"""Claude's visual review of data/vtv/review/_unsure: numbered 8x6 grids, labels typed per grid, moves logged.

Run from the project root (python -m src.data.vtv_claude_review <cmd>). Records in data/vtv/claude_review/.
Calibration (48 RAF-DB faces aligned like the VTV crops): Claude alone 58% right, ensemble 71%; where both agree
(28 of 48) 82%. So `apply` moves a face to a class only where Claude's label equals the ensemble guess; x (reject)
is applied as given. 2026-11-07: grids 1-43 (rare-class guesses, 2064 faces) reviewed.

  python -m src.data.vtv_claude_review calib              -> grids/calib.png + calib_key.csv (answers hidden in the csv)
  python -m src.data.vtv_claude_review make               -> grids/u###.png + grid_index.csv (unsure faces, rare guesses first)
  python -m src.data.vtv_claude_review grid N             -> print the ids of grid N
  python -m src.data.vtv_claude_review apply              -> move faces per labels.txt, log to data/vtv/claude_review.csv
Label codes: a angry, d disgust, f fear, h happy, n neutral, s sad, u surprise, x reject, ? keep in _unsure.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = ROOT / "data/vtv/claude_review"
GRIDS = HERE / "grids"
REVIEW = ROOT / "data/vtv/review"
CODES = {"a": "angry", "d": "disgust", "f": "fear", "h": "happy", "n": "neutral", "s": "sad", "u": "surprise", "x": "_reject"}
COLS, ROWS, T = 8, 6, 112


def grid(imgs, path):
    tiles = []
    for k, g in enumerate(imgs):
        t = np.full((T + 16, T), 255, np.uint8)
        t[16:] = cv2.resize(g, (T, T), interpolation=cv2.INTER_AREA)
        cv2.putText(t, str(k + 1), (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
        tiles.append(np.pad(t, ((0, 4), (0, 4)), constant_values=255))
    while len(tiles) < COLS * ROWS:
        tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]))


def calib():
    from src.data.vtv_faces import Detector, align, ckplus_template, imread
    det = Detector(ROOT / "data/external/models/yunet/face_detection_yunet_2023mar.onnx", 0.5, 1280)
    tpl = ckplus_template(ROOT / "data/ckplus/images", det)
    d = pd.read_csv(ROOT / "data/splits/test.csv")
    d = d[d.source == "rafdb"]
    d = pd.concat([g.sample(min(8, len(g)), random_state=1) for _, g in d.groupby("label")]).sample(frac=1, random_state=2)
    imgs, keys = [], []
    for p, lab in zip(d.path, d.label):
        g = cv2.resize(imread(ROOT / p, cv2.IMREAD_GRAYSCALE), (192, 192), interpolation=cv2.INTER_CUBIC)
        f = det(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
        if f:
            imgs.append(align(g, max(f, key=lambda x: x[0])[2], tpl, 192)[0]); keys.append(lab)
        if len(imgs) == COLS * ROWS:
            break
    GRIDS.mkdir(exist_ok=True)
    grid(imgs, GRIDS / "calib.png")
    pd.DataFrame({"n": range(1, len(keys) + 1), "label": keys}).to_csv(HERE / "calib_key.csv", index=False)
    print(len(imgs), "calibration faces")


def make():
    # a new round renumbers the grids: keep the previous round's labels and index (never overwrite review records)
    if (HERE / "labels.txt").exists():
        k = 1
        while (HERE / f"round{k}").exists():
            k += 1
        (HERE / f"round{k}").mkdir(parents=True)
        for name in ("labels.txt", "grid_index.csv"):
            if (HERE / name).exists():
                (HERE / name).replace(HERE / f"round{k}" / name)
        print(f"previous round kept in {HERE / f'round{k}'}")
    parts = [pd.read_csv(p, usecols=["id", "ensemble", "ensemble_conf"]) for p in (ROOT / "data/vtv").glob("faces*.csv")]
    f = pd.concat(parts).drop_duplicates("id").set_index("id")
    ids = [p.stem for p in (REVIEW / "_unsure").glob("*.png")]
    u = f.loc[[i for i in ids if i in f.index]].reset_index()
    order = {c: k for k, c in enumerate(["fear", "disgust", "angry", "surprise", "sad", "happy", "neutral"])}
    u["o"] = u["ensemble"].map(order)
    u = u.sort_values(["o", "ensemble_conf"], ascending=[True, False]).reset_index(drop=True)
    per = COLS * ROWS
    u["grid"] = u.index // per + 1
    u["n"] = u.index % per + 1
    GRIDS.mkdir(exist_ok=True)
    for gnum, g in u.groupby("grid"):
        grid([cv2.imdecode(np.fromfile(str(REVIEW / "_unsure" / f"{i}.png"), np.uint8), 0) for i in g["id"]], GRIDS / f"u{gnum:03d}.png")
    HERE.mkdir(parents=True, exist_ok=True)
    u[["grid", "n", "id", "ensemble", "ensemble_conf"]].to_csv(HERE / "grid_index.csv", index=False)
    print(len(u), "faces in", u["grid"].max(), "grids;", u.groupby("ensemble")["grid"].agg(["min", "max"]).to_dict("index"))


def apply():
    idx = pd.read_csv(HERE / "grid_index.csv")
    lab, ens = {}, dict(zip(idx["id"], idx["ensemble"]))
    for line in (HERE / "labels.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        gnum, codes = line.split(":", 1)
        codes = codes.split()
        g = idx[idx.grid == int(gnum)].sort_values("n")
        assert len(codes) == len(g), f"grid {gnum}: {len(codes)} labels for {len(g)} faces"
        lab.update(dict(zip(g["id"], codes)))
    log_path = ROOT / "data/vtv/claude_review.csv"
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame(columns=["id", "from", "to", "code"])
    done = set(log["id"])
    new, skipped = [], 0
    for i, c in lab.items():
        if i in done or c == "?":
            continue
        if c != "x" and CODES[c] != ens[i]:   # calibration: my label alone 58% right, with the ensemble 82%
            skipped += 1
            continue
        src = REVIEW / "_unsure" / f"{i}.png"
        if not src.exists():
            continue
        dst = REVIEW / CODES[c] / f"{i}.png"
        dst.parent.mkdir(exist_ok=True)
        src.replace(dst)
        new.append({"id": i, "from": "_unsure", "to": CODES[c], "code": c})
    log = pd.concat([log, pd.DataFrame(new)], ignore_index=True)
    log.to_csv(log_path, index=False)
    print(f"disagreed with the ensemble (stay in _unsure): {skipped}"); print(f"moved {len(new)} (total reviewed by Claude: {len(log)}); this batch:", pd.Series([n['to'] for n in new]).value_counts().to_dict() if new else {})


if __name__ == "__main__":
    {"calib": calib, "make": make, "apply": apply,
     "grid": lambda: print(pd.read_csv(HERE / "grid_index.csv").query(f"grid == {int(sys.argv[2])}").to_string())}[sys.argv[1]]()
