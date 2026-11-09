"""Accuracy against human labels on in-the-wild faces prepared exactly like the VTV crops.

The VTV labels are model suggestions, so the VTV test numbers are only agreement. This scores trained runs on
RAF-DB (human labels, val + test of the course dataset; no VTV-only model trained on it) after the same
preparation as the VTV faces and the demo: 48x48 -> 192 (cubic) -> YuNet -> CK+ landmark alignment -> 48x48.
It also scores the label-suggestion ensemble (FER+ + two HSEmotion models, src.data.vtv_relabel), which the
VTV models learned from: their ceiling, roughly. Faces YuNet does not find are skipped for every model alike.
Runs trained on data/mix (src.data.mix_build) picked their best epoch on the RAF-DB val faces: compare with --splits test.

  python -m src.eval_wild --runs runs/transfer_vtv_claude/resnet18_s2024_20261108-1516 runs/transfer_vtv_claude/mobilenet_v2_s42_*
"""
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from src.data.transforms import build_transforms, resize_batch
from src.data.vtv_faces import CLASSES, FERPLUS, Detector, FerPlus, align, ckplus_template, imread
from src.data.vtv_relabel import HSEmotion
from src.predict import load_run

ROOT = Path(__file__).resolve().parents[1]
FACES_CFG = ROOT / "configs/data/vtv_faces.yaml"
TEMPLATE = ROOT / "data/vtv/ckplus_template.npy"


def prepare(source: str, splits=("val", "test")):
    """Aligned 192x192 crops and labels of one source's faces in the given splits of the course dataset."""
    fc = yaml.safe_load(FACES_CFG.read_text(encoding="utf-8"))
    det = Detector(ROOT / fc["models"]["yunet"], 0.5, 1280)
    tpl = np.load(TEMPLATE) if TEMPLATE.exists() else ckplus_template(ROOT / fc["ckplus_images"], det)
    d = pd.concat([pd.read_csv(ROOT / f"data/splits/{s}.csv") for s in splits])
    d = d[d["source"] == source]
    crops, labels, missed = [], [], 0
    for p, lab in zip(d["path"], d["label"]):
        g = cv2.resize(imread(ROOT / p, cv2.IMREAD_GRAYSCALE), (192, 192), interpolation=cv2.INTER_CUBIC)
        f = det(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
        if not f:
            missed += 1
            continue
        crops.append(align(g, max(f, key=lambda x: x[0])[2], tpl, 192)[0])
        labels.append(lab)
    return crops, np.array(labels), missed, fc


@torch.inference_mode()
def run_probs(run: Path, crops, device) -> tuple[str, np.ndarray]:
    model, cfg, classes = load_run(run, device)
    assert classes == CLASSES, classes
    tf = build_transforms(cfg, train=False)
    out = []
    for s in range(0, len(crops), 128):
        x = torch.stack([tf(Image.fromarray(cv2.resize(c, (48, 48), interpolation=cv2.INTER_AREA), "L"))
                         for c in crops[s:s + 128]]).to(device)
        out.append(model(resize_batch(x, cfg)).float().softmax(1).cpu().numpy())
    return f"{cfg['model']['name']} s{cfg['seed']} ({run.parent.name.removeprefix('transfer_')})", np.concatenate(out)


def ensemble_probs(crops, fc) -> np.ndarray:
    fer = FerPlus(ROOT / fc["models"]["ferplus"])
    idx = [FERPLUS.index(c) for c in CLASSES]
    p = np.stack([fer(c)[idx] for c in crops])
    p /= p.sum(1, keepdims=True)
    for m in fc["relabel"]["hsemotion"]:
        h = HSEmotion(ROOT / m, 4)
        p += np.concatenate([h(crops[s:s + 16]) for s in range(0, len(crops), 16)])
    return p / (1 + len(fc["relabel"]["hsemotion"]))


def scores(y, p) -> dict:
    pred = np.array(CLASSES)[p.argmax(1)]
    rec = {c: 100 * (pred[y == c] == c).mean() for c in CLASSES if (y == c).any()}
    f1 = []
    for c in CLASSES:
        tp = ((pred == c) & (y == c)).sum()
        pr, rc = tp / max((pred == c).sum(), 1), tp / max((y == c).sum(), 1)
        f1.append(0 if tp == 0 else 2 * pr * rc / (pr + rc))
    return {"acc": 100 * (pred == y).mean(), "mf1": float(np.mean(f1)), **rec,
            "pred_share": {c: 100 * (pred == c).mean() for c in CLASSES}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True, help="run directories with best.pt (globs allowed)")
    ap.add_argument("--source", default="rafdb", help="rafdb (all 7 classes) or affectnet (angry/disgust/fear)")
    ap.add_argument("--splits", nargs="+", default=["val", "test"], help="course splits to score (test only for mix runs)")
    ap.add_argument("--out", default="reports/wild_check_rafdb.md")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crops, y, missed, fc = prepare(args.source, args.splits)
    print(f"{args.source}: {len(crops)} faces aligned, {missed} not found by YuNet")
    res = {"suggestion ensemble (FER+ + 2 HSEmotion)": ensemble_probs(crops, fc)}
    ours, done = [], []
    for r in [p for pat in args.runs for p in sorted(glob.glob(pat))]:
        name, p = run_probs(Path(r), crops, device)
        res[name] = p
        ours.append(p)
        done.append(r)
        print(name, f"{scores(y, p)['acc']:.1f}%")
    if len(ours) > 1 and len(set(Path(r).parent for r in done)) == 1:   # seeds of one setup, not different datasets
        res[f"mean of our {len(ours)} runs"] = np.mean(ours, 0)
    present = [c for c in CLASSES if (y == c).any()]
    L = [f"# Accuracy on {args.source} faces with human labels (prepared like the VTV crops)", "",
         f"{len(crops)} faces of the course dataset's {' + '.join(args.splits)} (`source = {args.source}`), 48 → 192 → YuNet → "
         f"CK+ alignment → 48×48 ({missed} not found by YuNet, skipped). No model here trained on these faces. "
         "This is real accuracy (human labels), unlike the VTV test agreement.", "",
         "| Class | " + " | ".join(present) + " |", "|---" * (len(present) + 1) + "|",
         "| faces | " + " | ".join(str((y == c).sum()) for c in present) + " |", "",
         "| Model | Accuracy (%) | Macro-F1 | " + " | ".join(f"{c} recall" for c in present) + " |",
         "|---|---|---|" + "---|" * len(present)]
    shares = []
    for name, p in res.items():
        s = scores(y, p)
        L.append(f"| {name} | {s['acc']:.1f} | {s['mf1']:.3f} | " + " | ".join(f"{s[c]:.0f}" for c in present) + " |")
        shares.append(f"| {name} | " + " | ".join(f"{s['pred_share'][c]:.0f}" for c in CLASSES) + " |")
    L += ["", "Share of faces predicted as each class (%), true share in the first table:", "",
          "| Model | " + " | ".join(CLASSES) + " |", "|---" * (len(CLASSES) + 1) + "|"] + shares
    text = "\n".join(L) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
