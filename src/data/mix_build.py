"""VTV + course dataset, all aligned like the VTV crops: more human-labelled faces for the weak classes.

The VTV models are weak on in-the-wild faces with human labels (RAF-DB accuracy ~53% vs 67% for the label
ensemble they learned from, src.eval_wild): the VTV train set has only a few hundred real angry, disgust,
fear and surprise faces. The course dataset (``dataset/``, never modified) has ~3,400 real faces per rare
class with human labels (local, RAF-DB, AffectNet, CK+). This aligns its real faces (not its augmented
copies) the same way as the VTV faces and the demo (48 -> 192 cubic -> YuNet -> CK+ landmark alignment ->
48x48) and writes split CSVs that join them with a VTV build:

  data/mix/course/<split>/<class>/<name>.png     aligned course faces (faces YuNet misses are skipped)
  data/mix/vtv/dataset/...                       the VTV build, unzipped (default: target_dataset_vtv_suggested.zip)
  data/mix/splits/{train,val,test}.csv           path,label,label_idx,source (sources: vtv, augmented, local, rafdb, ...)

Splits stay as they are in both datasets (VTV by person, course as given), so the RAF-DB test faces are
never trained on or used for model selection.

  python -m src.data.mix_build
  python -m src.train --config configs/transfer/resnet18_mix.yaml --seed 42 --init <CK+ resnet18 s42 run>
"""
import argparse
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from src.data.vtv_faces import CLASSES, Detector, align, ckplus_template, imread, imwrite

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/mix"
TEMPLATE = ROOT / "data/vtv/ckplus_template.npy"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vtv-zip", default="data/vtv/target_dataset_vtv_suggested.zip")
    args = ap.parse_args()
    fc = yaml.safe_load((ROOT / "configs/data/vtv_faces.yaml").read_text(encoding="utf-8"))
    det = Detector(ROOT / fc["models"]["yunet"], 0.5, 1280)
    tpl = np.load(TEMPLATE) if TEMPLATE.exists() else ckplus_template(ROOT / fc["ckplus_images"], det)

    vz = ROOT / args.vtv_zip
    (OUT / "vtv").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(vz) as z:
        bad = z.testzip()
        assert bad is None, f"{vz}: bad CRC in {bad}"
        z.extractall(OUT / "vtv")
    m = pd.read_csv(OUT / "vtv/dataset/manifest.csv")
    rows = {s: [] for s in ("train", "val", "test")}
    for r in m.itertuples():
        rows[r.split].append({"path": (OUT / "vtv/dataset" / r.split / r.file).as_posix(), "label": r.label,
                              "source": "augmented" if r.source == "augmented" else "vtv"})
    print(f"VTV ({vz.name}): " + ", ".join(f"{s} {len(v)}" for s, v in rows.items()))

    missed = {}
    for split in ("train", "val", "test"):
        d = pd.read_csv(ROOT / f"data/splits/{split}.csv")
        d = d[d["source"] != "augmented"]
        for p, lab, src in zip(d["path"], d["label"], d["source"]):
            dst = OUT / "course" / split / lab / Path(p).name
            if not dst.exists():
                g = cv2.resize(imread(ROOT / p, cv2.IMREAD_GRAYSCALE), (192, 192), interpolation=cv2.INTER_CUBIC)
                f = det(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
                if not f:
                    missed[(split, src)] = missed.get((split, src), 0) + 1
                    continue
                crop = align(g, max(f, key=lambda x: x[0])[2], tpl, 192)[0]
                dst.parent.mkdir(parents=True, exist_ok=True)
                imwrite(dst, cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA))
            rows[split].append({"path": dst.as_posix(), "label": lab, "source": src})
        print(f"course {split}: {len(d)} real faces, not found by YuNet: "
              f"{sum(v for (s, _), v in missed.items() if s == split)}", flush=True)

    (OUT / "splits").mkdir(parents=True, exist_ok=True)
    for split, r in rows.items():
        df = pd.DataFrame(r)
        df["label_idx"] = df["label"].map(CLASSES.index)
        df[["path", "label", "label_idx", "source"]].to_csv(OUT / f"splits/{split}.csv", index=False)
        print(f"\n{split}: {len(df)} faces\n" + pd.crosstab(df["label"], df["source"], margins=True).to_string())
    if missed:
        print("not found by YuNet:", missed)


if __name__ == "__main__":
    main()
