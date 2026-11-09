"""VTV target dataset, step 2: build the 48x48 dataset from the review folders.

Reads ``data/vtv/review/<class>/*.png`` (only the 7 class folders: ``_unsure``, ``_reject`` and
``_unlabeled`` are left out) and ``data/vtv/faces*.csv`` (written by src.data.vtv_faces).

The split is **subject-independent**, like the CK+ CV: one actor appears in many videos and
episodes, so splitting by video alone would put the same person in train and test. Faces are
grouped by person with SFace embeddings (OpenCV face recognition model; average-linkage
clustering, cosine similarity >= ``identity_sim``, calibrated in the config), then
StratifiedGroupKFold over people: test = fold 0, val = fold 1, train = the other folds.
The report checks the result: share of val/test faces that still have ``leak_k`` train faces
of very high similarity (the same person in train), plus ``leak_check.png`` to check by eye.

Writes ``<dataset_dir>/<split>/<class>/<id>.png`` (48x48 grayscale, area downscale of the aligned
192x192 review crop, the same CK+ framing), ``manifest.csv``, ``report.md``, ``person_check.png``
(largest person clusters, to check the grouping by eye) and ``<zip>`` with the same files under
``dataset/`` for Colab (upload as MyDrive/facial_emotion/target_dataset.zip, notebook section 5).

Re-run after every change in review/ (it rebuilds from scratch; embeddings are cached).

Example:
    python -m src.data.vtv_build
"""
import argparse
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.model_selection import StratifiedGroupKFold

from src.data.vtv_faces import CLASSES, Detector, imread, imwrite
from src.data.vtv_claude_suggest import seen_kept
from src.data.vtv_review_app import human_checked
from src.utils.config import load_config

SPLITS = ("train", "val", "test")


class Embedder:
    """SFace identity embedding (128-d, L2-normalised) of an aligned review crop; None if no face."""

    def __init__(self, sface: Path, yunet: Path):
        self.rec = cv2.FaceRecognizerSF.create(str(sface), "")
        self.det = Detector(yunet, 0.5, 1280)

    def __call__(self, gray: np.ndarray):
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.det.net.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = self.det.net.detect(bgr)
        if faces is None:
            return None
        row = max(faces, key=lambda f: f[14])
        e = self.rec.feature(self.rec.alignCrop(bgr, row)).ravel().astype(np.float32)
        return e / (np.linalg.norm(e) + 1e-9)


def load_meta(out: Path) -> pd.DataFrame:
    """All rows of faces*.csv (main run + --append runs), indexed by face id."""
    parts = [pd.read_csv(f) for f in sorted(out.glob("faces*.csv"))]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=["group", "source", "review_folder"]).rename_axis("id")
    return pd.concat(parts).drop_duplicates("id", keep="last").set_index("id")


def embeddings(ids, files, cache: Path, embed: Embedder) -> dict:
    """id -> embedding (or None); computed once and cached in ``cache`` (npz)."""
    done = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        done = {i: (e if np.isfinite(e).all() else None) for i, e in zip(z["ids"], z["emb"])}
    for i, f in zip(ids, files):
        if i not in done:
            done[i] = embed(imread(f, cv2.IMREAD_GRAYSCALE))
    keys = sorted(done)
    emb = np.stack([done[k] if done[k] is not None else np.full(128, np.nan, np.float32) for k in keys])
    np.savez(cache, ids=np.array(keys), emb=emb)
    return done


def cross_pairs(df: pd.DataFrame, emb: dict, sim: float) -> list:
    """(id, id, cosine) of faces from different groups with SFace cosine >= sim (chunked)."""
    ids = [i for i in df.index if emb.get(i) is not None]
    x = np.stack([emb[i] for i in ids])
    g = df.loc[ids, "group"].to_numpy()
    out = []
    for c in range(0, len(ids), 2000):
        s = x[c:c + 2000] @ x.T
        a, b = np.where(s >= sim)
        a2 = a + c
        m = (a2 < b) & (g[a2] != g[b])
        out += [(ids[i], ids[j], float(v)) for i, j, v in zip(a2[m], b[m], s[a[m], b[m]])]
    return out


def link_groups(df: pd.DataFrame, pairs: list) -> pd.Series:
    """Merge groups that share a face pair (same person across sources) into one component.

    The frame_*.jpg images have no video name and are grouped in blocks, but they are frames of
    the same YouTube videos as the playlist_*.jpg images: without this, near-identical frames of
    one person end up in train and test.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    gi = {k: n for n, k in enumerate(sorted(df["group"].unique()))}
    a = [gi[df.at[i, "group"]] for i, _, _ in pairs]
    b = [gi[df.at[j, "group"]] for _, j, _ in pairs]
    _, lab = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(len(gi), len(gi))), directed=False)
    return df["group"].map(lambda k: f"c{lab[gi[k]]:03d}")


def duplicates(pairs: list, files: dict, sim: float, ncc: float) -> set:
    """Faces that are a copy of a face from another source (SFace >= sim and pixel NCC >= ncc); keeps the first id."""
    from src.data.vtv_faces import ncc_vector
    vec = {}

    def v(i):
        if i not in vec:
            g = cv2.resize(imread(files[i], cv2.IMREAD_GRAYSCALE), (48, 48), interpolation=cv2.INTER_AREA)
            vec[i] = ncc_vector(g)
        return vec[i]
    drop = set()
    for i, j, s in sorted(pairs):
        if s >= sim and i not in drop and j not in drop and float(v(i) @ v(j)) >= ncc:
            drop.add(j)
    return drop


def person_ids(df: pd.DataFrame, emb: dict, sim: float, group_col: str = "group") -> pd.Series:
    """Person cluster of each face; a face without embedding counts as its video group.

    Two stages, so memory grows with the largest video, not with all faces (one average linkage
    over 16.6k faces needs ~2.2 GB and was killed for low memory):
    1. average linkage inside each group (video / image block) at ``sim``;
    2. average linkage across groups over these clusters, with the exact mean pairwise cosine
       similarity between two clusters (sum vectors: S_a . S_b / (n_a n_b)) at the same ``sim``.
    """
    ok = [i for i in df.index if emb.get(i) is not None]
    person = pd.Series({i: f"g_{df.at[i, 'group']}" for i in df.index})
    if not ok:
        return person
    pre = {}
    for g, ids in df.loc[ok].groupby(group_col).groups.items():
        ids = list(ids)
        if len(ids) == 1:
            pre[ids[0]] = (g, 1)
            continue
        lab = fcluster(linkage(np.stack([emb[i] for i in ids]).astype(np.float64), method="average", metric="cosine"),
                       t=1 - sim, criterion="distance")
        pre.update({i: (g, int(c)) for i, c in zip(ids, lab)})
    keys = sorted(set(pre.values()))
    kidx = {k: n for n, k in enumerate(keys)}
    s = np.zeros((len(keys), len(emb[ok[0]])))
    cnt = np.zeros(len(keys))
    for i in ok:
        s[kidx[pre[i]]] += emb[i]
        cnt[kidx[pre[i]]] += 1
    if len(keys) > 1:
        d = np.clip(1 - (s @ s.T) / np.outer(cnt, cnt), 0, None)
        np.fill_diagonal(d, 0)
        lab2 = fcluster(linkage(squareform(d, checks=False), method="average"), t=1 - sim, criterion="distance")
    else:
        lab2 = np.ones(1, int)
    person[ok] = [f"p{lab2[kidx[pre[i]]]:05d}" for i in ok]
    return person


def cap_per_person(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Keep at most ``cap`` faces per (person, label), spread evenly over the person's scenes.

    A few people with thousands of faces (e.g. the cast of one show) teach the model their faces
    rather than the expressions; faces evenly spaced in (video, frame) order keep different scenes.
    """
    frame = df.index.str.rsplit("_", n=2).str[1].astype(int)
    keep = []
    for _, g in df.assign(_f=frame).groupby(["person", "label"]):
        g = g.sort_values(["group", "_f"])
        keep += list(g.index if len(g) <= cap else g.index[np.linspace(0, len(g) - 1, cap).round().astype(int)])
    return df.loc[sorted(keep)]


def leakage(df: pd.DataFrame, emb: dict, files: dict, sim: float, k: int, path: Path, n_pairs: int = 16):
    """Share of val/test faces with >= k train faces at cosine >= sim (= same person in train).

    Also writes ``path``: for the test people closest to train, the test face (left) next to its
    nearest train face (right), to check by eye that they are different people.
    """
    has = df.index[[emb.get(i) is not None for i in df.index]]
    tr = [i for i in has if df.at[i, "split"] == "train"]
    Xtr = np.stack([emb[i] for i in tr]) if tr else np.zeros((0, 128), np.float32)
    out, pairs = {}, []
    for split in ("val", "test"):
        ids = [i for i in has if df.at[i, "split"] == split]
        if not ids or len(tr) < k:
            out[split] = float("nan")
            continue
        kth, best = [], []
        for c in range(0, len(ids), 1000):  # chunks: test x train similarity can be large
            s = np.stack([emb[i] for i in ids[c:c + 1000]]) @ Xtr.T
            kth.append(np.partition(s, -k, axis=1)[:, -k])
            best += [(i, tr[j], v) for i, j, v in zip(ids[c:c + 1000], s.argmax(1), s.max(1))]
        out[split] = float(np.mean(np.concatenate(kth) >= sim))
        if split == "test":
            b = pd.DataFrame(best, columns=["id", "train_id", "sim"])
            b["person"] = df.loc[b["id"], "person"].values
            pairs = b.sort_values("sim", ascending=False).drop_duplicates("person").head(n_pairs)
    if len(pairs):
        tile = lambda i: cv2.resize(imread(files[i], cv2.IMREAD_GRAYSCALE), (80, 80), interpolation=cv2.INTER_AREA)
        rows = []
        for _, r in pairs.iterrows():
            lab = np.full((80, 60), 255, np.uint8)
            cv2.putText(lab, f"{r['sim']:.2f}", (4, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
            rows.append(np.hstack([tile(r["id"]), tile(r["train_id"]), lab]))
        imwrite(path, np.vstack(rows))
    return out


def person_sheet(df: pd.DataFrame, files: dict, path: Path, n_people: int = 12, per: int = 10) -> None:
    """Grid: the largest person clusters (rows) with random faces of each, to check the grouping."""
    rng = random.Random(0)
    top = df["person"].value_counts().head(n_people)
    rows = []
    for p, cnt in top.items():
        ids = rng.sample(list(df.index[df["person"] == p]), min(per, cnt))
        tiles = [cv2.resize(imread(files[i], cv2.IMREAD_GRAYSCALE), (64, 64), interpolation=cv2.INTER_AREA) for i in ids]
        tiles += [np.zeros((64, 64), np.uint8)] * (per - len(tiles))
        label = np.full((64, 90), 255, np.uint8)
        cv2.putText(label, p, (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
        cv2.putText(label, f"{cnt} faces", (2, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
        rows.append(np.hstack([label] + tiles))
    if rows:
        imwrite(path, np.vstack(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/data/vtv_faces.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    b = cfg["build"]
    out = Path(cfg["out_dir"])
    review, ds = out / "review", Path(b["dataset_dir"])

    files = {f.stem: f for c in CLASSES for f in sorted((review / c).glob("*.png"))}
    label = {f.stem: f.parent.name for f in files.values()}
    assert files, f"no faces in {review}/<class>/"
    meta = load_meta(out)
    df = pd.DataFrame({"label": pd.Series(label)})
    df["group"] = [meta.at[i, "group"] if i in meta.index else i.rsplit("_", 2)[0] for i in df.index]
    df["source"] = [meta.at[i, "source"] if i in meta.index else "" for i in df.index]
    df["suggested"] = [meta.at[i, "review_folder"] if i in meta.index else "" for i in df.index]
    print(f"{len(df)} faces in the 7 class folders", flush=True)

    embed = Embedder(Path(b["sface"]), Path(cfg["models"]["yunet"]))
    emb = embeddings(list(df.index), [files[i] for i in df.index], out / "embeddings.npz", embed)
    pairs = cross_pairs(df, emb, b["link_sim"])
    dup = duplicates(pairs, files, b["dup_sim"], b["dup_ncc"])
    dup_disagree = sum(1 for i, j, _ in pairs if j in dup and df.at[i, "label"] != df.at[j, "label"])
    df = df.drop(index=sorted(dup))
    pairs = [p for p in pairs if p[0] not in dup and p[1] not in dup]
    df["component"] = link_groups(df, pairs)
    print(f"{len(dup)} copies of faces from another source dropped ({dup_disagree} with another label); "
          f"{df['group'].nunique()} groups -> {df['component'].nunique()} linked components", flush=True)
    df["person"] = person_ids(df, emb, b["identity_sim"], "component")
    print(f"{df['person'].nunique()} people (clusters), {sum(emb.get(i) is None for i in df.index)} faces without embedding",
          flush=True)
    n_before = len(df)
    if b.get("max_per_person_class"):
        df = cap_per_person(df, b["max_per_person_class"])
        print(f"cap {b['max_per_person_class']} faces per person and class: {n_before} -> {len(df)} faces", flush=True)

    # subject-independent split; a class with fewer people than folds just misses some folds
    folds = np.zeros(len(df), int)
    sgkf = StratifiedGroupKFold(n_splits=b["n_folds"], shuffle=True, random_state=b["seed"])
    for k, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(df)), df["label"], df["person"])):
        folds[test_idx] = k
    df["split"] = np.where(folds == 0, "test", np.where(folds == 1, "val", "train"))
    assert df.groupby("person")["split"].nunique().max() == 1

    if ds.exists():
        shutil.rmtree(ds)
    for s in SPLITS:
        for c in CLASSES:
            (ds / s / c).mkdir(parents=True, exist_ok=True)  # all 7 folders, even if empty (index_splits checks them)
    size = b["size"]
    for i, r in df.iterrows():
        g = cv2.resize(imread(files[i], cv2.IMREAD_GRAYSCALE), (size, size), interpolation=cv2.INTER_AREA)
        imwrite(ds / r["split"] / r["label"] / f"{i}.png", g)
    man = df.reset_index().rename(columns={"index": "id"})
    man.insert(0, "file", man["label"] + "/" + man["id"] + ".png")
    man["source_video"] = man["source"]
    man["source"] = "vtv"  # src.data.index_splits reads this column
    man["changed_in_review"] = man["suggested"].ne(man["label"]) & man["suggested"].ne("")
    # who set the label: a human (checked in the review app, data/vtv/human_review.csv, or moved by hand), Claude's
    # visual review of _unsure (data/vtv/claude_review.csv, only where it agreed with the ensemble), or the ensemble
    log = out / "claude_review.csv"
    by_claude = set(pd.read_csv(log)["id"]) if log.exists() else set()
    by_claude |= seen_kept()  # class faces Claude's pre-screen saw and kept (src.data.vtv_claude_suggest)
    by_human = set(human_checked(out))
    man["label_by"] = np.where(man["id"].isin(by_human), "human", np.where(
        man["id"].isin(by_claude), "claude", np.where(man["changed_in_review"], "human", "ensemble")))
    man[["file", "label", "source", "split", "person", "group", "source_video", "suggested", "changed_in_review", "label_by",
         "id"]].to_csv(ds / "manifest.csv", index=False)
    person_sheet(df, files, ds / "person_check.png")
    leak = leakage(df, emb, files, b["leak_sim"], b["leak_k"], ds / "leak_check.png")

    tab = pd.crosstab(df["label"], df["split"]).reindex(index=CLASSES, columns=list(SPLITS), fill_value=0)
    tab.loc["total"] = tab.sum()
    people = df.groupby("split")["person"].nunique().reindex(SPLITS, fill_value=0)
    lines = ["# VTV target dataset\n",
             f"{len(df)} faces, 48x48 grayscale, CK+ landmark alignment. {df['person'].nunique()} people "
             f"(SFace clusters, cosine >= {b['identity_sim']}); split by person "
             f"(StratifiedGroupKFold, {b['n_folds']} folds: test = fold 0, val = fold 1).\n",
             f"Labels checked by a human (review app log or moved by hand): {int((man['label_by'] == 'human').sum())} "
             f"of {len(man)} (" + ", ".join(f"{s} {int(((man['split'] == s) & (man['label_by'] == 'human')).sum())}/"
                                           f"{int((man['split'] == s).sum())}" for s in SPLITS) + "); "
             f"checked by Claude's visual review where it agreed with the ensemble (sorted out of _unsure, or seen and "
             f"kept in the pre-screen of the weak classes): "
             f"{int((man['label_by'] == 'claude').sum())}; ensemble suggestion as is: {int((man['label_by'] == 'ensemble').sum())}. "
             + ("**No human review yet: the labels are model suggestions (src.data.vtv_relabel: FER+ + HSEmotion) "
                "and Claude's check, not human-checked labels.**" if not (man["label_by"] == "human").any() else ""),
             "", "| Class | " + " | ".join(SPLITS) + " |", "|---|---|---|---|"]
    lines += [f"| {c} | " + " | ".join(str(tab.at[c, s]) for s in SPLITS) + " |" for c in tab.index]
    lines += [f"| people | " + " | ".join(str(people[s]) for s in SPLITS) + " |", ""]
    empty = [f"{c} in {s}" for c in CLASSES for s in SPLITS if tab.at[c, s] == 0]
    if empty:
        lines += [f"Empty: {', '.join(empty)}. Training needs every class in train (class weights) and ideally in val/test.", ""]
    top = df["person"].value_counts().head(5)
    lines += [f"Copies of a face from another source dropped: {len(dup)} ({dup_disagree} of them had another label "
              f"than their copy); sources sharing a person were split together ({df['component'].nunique()} linked "
              f"components from {df['group'].nunique()} groups).", ""]
    lines += ["Largest people: " + ", ".join(f"{p} {n}" for p, n in top.items()) + " (check `person_check.png`). "
              f"Top 5 people = {top.sum() / len(df):.0%} of the faces, top 10 = "
              f"{df['person'].value_counts().head(10).sum() / len(df):.0%}"
              + (f"; capped at {b['max_per_person_class']} per person and class ({n_before} -> {len(df)} faces)."
                 if b.get("max_per_person_class") else "; no cap per person (build: max_per_person_class)."), ""]
    src = df["group"].str.replace(r"^(pl\d+|frm\d+)_?.*$", r"\1", regex=True).str.replace(r"^pl\d+$", "playlist images", regex=True) \
        .str.replace(r"^frm\d+$", "frame images", regex=True).str.replace("^vid_", "", regex=True).str[:30]
    lines += ["| Source | Faces | People |", "|---|---|---|"]
    lines += [f"| {s} | {n} | {df.loc[src == s, 'person'].nunique()} |" for s, n in src.value_counts().items()] + [""]
    lines += [f"Identity leakage check: faces with >= {b['leak_k']} train faces at SFace cosine >= {b['leak_sim']} "
              f"(= same person in train): val {leak['val']:.1%}, test {leak['test']:.1%} (should be ~0). "
              "`leak_check.png`: the test people closest to train (test face | nearest train face).", ""]
    bal = (b.get("balance") or {}).get("min_per_class")
    if bal:  # step 2b: augmented train copies until every class has >= bal faces (src.data.vtv_balance)
        from src.data.vtv_balance import balance
        lines += ["", balance(ds, int(bal), int(b["seed"]))]
    (ds / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    zpath = Path(b["zip"])
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(ds.rglob("*")):
            if f.is_file() and f.suffix in (".png", ".csv") and f.name not in ("person_check.png", "leak_check.png"):
                z.write(f, "dataset/" + f.relative_to(ds).as_posix())
    print(f"dataset: {ds.resolve()}\nzip for Colab: {zpath.resolve()} ({zpath.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
