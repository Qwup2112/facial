"""Inspect the dataset folder and write reports/data_report.md (CLAUDE.md 3.3).

Reads every image once (no torch; only hashes are kept in memory): images per class and per
source for each split, the imbalance ratio on train, image sizes and color modes, unreadable
files, and a leakage check across splits:
  - exact copies: MD5 of the grayscale pixels;
  - mirrored copies: a flip-invariant 64-bit dHash (the manifest.csv convention);
  - near copies: dHash Hamming distance <= --max-dist, comparing both orientations;
  - likely copies of one photo: near pairs with distance <= --copy-dist, or with a pixel
    correlation >= --ncc-min (best over shifts of up to 3 px and mirroring), since dHash alone
    also matches different faces with a similar layout. Two images whose source names different
    people (CK+ subject ids, kept in manifest.csv `origin`) are never a copy: lab photos of
    different people (same pose, light and background) look alike to both tests.
Hashes are computed from the files because the augmented rows of manifest.csv have none; for
the other rows they are checked against the manifest. Augmented copies are near their train
originals by design, so only matches between different splits count as leakage.
Duplicates are only listed (with an image grid): nothing is moved or deleted.
Also draws reports/figures/class_distribution.png.

Example:
    python -m src.data.inspect --raw dataset
"""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.data.index_splits import IMG_EXTS, SPLITS, read_class_names
from src.utils.config import load_config
from src.utils.logger import INK, INK_MUTED, plt, set_report_style

# Categorical slots of the validated palette, fixed per source (1-4, then 5 for ckplus); stacking order.
SOURCE_COLORS = {"local": "#2a78d6", "rafdb": "#eb6834", "affectnet": "#1baf7a", "ckplus": "#e87ba4",
                 "augmented": "#eda100"}
PERSON_ID = {"ckplus": r"(S\d{3})_\d{3}_\d+$"}   # source -> person id pattern in manifest `origin`


def dhash(gray: np.ndarray) -> int:
    """64-bit difference hash (9x8 area resize, left > right), as used to build manifest.csv."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return int("".join("1" if b else "0" for b in (small[:, 1:] > small[:, :-1]).flatten()), 2)


def read_image(path: Path) -> dict:
    """Size, color mode, pixel MD5 and dHash of the image and of its mirror (or the read error)."""
    try:
        with Image.open(path) as im:
            mode, (w, h) = im.mode, im.size
            gray = np.asarray(im.convert("L"))
    except Exception as e:  # corrupt or unreadable file
        return {"error": f"{type(e).__name__}: {e}"}
    h0, hf = dhash(gray), dhash(np.ascontiguousarray(gray[:, ::-1]))
    # hex strings: 64-bit ints would turn into float64 (and lose bits) in a column with gaps
    return {"width": w, "height": h, "mode": mode, "md5": hashlib.md5(gray.tobytes()).hexdigest(),
            "h": f"{h0:016x}", "hf": f"{hf:016x}", "dhash_px": f"{min(h0, hf):016x}", "error": None}


def scan(root: Path, classes: list[str], workers: int) -> pd.DataFrame:
    """One row per image file under root/<split>/<class>/."""
    files = [(split, name, f) for split in SPLITS for name in classes
             for f in sorted((root / split / name).iterdir()) if f.suffix.lower() in IMG_EXTS]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        info = list(pool.map(lambda t: read_image(t[2]), files, chunksize=64))
    df = pd.DataFrame([{"split": s, "label": c, "file": f"{c}/{f.name}"} for s, c, f in files])
    return pd.concat([df, pd.DataFrame(info)], axis=1)


def cross_split_equal(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Images whose `key` hash also occurs in another split (sorted by hash)."""
    n_splits = df.groupby(key)["split"].nunique()
    shared = n_splits[n_splits > 1].index
    return df[df[key].isin(shared)].sort_values([key, "split"])[[key, "split", "file", "label", "source"]]


def popcount64(x: np.ndarray) -> np.ndarray:
    """Number of set bits of each uint64 (SWAR bit counting)."""
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def to_u64(hex_hashes: pd.Series) -> np.ndarray:
    """Hex hash strings -> uint64 array."""
    return np.array([int(h, 16) for h in hex_hashes], dtype=np.uint64)


def near_pairs(query: pd.DataFrame, ref: pd.DataFrame, max_dist: int, chunk: int = 32) -> pd.DataFrame:
    """Pairs (query image, reference image) whose dHash differ in <= max_dist bits, query mirrored or not."""
    rh = to_u64(ref["h"])[None, :]
    qh, qf = to_u64(query["h"]), to_u64(query["hf"])
    found = []
    for s in range(0, len(qh), chunk):     # small chunks keep memory low
        d = np.minimum(popcount64(qh[s:s + chunk, None] ^ rh), popcount64(qf[s:s + chunk, None] ^ rh))
        qi, ri = np.nonzero(d <= max_dist)
        found.append((qi + s, ri, d[qi, ri]))
    qi, ri, dist = (np.concatenate(a) for a in zip(*found)) if found else ([], [], [])
    cols = ["split", "file", "label", "source"]
    a = query.iloc[qi][cols].reset_index(drop=True).add_suffix("_a")
    b = ref.iloc[ri][cols].reset_index(drop=True).add_suffix("_b")
    return pd.concat([a, b], axis=1).assign(distance=np.asarray(dist, dtype=int))


def max_ncc(a: np.ndarray, b: np.ndarray, r: int = 3) -> float:
    """Highest pixel correlation between the centre of `a` and `b` (or its mirror) shifted by up to r px.

    Copies of one photo (other crop, brightness or compression) score ~0.85-1; different faces
    with a similar layout, which dHash also matches, mostly score lower.
    """
    core = a[r:-r, r:-r].astype(np.float32)
    core = (core - core.mean()) / (core.std() + 1e-6)
    best = -1.0
    for img in (b, b[:, ::-1]):
        img = img.astype(np.float32)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                w = img[r + dy:img.shape[0] - r + dy, r + dx:img.shape[1] - r + dx]
                best = max(best, float((core * (w - w.mean()) / (w.std() + 1e-6)).mean()))
    return best


def person_ids(m: pd.DataFrame) -> dict:
    """(split, file) -> person id for sources that name the person; augmented copies inherit it."""
    ids = {}
    for source, pattern in PERSON_ID.items():
        rows = m[m["source"] == source]
        found = rows["origin"].str.extract(pattern)[0]
        ids.update({(s, f): f"{source}:{p}" for s, f, p in zip(rows["split"], rows["file"], found) if pd.notna(p)})
    aug = m[m["source"] == "augmented"]
    for f, o in zip(aug["file"], aug["origin"]):
        src = ("train", o.split("/", 1)[1])
        if src in ids:
            ids[("train", f)] = ids[src]
    return ids


def add_copy_flags(near: pd.DataFrame, root: Path, copy_dist: int, ncc_min: float, persons: dict) -> pd.DataFrame:
    """Add the pixel correlation of each pair and `likely_copy` (distance <= copy_dist or ncc >= ncc_min),
    except for pairs of two different known people (`different_person`)."""
    def gray(split, file):
        with Image.open(root / split / file) as im:
            return np.asarray(im.convert("L"))
    ncc = [max_ncc(gray(r.split_a, r.file_a), gray(r.split_b, r.file_b)) for r in near.itertuples()]
    pa = [persons.get((s, f)) for s, f in zip(near["split_a"], near["file_a"])]
    pb = [persons.get((s, f)) for s, f in zip(near["split_b"], near["file_b"])]
    other = np.array([a is not None and b is not None and a != b for a, b in zip(pa, pb)], dtype=bool)
    near = near.assign(ncc=np.round(ncc, 3), different_person=other)
    return near.assign(likely_copy=((near["distance"] <= copy_dist) | (near["ncc"] >= ncc_min)) & ~other)


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Small DataFrame -> GitHub markdown table."""
    df = df.reset_index() if index else df
    rows = [list(map(str, df.columns))] + [[f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)
                                            for v in r] for r in df.itertuples(index=False)]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def plot_distribution(df: pd.DataFrame, classes: list[str], out_png: Path) -> None:
    """Images per class for each split, stacked by source (one panel per split, own y scale)."""
    set_report_style()
    sources = [s for s in SOURCE_COLORS if s in set(df["source"])]
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(10, 3.4))
    x = np.arange(len(classes))
    for ax, split in zip(axes, SPLITS):
        counts = (df[df["split"] == split].groupby(["label", "source"]).size()
                  .unstack(fill_value=0).reindex(index=classes, columns=sources, fill_value=0))
        bottom = np.zeros(len(classes))
        for src in sources:
            ax.bar(x, counts[src], 0.7, bottom=bottom, color=SOURCE_COLORS[src], label=src,
                   edgecolor="white", linewidth=0.8)   # surface gap between stacked segments
            bottom += counts[src].to_numpy()
        per = f", {int(bottom[0]):,} per class" if len(set(bottom)) == 1 else ""   # one label, not seven
        ax.set_title(f"{split}: {int(bottom.sum()):,} images{per}", fontsize=10)
        ax.set_ylim(0, bottom.max() * 1.06)
        ax.set_axisbelow(True)                             # grid behind the bars
        ax.set_xticks(x, classes, rotation=45, ha="right")
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Images")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(sources), bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_pairs(pairs: pd.DataFrame, root: Path, out_png: Path, n: int = 24, cols: int = 4) -> None:
    """Side-by-side image pairs (closest first) so a person can judge whether they are copies."""
    set_report_style()
    pairs = pairs.sort_values("distance").head(n)
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, 2 * cols, figsize=(2 * cols * 1.1, rows * 1.45), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, r in enumerate(pairs.itertuples()):
        for j, (split, file) in enumerate([(r.split_a, r.file_a), (r.split_b, r.file_b)]):
            ax = axes[k // cols, 2 * (k % cols) + j]
            with Image.open(root / split / file) as im:
                ax.imshow(im.convert("L"), cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{split}\n{file.split('/')[0]}", fontsize=6.5, color=INK, pad=2)
        axes[k // cols, 2 * (k % cols)].text(1.02, -0.08, f"d={r.distance}", transform=axes[k // cols, 2 * (k % cols)].transAxes,
                                             ha="center", va="top", fontsize=6.5, color=INK_MUTED)
    fig.suptitle("Closest image pairs across splits (dHash Hamming distance d)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Inspect the dataset and write a markdown report.")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--raw", help="dataset folder with train/val/test (default: data.root)")
    ap.add_argument("--out", default="reports/data_report.md")
    ap.add_argument("--figures", default="reports/figures")
    ap.add_argument("--max-dist", type=int, default=4, help="dHash Hamming distance for near copies")
    ap.add_argument("--copy-dist", type=int, default=1, help="near pairs this close are copies outright")
    ap.add_argument("--ncc-min", type=float, default=0.84, help="pixel correlation that marks a copy")
    ap.add_argument("--workers", type=int, default=2, help="threads reading images (keep low while training)")
    args = ap.parse_args()

    data_cfg = load_config(args.config)["data"]
    root = Path(args.raw or data_cfg["root"])
    classes = read_class_names(root, data_cfg["num_classes"])
    df = scan(root, classes, args.workers)

    # manifest: source per image, and the hashes recorded when the dataset was built
    m = pd.read_csv(root / "manifest.csv", dtype=str)
    df = df.merge(m[["split", "file", "source", "md5", "dhash"]].rename(columns={"md5": "md5_manifest"}),
                  on=["split", "file"], how="left")
    df["source"] = df["source"].fillna("unknown")
    on_disk = set(zip(df["split"], df["file"]))
    missing_files = [f"{s}/{f}" for s, f in zip(m["split"], m["file"]) if (s, f) not in on_disk]
    unreadable = df[df["error"].notna()]
    ok = df[df["error"].isna()].reset_index(drop=True)
    hashed = ok[ok["md5_manifest"].notna()]
    md5_mismatch = hashed[hashed["md5"] != hashed["md5_manifest"]]
    dhash_mismatch = hashed[hashed["dhash_px"] != hashed["dhash"]]

    per_class = df.pivot_table(index="label", columns="split", values="file", aggfunc="size",
                               fill_value=0).reindex(columns=list(SPLITS))
    per_source = df.pivot_table(index="source", columns="split", values="file", aggfunc="size",
                                fill_value=0).reindex(columns=list(SPLITS))
    imbalance = per_class["train"].max() / per_class["train"].min()

    exact = {key: cross_split_equal(ok, key) for key in ("md5", "dhash_px")}
    by_split = {s: ok[ok["split"] == s].reset_index(drop=True) for s in SPLITS}
    near = pd.concat([near_pairs(by_split["val"], by_split["train"], args.max_dist),
                      near_pairs(by_split["test"], by_split["train"], args.max_dist),
                      near_pairs(by_split["test"], by_split["val"], args.max_dist)], ignore_index=True)
    near = near.sort_values(["distance", "split_a", "file_a"]).reset_index(drop=True)
    near = add_copy_flags(near, root, args.copy_dist, args.ncc_min, person_ids(m))
    copies = near[near["likely_copy"]]
    copy_imgs = copies[["split_a", "file_a"]].drop_duplicates()

    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    dist_png = figures / "class_distribution.png"
    plot_distribution(df, classes, dist_png)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs_png = figures / "near_duplicates.png"
    if len(near):
        plot_pairs(near, root, pairs_png)
        near.to_csv(out.with_name("near_duplicates.csv"), index=False)

    def rel(p: Path) -> str:
        """Path as written in the markdown file (relative to it when possible)."""
        return p.relative_to(out.parent).as_posix() if p.is_relative_to(out.parent) else p.as_posix()

    sizes = {d: ok[d].describe()[["min", "50%", "max"]].astype(int).tolist() for d in ("width", "height")}
    totals = per_class.sum()
    near_imgs = near[["split_a", "file_a"]].drop_duplicates()
    lines = [
        "# Data report", "",
        f"Generated by `python -m src.data.inspect --raw {root.as_posix()}` on {datetime.now():%Y-%m-%d %H:%M}.", "",
        "## Summary", "",
        f"- {len(df):,} images in {len(classes)} classes: " + ", ".join(classes) + ".",
        "- Images per split: " + ", ".join(f"{s} {totals[s]:,}" for s in SPLITS) + ".",
        f"- Imbalance ratio on train (largest / smallest class): {imbalance:.2f}.",
        f"- Image size (min / median / max): width {'/'.join(map(str, sizes['width']))}, "
        f"height {'/'.join(map(str, sizes['height']))} px.",
        "- Color modes: " + ", ".join(f"{k} {v:,}" for k, v in ok["mode"].value_counts().items()) + ".",
        f"- Unreadable files: {len(unreadable)}.",
        f"- Copies across splits: {exact['md5']['md5'].nunique()} exact (MD5), "
        f"{exact['dhash_px']['dhash_px'].nunique()} mirrored or exact (dHash), "
        f"{len(near):,} near pairs (dHash distance <= {args.max_dist}) involving "
        f"{len(near_imgs):,} val/test images, of which {len(copies):,} pairs ({len(copy_imgs):,} val/test "
        "images) are likely copies of one photo.",
        "", "## Images per class", "", md_table(per_class),
        "", "## Images per source", "",
        "`augmented` = offline-augmented copies, one per train original (train only).", "", md_table(per_source),
    ]
    for split in SPLITS:
        t = df[df["split"] == split].pivot_table(index="label", columns="source", values="file",
                                                 aggfunc="size", fill_value=0)
        lines += ["", f"### {split}: images per class and source", "", md_table(t)]
    lines += ["", "## Manifest consistency", "",
              f"- Files on disk without a manifest row: {int((df['source'] == 'unknown').sum())}.",
              f"- Manifest rows without a file: {len(missing_files)}.",
              f"- Pixel MD5 differs from the manifest: {len(md5_mismatch)} of {len(hashed):,} hashed rows.",
              f"- dHash differs from the manifest: {len(dhash_mismatch)} of {len(hashed):,} hashed rows.",
              "", "## Leakage check across splits", "",
              "Hashes are computed from the image files: MD5 of the 48x48 grayscale pixels (exact copies), "
              "a 64-bit dHash taken as the minimum over the image and its mirror (mirrored copies), and the "
              f"dHash Hamming distance, both orientations, for near copies (distance <= {args.max_dist}). "
              "Pairs are val-train, test-train and test-val; augmented train copies are included.", "",
              f"- Exact copies across splits (MD5): **{exact['md5']['md5'].nunique()}**.",
              f"- Mirrored or exact copies across splits (dHash): **{exact['dhash_px']['dhash_px'].nunique()}**.",
              f"- Near pairs across splits: **{len(near):,}**, involving {len(near_imgs):,} val/test images.",
              f"- Likely copies of one photo (distance <= {args.copy_dist}, or pixel correlation >= "
              f"{args.ncc_min} after shifts of up to 3 px and mirroring): **{len(copies):,}** pairs, "
              f"involving {len(copy_imgs):,} val/test images.",
              f"- Near pairs of two different known people (CK+ subject ids), never counted as copies: "
              f"{int(near['different_person'].sum()) if len(near) else 0:,}."]
    for key, name in (("md5", "Exact copies"), ("dhash_px", "Mirrored or exact copies")):
        if len(exact[key]):
            lines += ["", f"### {name}", "", md_table(exact[key].head(100), index=False)]
    if len(near):
        # pairs of unrelated faces share a label ~1/7 of the time (balanced classes), copies almost always
        by_d = near.groupby("distance").agg(
            pairs=("file_a", "size"),
            val_test_images=("file_a", lambda f: len(set(zip(near.loc[f.index, "split_a"], f)))),
            same_label=("label_a", lambda a: f"{(a == near.loc[a.index, 'label_b']).mean():.0%}"),
            likely_copies=("likely_copy", "sum"),
            copies_same_label=("likely_copy", lambda c: "-" if not c.any() else
                               f"{(near.loc[c.index[c], 'label_a'] == near.loc[c.index[c], 'label_b']).mean():.0%}"))
        lines += ["", "### Near pairs per distance", "",
                  f"Unrelated faces share a label about {1 / len(classes):.0%} of the time, so the same-label "
                  "share hints at how many pairs are real copies at each distance. Likely copies also need a "
                  f"pixel correlation >= {args.ncc_min} (or distance <= {args.copy_dist}).", "", md_table(by_d)]
        lines += ["", "### Near pairs (closest first, first 100)", "",
                  f"Full list: `{rel(out.with_name('near_duplicates.csv'))}`. A small distance is a candidate "
                  "copy, not proof: check the image grid below before acting. Nothing was moved or deleted.", "",
                  md_table(near.head(100), index=False), "",
                  f"![Closest pairs across splits]({rel(pairs_png)})"]
    if len(unreadable):
        lines += ["", "## Unreadable files", "", md_table(unreadable[["split", "file", "error"]], index=False)]
    lines += ["", "## Class distribution", "", f"![Images per class and source]({rel(dist_png)})", ""]
    out.write_text("\n".join(lines), encoding="utf-8")

    print(per_class.to_string())
    print(f"imbalance ratio (train): {imbalance:.2f} | unreadable: {len(unreadable)} | "
          f"modes: {ok['mode'].value_counts().to_dict()}")
    print(f"manifest: {len(missing_files)} rows without file, {len(md5_mismatch)} MD5 and "
          f"{len(dhash_mismatch)} dHash mismatches of {len(hashed):,}")
    print(f"across splits: {exact['md5']['md5'].nunique()} exact (MD5), "
          f"{exact['dhash_px']['dhash_px'].nunique()} mirrored/exact (dHash), "
          f"{len(near):,} near pairs (distance <= {args.max_dist}) involving {len(near_imgs):,} val/test images; "
          f"likely copies: {len(copies):,} pairs, {len(copy_imgs):,} val/test images")
    if len(near):
        print(near.groupby(["split_a", "split_b", "distance"])["likely_copy"].agg(["size", "sum"])
              .rename(columns={"size": "pairs", "sum": "likely_copies"}).to_string())
    print(f"-> {out}, {dist_png}" + (f", {pairs_png}" if len(near) else ""))


if __name__ == "__main__":
    main()
