"""Claude's pre-screen of class folders in data/vtv/review/: suggestions for the human review page, no file moves.

The user's choice (2026-11-08): Claude looks through the weakest class folders first and flags the faces it is
fairly sure are wrong. The suggestions are shown in the review page (src.data.vtv_review_app) as pending changes;
the human accepts, edits or drops each one and saves the page, so every label that reaches the dataset is still a
human decision. Claude's eye is weaker than the ensemble on in-the-wild faces (RAF-DB: 58% vs 71%,
src.data.vtv_claude_review), so it only flags clear cases: not a face, occluded, blinking, turned, blurred, or a
clearly different expression (a smile in disgust/, a relaxed face in fear/). Unflagged faces carry no opinion.

  python -m src.data.vtv_claude_suggest make disgust fear surprise angry   -> grids/<folder>_###.png + grid_index.csv
  python -m src.data.vtv_claude_suggest grid disgust_001                   -> ids of one grid
  python -m src.data.vtv_claude_suggest apply                              -> data/vtv/claude_suggest.csv
  python -m src.data.vtv_claude_suggest move                               -> act on claude_suggest.csv without a human

``move`` (user, 2026-11-08: "lâu quá bạn làm đi cho nhanh": Claude applies its pre-screen itself): a reject flag moves the
face to ``_reject/``; an emotion flag disagrees with the ensemble, whose label put the face in the folder, so the label
is uncertain and the face goes to ``_unsure/`` (not to Claude's class: Claude alone is weaker than the ensemble, and
only labels where both agree are used, as in src.data.vtv_claude_review). Faces a human checked or moved since are
left alone. Logged in ``data/vtv/claude_suggest_moves.csv``; the faces Claude saw and kept get ``label_by = claude``
in src.data.vtv_build (Claude agreed with the ensemble). Never overwrites a file.
labels.txt (one line per reviewed grid, written while reviewing): ``disgust_001: 3n 7h 12x 30?`` (tile number +
code; a angry, d disgust, f fear, h happy, n neutral, s sad, u surprise, x reject, ? unsure; ``disgust_002:`` alone
= reviewed, nothing flagged). Grids follow the page's default order (least certain first).
"""
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.vtv_faces import CLASSES, imread
from src.data.vtv_pipeline import lock_holder
from src.data.vtv_review_app import FOLDERS, human_checked

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/vtv"
HERE = OUT / "claude_suggest"
GRIDS = HERE / "grids"
CODES = {"a": "angry", "d": "disgust", "f": "fear", "h": "happy", "n": "neutral", "s": "sad", "u": "surprise",
         "x": "_reject", "?": "_unsure"}
COLS, ROWS, T = 8, 6, 128


def grid(imgs, path: Path) -> None:
    tiles = []
    for k, g in enumerate(imgs):
        t = np.full((T + 16, T), 255, np.uint8)
        t[16:] = cv2.resize(g, (T, T), interpolation=cv2.INTER_AREA)
        cv2.putText(t, str(k + 1), (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
        tiles.append(np.pad(t, ((0, 4), (0, 4)), constant_values=255))
    while len(tiles) < COLS * ROWS:
        tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]))


def probs() -> dict:
    cols = ["id"] + [f"e_{c}" for c in CLASSES]
    d = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols) for f in sorted(OUT.glob("faces*.csv"))])
    d = d.dropna().drop_duplicates("id", keep="last")
    return dict(zip(d["id"], d[[f"e_{c}" for c in CLASSES]].to_numpy()))


def make(folders: list) -> None:
    GRIDS.mkdir(parents=True, exist_ok=True)
    p = probs()
    idx = HERE / "grid_index.csv"
    rows = pd.read_csv(idx).to_dict("records") if idx.exists() else []
    done = {r["folder"] for r in rows}
    for f in folders:
        assert f in CLASSES, f"{f}: class folders only"
        if f in done:
            print(f"{f}: grids already made (delete its rows in grid_index.csv to redo)")
            continue
        k = CLASSES.index(f)
        ids = sorted((q.stem for q in (OUT / "review" / f).glob("*.png")),
                     key=lambda i: (i not in p, p[i][k] if i in p else 0.0, i))  # as the page: least certain first
        per = COLS * ROWS
        for g in range(0, len(ids), per):
            name = f"{f}_{g // per + 1:03d}"
            chunk = ids[g:g + per]
            grid([imread(OUT / "review" / f / f"{i}.png", cv2.IMREAD_GRAYSCALE) for i in chunk], GRIDS / f"{name}.png")
            rows += [{"grid": name, "n": n + 1, "id": i, "folder": f} for n, i in enumerate(chunk)]
        print(f"{f}: {len(ids)} faces -> {-(-len(ids) // per)} grids")
    pd.DataFrame(rows).to_csv(idx, index=False)


def parse(path: Path) -> dict:
    """grid -> {tile number: code} from labels.txt (the last line of a grid wins)."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        name, _, rest = line.partition(":")
        marks = {}
        for tok in rest.split():
            m = re.fullmatch(r"(\d+)([adfhnsux?])", tok)
            assert m, f"bad token {tok!r} in line {line!r}"
            marks[int(m.group(1))] = m.group(2)
        out[name.strip()] = marks
    return out


def apply() -> None:
    idx = pd.read_csv(HERE / "grid_index.csv")
    labels = parse(HERE / "labels.txt")
    unknown = set(labels) - set(idx["grid"])
    assert not unknown, f"grids not in grid_index.csv: {sorted(unknown)}"
    rows = []
    for name, marks in labels.items():
        g = idx[idx["grid"] == name].set_index("n")
        for n, code in marks.items():
            assert n in g.index, f"{name}: no tile {n}"
            r = g.loc[n]
            if CODES[code] != r["folder"]:
                rows.append({"id": r["id"], "folder": r["folder"], "suggest": CODES[code], "grid": name, "n": n})
    d = pd.DataFrame(rows, columns=["id", "folder", "suggest", "grid", "n"])
    assert d["suggest"].isin(FOLDERS).all()
    d.to_csv(OUT / "claude_suggest.csv", index=False)
    seen = idx[idx["grid"].isin(labels)]
    print(f"{len(labels)} grids reviewed ({len(seen)} faces), {len(d)} suggestions -> {OUT / 'claude_suggest.csv'}")
    if len(d):
        print(pd.crosstab(d["folder"], d["suggest"]).to_string())


def seen_kept() -> set:
    """Ids Claude looked at in a screened class folder and did not flag (Claude agreed with the ensemble)."""
    idx, sug = HERE / "grid_index.csv", OUT / "claude_suggest.csv"
    if not idx.exists() or not (HERE / "labels.txt").exists():
        return set()
    done = set(parse(HERE / "labels.txt"))
    seen = pd.read_csv(idx)
    seen = set(seen.loc[seen["grid"].isin(done), "id"])
    return seen - (set(pd.read_csv(sug)["id"]) if sug.exists() else set())


def move() -> None:
    assert not lock_holder(OUT / "pipeline.lock"), "vtv_pipeline is running"
    d = pd.read_csv(OUT / "claude_suggest.csv")
    checked = human_checked(OUT)
    review = OUT / "review"
    now = datetime.now().isoformat(timespec="seconds")
    log, skipped = [], 0
    for r in d.itertuples():
        src = review / r.folder / f"{r.id}.png"
        if r.id in checked or not src.exists():
            skipped += 1  # checked or moved by a human since the screen
            continue
        to = "_reject" if r.suggest == "_reject" else "_unsure"
        dst = review / to / f"{r.id}.png"
        assert not dst.exists(), dst
        os.rename(src, dst)
        log.append({"time": now, "id": r.id, "from": r.folder, "to": to, "suggest": r.suggest})
    out = OUT / "claude_suggest_moves.csv"
    pd.DataFrame(log).to_csv(out, mode="a", header=not out.exists(), index=False)
    m = pd.DataFrame(log)
    print(f"moved {len(m)}, skipped {skipped} (checked or moved by a human) -> {out}")
    if len(m):
        print(pd.crosstab(m["from"], m["to"]).to_string())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "make":
        make(sys.argv[2:])
    elif cmd == "grid":
        idx = pd.read_csv(HERE / "grid_index.csv")
        print(idx[idx["grid"] == sys.argv[2]].to_string(index=False))
    elif cmd == "apply":
        apply()
    elif cmd == "move":
        move()
    else:
        print(__doc__)
