"""VTV target dataset, step 1e: human review of the review folders in the browser.

A local page (http://127.0.0.1:8765, reachable from this PC only) shows the faces of one folder of
``<out_dir>/review/`` as a grid that fits the screen, the least certain faces first. Number keys set
the class of the selected face; Enter saves the page. Faces with a new class are moved to that
folder, exactly like a hand move in File Explorer. Every face on the page, moved or kept, is
appended to ``<out_dir>/human_review.csv`` as checked by a human. Ctrl+Z undoes the last saved
page, also after a restart.

Claude's pre-screen (src.data.vtv_claude_suggest, ``<out_dir>/claude_suggest.csv``) is shown as pending changes
marked "Claude": the reviewer accepts, edits or drops them before saving; the log keeps the suggestion in ``hint``.

Why a log: a face the reviewer looked at and kept looks the same on disk as a face nobody looked
at. With the log, src.data.vtv_build sets ``label_by = human`` for every checked face and reports
how much of each split is human-checked, and src.data.vtv_relabel never moves a checked face.

Safety: the app only moves PNGs between the folders of review/. It never deletes, overwrites or
renames a file, and never touches faces*.csv. It refuses to move anything while
src.data.vtv_pipeline holds its lock. The pipeline in turn refuses to start while this app runs
(its FOREIGN pattern matches ``src.data.vtv_review_app``): close the app (Ctrl+C) before a rebuild.

Examples:
    python -m src.data.vtv_review_app                      # opens the browser
    python -m src.data.vtv_review_app --port 8766 --no-browser
"""
import argparse
import csv
import json
import os
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
import pandas as pd

from src.data.vtv_faces import CLASSES, imread
from src.data.vtv_pipeline import lock_holder
from src.utils.config import load_config

FOLDERS = CLASSES + ["_unsure", "_reject"]
RARE = ["angry", "disgust", "fear", "surprise"]  # weakest classes of the vtv_suggested transfer
LOG = "human_review.csv"
LOG_COLS = ["time", "batch", "action", "id", "from", "to", "hint"]  # hint: Claude's suggestion shown, if any
PAGE = Path(__file__).with_name("vtv_review_app.html")


class Busy(Exception):
    """A request that cannot be done now (pipeline running, file moved outside the app, bad input)."""


def read_log(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def live_batches(rows: list) -> dict:
    """Batch -> its decision rows, in saving order, without the undone batches."""
    out, undone = {}, set()
    for r in rows:
        if r["action"] == "undo":
            undone.add(r["batch"])
        else:
            out.setdefault(r["batch"], []).append(r)
    return {b: rs for b, rs in out.items() if b not in undone}


def human_checked(out: Path) -> dict:
    """Face id -> folder a human put it in or kept it in, from ``<out>/human_review.csv`` (undone pages left out)."""
    res = {}
    for rows in live_batches(read_log(out / LOG)).values():
        for r in rows:
            res[r["id"]] = r["to"]
    return res


class Review:
    def __init__(self, out: Path):
        self.out, self.review, self.log = out, out / "review", out / LOG
        self.lock = out / "pipeline.lock"
        self.mutex = threading.Lock()
        self.scan()
        cols = {"id"} | {f"e_{c}" for c in CLASSES}
        self.probs = {}
        for f in sorted(out.glob("faces*.csv")):
            d = pd.read_csv(f, usecols=lambda c: c in cols)
            if len(d) and all(f"e_{c}" in d for c in CLASSES):
                p = d[[f"e_{c}" for c in CLASSES]].to_numpy(float).round(3)
                self.probs.update({i: (None if np.isnan(r).any() else r.tolist()) for i, r in zip(d["id"], p)})
        cl = out / "claude_review.csv"
        d = pd.read_csv(cl) if cl.exists() else pd.DataFrame(columns=["id", "to"])
        self.claude = dict(zip(d["id"], d["to"]))  # faces Claude sorted out of _unsure (shown as a hint)
        self.checked = human_checked(out)
        self.migrate()
        self.hint, self.hint_mtime = {}, None
        self.load_hints()

    def migrate(self) -> None:
        """Give an older log the columns it lacks (rows kept, new cells empty); the old file is copied first."""
        if not self.log.exists():
            return
        with open(self.log, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            head = list(rows[0].keys()) if rows else LOG_COLS
        if all(c in head for c in LOG_COLS):
            return
        bak = self.log.with_name(f"{self.log.stem}.bak-{datetime.now():%Y%m%d-%H%M%S}.csv")
        bak.write_bytes(self.log.read_bytes())
        with open(self.log, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, LOG_COLS + [c for c in head if c not in LOG_COLS])
            w.writeheader()
            w.writerows(rows)
        print(f"log columns added (old file kept as {bak.name})")

    def load_hints(self) -> None:
        """Claude's pre-screen (claude_suggest.csv), re-read whenever the file changes."""
        cs = self.out / "claude_suggest.csv"
        m = cs.stat().st_mtime if cs.exists() else None
        if m == self.hint_mtime:
            return
        d = pd.read_csv(cs) if cs.exists() else pd.DataFrame(columns=["id", "folder", "suggest"])
        self.hint = {i: (f, s) for i, f, s in zip(d["id"], d["folder"], d["suggest"])}
        self.hint_mtime = m

    def scan(self) -> None:
        self.loc = {p.stem: f for f in FOLDERS for p in (self.review / f).glob("*.png")}

    def counts(self) -> dict:
        c = {f: [0, 0] for f in FOLDERS}
        for i, f in self.loc.items():
            c[f][0] += 1
            c[f][1] += i in self.checked
        return {"counts": c, "checked": len(self.checked), "faces": len(self.loc)}

    def listing(self, folder: str, order: str) -> list:
        if folder not in FOLDERS:
            raise Busy(f"unknown folder {folder}")
        self.load_hints()
        k = CLASSES.index(folder) if folder in CLASSES else None
        rare = [CLASSES.index(c) for c in RARE]

        def hint(i):
            h = self.hint.get(i)
            return h[1] if h and h[0] == folder else None

        def key(i):
            if order == "claude":  # Claude's flags first, then least certain first
                return (hint(i) is None,) + key_p(i, "unsure")
            return key_p(i, order)

        def key_p(i, order):
            p = self.probs.get(i)
            if order == "scene" or p is None:
                return (order != "scene" and p is None, 0.0, i)
            if order == "rare":
                v = -max(p[j] for j in rare)
            else:  # probability of this folder's class (top probability in _unsure / _reject)
                v = p[k] if k is not None else max(p)
                v = -v if order == "sure" else v
            return (False, v, i)

        ids = sorted((i for i, f in self.loc.items() if f == folder), key=key)
        return [{"id": i, "p": self.probs.get(i), "checked": i in self.checked, "claude": self.claude.get(i),
                 "hint": hint(i)} for i in ids]

    def image(self, fid: str, small: bool) -> bytes:
        f = self.loc.get(fid)
        if f is None:
            raise KeyError(fid)
        path = self.review / f / f"{fid}.png"
        if not small:
            return path.read_bytes()
        g = cv2.resize(imread(path, cv2.IMREAD_GRAYSCALE), (48, 48), interpolation=cv2.INTER_AREA)  # as vtv_build
        return cv2.imencode(".png", g)[1].tobytes()

    def _pipeline_idle(self) -> None:
        holder = lock_holder(self.lock)
        if holder:
            raise Busy(f"vtv_pipeline đang chạy (pid {holder['pid']}): đợi nó xong rồi lưu lại")

    def _move(self, fid: str, src: str, dst: str) -> None:
        a, b = self.review / src / f"{fid}.png", self.review / dst / f"{fid}.png"
        if b.exists():
            raise Busy(f"{b} đã có: không ghi đè")
        b.parent.mkdir(exist_ok=True)
        os.rename(a, b)  # same folder tree; never overwrites on Windows
        self.loc[fid] = dst

    def _fields(self) -> list:
        """Columns of the existing log (an older log keeps its header), else LOG_COLS."""
        if self.log.exists():
            with open(self.log, newline="", encoding="utf-8") as f:
                return next(csv.reader(f), LOG_COLS)
        return LOG_COLS

    def _append(self, rows: list) -> None:
        new = not self.log.exists()
        with open(self.log, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, self._fields() if not new else LOG_COLS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(rows)
            f.flush()
            os.fsync(f.fileno())

    def save(self, decisions: list) -> dict:
        with self.mutex:
            self._pipeline_idle()
            if not decisions:
                raise Busy("trang trống")
            for d in decisions:
                if d.get("to") not in FOLDERS or d.get("id") not in self.loc:
                    raise Busy(f"quyết định không hợp lệ: {d}")
                if not (self.review / self.loc[d["id"]] / f"{d['id']}.png").exists():
                    self.scan()
                    raise Busy(f"{d['id']} đã bị chuyển ngoài trang duyệt: tải lại thư mục")
            batch = str(1 + max((int(r["batch"]) for r in read_log(self.log)), default=0))
            now = datetime.now().isoformat(timespec="seconds")
            moved = 0
            new = not self.log.exists()
            fields = LOG_COLS if new else self._fields()
            with open(self.log, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fields, extrasaction="ignore")
                if new:
                    w.writeheader()
                for d in decisions:  # one log row per face right after its move, so a crash leaves the log true
                    fid, src, dst = d["id"], self.loc[d["id"]], d["to"]
                    if dst != src:
                        self._move(fid, src, dst)
                        moved += 1
                    h = self.hint.get(fid)
                    w.writerow({"time": now, "batch": batch, "action": "move" if dst != src else "keep",
                                "id": fid, "from": src, "to": dst, "hint": h[1] if h and h[0] == src else ""})
                    f.flush()
                    self.checked[fid] = dst
                os.fsync(f.fileno())
            return {"batch": batch, "saved": len(decisions), "moved": moved, **self.counts()}

    def undo(self) -> dict:
        with self.mutex:
            self._pipeline_idle()
            live = live_batches(read_log(self.log))
            if not live:
                raise Busy("không còn trang nào để hoàn tác")
            b = max(live, key=int)
            back, skipped = 0, 0
            for r in reversed(live[b]):
                if r["from"] == r["to"]:
                    continue
                if self.loc.get(r["id"]) == r["to"] and not (self.review / r["from"] / f"{r['id']}.png").exists():
                    self._move(r["id"], r["to"], r["from"])
                    back += 1
                else:
                    skipped += 1  # moved again since (by hand or a later page): leave it
            self._append([{"time": datetime.now().isoformat(timespec="seconds"), "batch": b, "action": "undo",
                           "id": "", "from": "", "to": ""}])
            self.checked = human_checked(self.out)
            return {"batch": b, "undone": len(live[b]), "moved_back": back, "skipped": skipped, **self.counts()}


def handler(rv: Review, page: Path):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code: int, body: bytes, ctype: str, cache: bool = False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def json(self, obj, code: int = 200):
            self.send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/":
                    return self.send(200, page.read_bytes(), "text/html; charset=utf-8")  # re-read: edits show on reload
                if u.path == "/api/state":
                    return self.json(rv.counts())
                if u.path == "/api/list":
                    return self.json(rv.listing(q.get("folder", [""])[0], q.get("order", ["unsure"])[0]))
                if u.path.startswith("/img/") and u.path.endswith(".png"):
                    fid = unquote(u.path[5:-4])
                    return self.send(200, rv.image(fid, q.get("s") == ["48"]), "image/png", cache=True)
            except KeyError:
                pass
            except Busy as e:
                return self.json({"error": str(e)}, 409)
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.headers.get("X-Review") != "1":  # a custom header: other web pages cannot send it here
                return self.send(403, b"forbidden", "text/plain")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/api/save":
                    return self.json(rv.save(body.get("decisions", [])))
                if self.path == "/api/undo":
                    return self.json(rv.undo())
            except (Busy, OSError, ValueError) as e:
                return self.json({"error": str(e)}, 409)
            self.send(404, b"not found", "text/plain")

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/data/vtv_faces.yaml")
    ap.add_argument("--out", help="override out_dir of the config (tests)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    out = Path(args.out or load_config(args.config)["out_dir"])
    assert (out / "review").is_dir(), f"no {out / 'review'}"
    rv = Review(out)
    c = rv.counts()
    print(f"{c['faces']} faces in {out / 'review'}, {c['checked']} checked by a human so far (log {rv.log})")
    print("  " + ", ".join(f"{f} {n}/{t}" for f, (t, n) in c["counts"].items()))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(rv, PAGE))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"review page: {url}  (Ctrl+C to stop; close it before running src.data.vtv_pipeline)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
