"""Facial emotion app in a browser window: open any video or image, seek, jump to any time (CLAUDE.md section 7).

A local Flask server (127.0.0.1 only) runs the face pipeline of src.demo.webcam: YuNet, CK+ landmark
alignment, the quality gates of src.data.vtv_faces and the run's val transform. The page
(src/demo/web/index.html) plays the file in the browser. It sends each analysed frame as a JPEG at full
resolution, because the gates count pixels, and draws the returned boxes and probabilities. Files never
leave the PC. Closing the window stops the server.

  python -m src.demo.app                                   # app window (Chrome --app if found, else the browser)
  python -m src.demo.app --video "D:/film.mp4"             # also list this file under "Có sẵn trên máy"
  python -m src.demo.app --run runs/transfer_mix/<run> --port 7860 --no-browser
"""
import argparse
import base64
import logging
import os
import re
import socket
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from flask import Flask, abort, jsonify, request, send_file

from src.demo.webcam import ROOT, VIDEO_DIR, Emotion, Tracks, default_run

WEB = Path(__file__).with_name("web")
SHOTS = ROOT / "reports/demo"
RUN_PATTERNS = ("runs/transfer_mix/*_s*/best.pt", "runs/transfer_vtv_*/*_s*/best.pt")
DATASET_NAMES = {"transfer_mix": "VTV + dataset môn học", "transfer_vtv_claude": "VTV (Claude lọc)",
                 "transfer_vtv_suggested": "VTV (nhãn gợi ý)"}
MEDIA = {".mp4": "video", ".webm": "video", ".mkv": "video", ".mov": "video", ".m4v": "video",
         ".jpg": "image", ".jpeg": "image", ".png": "image", ".webp": "image", ".bmp": "image"}
CHROME = (Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
          Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
          Path(os.environ.get("LOCALAPPDATA", "")) / r"Google\Chrome\Application\chrome.exe")


def list_runs(extra: Path | None) -> list[dict]:
    """Finished runs the app can switch to (7 classes), best val macro-F1 first within each dataset."""
    runs = []
    for pattern in RUN_PATTERNS:
        for p in ROOT.glob(pattern):
            d = p.parent
            if (d / "curves.png").exists() and not d.name.startswith("dryrun"):
                runs.append(d)
    if extra is not None and extra.resolve() not in {d.resolve() for d in runs}:
        runs.append(extra)
    out = []
    for d in runs:
        m = re.match(r"(.+)_s(\d+)_", d.name + "_")
        log = d / "log.csv"
        f1 = float(pd.read_csv(log)["val_macro_f1"].max()) if log.exists() else None
        rel = d.resolve().relative_to(ROOT) if d.resolve().is_relative_to(ROOT) else d.resolve()
        out.append({"id": Path(rel).as_posix(), "model": m.group(1) if m else d.name, "seed": m.group(2) if m else "",
                    "dataset": DATASET_NAMES.get(d.parent.name, d.parent.name), "val_f1": f1})
    rank = {v: i for i, v in enumerate(DATASET_NAMES.values())}
    out.sort(key=lambda r: (rank.get(r["dataset"], 99), -(r["val_f1"] or 0)))
    return out


def list_media(extra: list[Path]) -> list[Path]:
    files = []
    for d in (VIDEO_DIR, VIDEO_DIR / "tro_doi_playlist"):
        if d.is_dir():
            files += sorted(p for p in d.iterdir() if p.suffix.lower() in MEDIA)
    return files + [p for p in extra if p.is_file() and p not in files]


def png64(img: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


def free_port(start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit("no free port")


class State:
    """The loaded model, the face tracks of the playing file, and when the page was last seen."""

    def __init__(self, device, det_side: int, alpha: float):
        self.lock = threading.Lock()
        self.device, self.det_side, self.alpha = device, det_side, alpha
        self.emo, self.run_id = None, None
        self.tracks = Tracks(alpha)
        self.seen, self.bye = time.time(), None

    def load(self, run_id: str):
        path = Path(run_id) if Path(run_id).is_absolute() else ROOT / run_id
        emo = Emotion(path, self.device, self.det_side)
        with self.lock:
            self.emo, self.run_id = emo, run_id
            self.tracks.reset()
        print(f"model {emo.cfg['model']['name']} (seed {emo.cfg['seed']}) from {run_id} on {self.device}")


def make_app(state: State, runs: list[dict], media: list[Path]) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 2**20

    @app.after_request
    def no_cache(resp):
        if request.path == "/" or request.path.startswith("/api/") and not request.path.startswith("/api/file/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/")
    def index():
        return send_file(WEB / "index.html")

    @app.get("/api/info")
    def info():
        state.seen = time.time()
        emo = state.emo
        gpu = torch.cuda.get_device_name(0) if state.device.type == "cuda" else "CPU"
        return jsonify(classes=emo.classes, run=state.run_id, runs=runs, device=gpu,
                       model=f"{emo.cfg['model']['name']} seed {emo.cfg['seed']}",
                       media=[{"id": i, "name": p.name, "kind": MEDIA[p.suffix.lower()],
                               "folder": p.parent.name, "mb": round(p.stat().st_size / 2**20)}
                              for i, p in enumerate(media)])

    @app.post("/api/model")
    def set_model():
        run_id = (request.get_json(silent=True) or {}).get("id")
        if run_id not in {r["id"] for r in runs}:
            abort(400)
        state.load(run_id)
        return info()

    @app.get("/api/file/<int:i>")
    def file(i: int):
        if not 0 <= i < len(media):
            abort(404)
        return send_file(media[i], conditional=True)   # range requests: the browser can seek

    @app.post("/api/infer")
    def infer():
        state.seen = time.time()
        frame = cv2.imdecode(np.frombuffer(request.get_data(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify(error="not an image"), 400
        track = request.args.get("track") == "1"
        alpha = min(1.0, max(0.05, float(request.args.get("alpha", state.alpha))))
        with state.lock:
            t0 = time.perf_counter()
            faces = state.emo(frame)
            probs = [f["probs"] for f in faces]
            if track:
                if request.args.get("reset") == "1":
                    state.tracks.reset()
                state.tracks.alpha = alpha
                smooth = state.tracks.update([f["box"] for f in faces], probs)
            else:
                smooth = probs
            ms = (time.perf_counter() - t0) * 1000
        return jsonify(w=frame.shape[1], h=frame.shape[0], ms=round(ms, 1), faces=[
            {"box": [round(float(v), 1) for v in f["box"]], "probs": np.round(f["probs"], 4).tolist(),
             "smooth": np.round(s, 4).tolist(), "gate": f["gate"], "crop": png64(f["g48"])}
            for f, s in zip(faces, smooth)])

    @app.post("/api/screenshot")
    def screenshot():
        name = re.sub(r"[^\w\-. ]", "_", request.args.get("name", "frame"))[:80].strip() or "frame"
        img = cv2.imdecode(np.frombuffer(request.get_data(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            abort(400)
        SHOTS.mkdir(parents=True, exist_ok=True)
        p = SHOTS / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.png"
        cv2.imencode(".png", img)[1].tofile(str(p))   # imwrite cannot write non-ASCII paths on Windows
        return jsonify(path=str(p))

    @app.get("/api/ping")
    def ping():
        state.seen, state.bye = time.time(), None
        return jsonify(ok=True)

    @app.post("/api/bye")
    def bye():
        state.bye = time.time()
        return ("", 204)

    return app


def watchdog(state: State, idle: float):
    """Stop the server when the page is closed (bye, no reload within 5 s) or silent for `idle` seconds."""
    while True:
        time.sleep(1)
        now = time.time()
        if state.bye and now - state.bye > 5 and state.seen < state.bye or now - state.seen > idle:
            print("app window closed: server stops")
            os._exit(0)


def open_window(url: str):
    chrome = next((p for p in CHROME if p.is_file()), None)
    if chrome:
        subprocess.Popen([str(chrome), f"--app={url}", "--window-size=1560,940"])
    else:
        webbrowser.open(url)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", help="run directory with best.pt (default: as src.demo.webcam, best val run)")
    ap.add_argument("--video", nargs="*", default=[], help="extra files to list in the app")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--det-side", type=int, default=960, help="frames are downscaled to this for detection")
    ap.add_argument("--alpha", type=float, default=0.6, help="EMA weight of the new frame")
    ap.add_argument("--no-browser", action="store_true", help="do not open a window (and never stop on idle)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    run = Path(args.run) if args.run else default_run()
    runs = list_runs(run)
    run_id = next(r["id"] for r in runs if (ROOT / r["id"]).resolve() == run.resolve())
    state = State(device, args.det_side, args.alpha)
    state.load(run_id)
    media = list_media([Path(v) for v in args.video])
    app = make_app(state, runs, media)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)   # no line per frame

    port = free_port(args.port)
    url = f"http://127.0.0.1:{port}/"
    print(f"app: {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Thread(target=watchdog, args=(state, 180.0), daemon=True).start()
        threading.Timer(0.8, open_window, args=(url,)).start()
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
