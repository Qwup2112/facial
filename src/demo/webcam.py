"""Real-time facial emotion demo on the webcam or a video file (CLAUDE.md section 7).

Faces: YuNet + the CK+ landmark alignment of src.data.vtv_faces, so each face is cropped exactly like the VTV
training faces (192x192 aligned crop -> 48x48 grayscale), then the run's val transform, resize_batch and
softmax. Probabilities are smoothed over time per tracked face (EMA, alpha = weight of the new frame).
Faces outside the range the model was trained on (turned, small, blurred, dark: the gates of
src.data.vtv_faces) are drawn grey with the failed gate: the model never saw such faces, so their label is a guess.

  python -m src.demo.webcam                          # webcam 0, default run (DEFAULT_RUNS: VTV + course mix first)
  python -m src.demo.webcam --pick                   # choose a video file in a dialog (nhan_dien_phim.bat)
  python -m src.demo.webcam --video film.mp4 --start 600 --run runs/transfer_vtv_claude/<run>
  python -m src.demo.webcam --video film.mp4 --out reports/demo/film.mp4 --csv reports/demo/film.csv --no-window

Keys: q / Esc quit, space pause, s screenshot (reports/demo/), left/right arrow (or a/d) -10 s / +10 s.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from src.data.transforms import build_transforms, resize_batch
from src.data.vtv_faces import Detector, align, ckplus_template, failed_gate, measure
from src.predict import load_run

ROOT = Path(__file__).resolve().parents[2]
FACES_CFG = ROOT / "configs/data/vtv_faces.yaml"
TEMPLATE = ROOT / "data/vtv/ckplus_template.npy"   # cache of the CK+ landmark template (48x48 pixels)
VIDEO_DIR = Path("D:/Download/New folder")
COLORS = {"angry": (60, 60, 230), "disgust": (60, 160, 60), "fear": (170, 80, 170), "happy": (0, 200, 255),
          "neutral": (200, 200, 200), "sad": (210, 140, 50), "surprise": (0, 140, 255)}   # BGR
GREY = (130, 130, 130)
GATE_TEXT = {"score": "unclear", "small": "small", "roll": "tilted", "outside": "cut off", "yaw": "turned",
             "pitch": "head up/down", "fit": "odd shape", "dark_bright": "dark/bright", "contrast": "flat",
             "blur": "blurred"}
# default model, first pattern with a match wins (its finished run with the best val macro-F1): VTV + course
# faces (src.data.mix_build) scored RAF-DB test ~80% vs 53% for VTV-only training (reports/transfer_mix_summary.md)
DEFAULT_RUNS = ("runs/transfer_mix/resnet18_s*/best.pt", "runs/transfer_vtv_*/resnet18_s*/best.pt")
LEFT, RIGHT = (2424832, 81, ord("a")), (2555904, 83, ord("d"))   # waitKeyEx codes (Windows, Linux) + a/d


def default_run() -> Path:
    """Finished run (curves.png) with the best val macro-F1 of the first matching pattern; else its newest run.

    Chosen on val, never on test, like every other model choice in the project."""
    for pattern in DEFAULT_RUNS:
        hits = [p.parent for p in ROOT.glob(pattern)]
        done = [d for d in hits if (d / "curves.png").exists()]
        if done:
            return max(done, key=lambda d: pd.read_csv(d / "log.csv")["val_macro_f1"].max())
        if hits:
            return max(hits, key=lambda d: (d / "best.pt").stat().st_mtime)
    raise SystemExit(f"no run with best.pt matching {DEFAULT_RUNS}: pass --run")


def template(det: Detector, ckplus_dir: Path) -> np.ndarray:
    if TEMPLATE.exists():
        return np.load(TEMPLATE)
    tpl = ckplus_template(ckplus_dir, det)
    np.save(TEMPLATE, tpl)
    return tpl


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return w * h / max(aw * ah + bw * bh - w * h, 1e-6)


class Tracks:
    """Match faces to the previous frame's faces by box overlap and smooth their probabilities (EMA)."""

    def __init__(self, alpha: float):
        self.alpha, self.items = alpha, []

    def update(self, boxes, probs):
        out, used = [], set()
        for b, p in zip(boxes, probs):
            best, j = 0.3, None
            for i, (tb, _) in enumerate(self.items):
                v = iou(b, tb)
                if i not in used and v > best:
                    best, j = v, i
            if j is not None:
                used.add(j)
                p = self.alpha * p + (1 - self.alpha) * self.items[j][1]
            out.append((b, p))
        self.items = out
        return [p for _, p in out]

    def reset(self):
        self.items = []


class Emotion:
    """Detect, align, check and classify every face of a frame."""

    def __init__(self, run: Path, device, det_side: int):
        fc = yaml.safe_load(FACES_CFG.read_text(encoding="utf-8"))
        self.det = Detector(ROOT / fc["models"]["yunet"], 0.6, det_side)
        self.tpl = template(self.det, ROOT / fc["ckplus_images"])
        self.gates, self.size = fc["gates"], fc["review"]["size"]
        self.model, self.cfg, self.classes = load_run(run, device)
        self.tf = build_transforms(self.cfg, train=False)
        self.device = device

    @torch.inference_mode()
    def __call__(self, frame: np.ndarray) -> list[dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []
        for score, box, pts in self.det(frame):
            crop, outside = align(gray, pts, self.tpl, self.size)
            g48 = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)   # as src.data.vtv_build
            g96 = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
            m = {"score": score, "outside": outside, **measure(pts, self.tpl, g48, g96)}
            faces.append({"box": box, "g48": g48, "gate": failed_gate(m, self.gates)})
        if faces:
            x = torch.stack([self.tf(Image.fromarray(f["g48"], "L")) for f in faces]).to(self.device)
            probs = self.model(resize_batch(x, self.cfg)).float().softmax(1).cpu().numpy()
            for f, p in zip(faces, probs):
                f["probs"] = p
        return faces


def text(img, s, org, scale, color, thick=1):
    """Text on a dark box (an outline drawn thicker would be wider: Hershey advances grow with thickness)."""
    (w, h), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + base + 1), (20, 20, 20), -1)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw(view, faces, k, classes, info: str):
    """Boxes and labels (frame coordinates scaled by k), probability bars of the largest clear face, status."""
    fs = view.shape[0] / 720
    for f in faces:
        x, y, w, h = (np.asarray(f["box"]) * k).astype(int)
        i = int(f["smooth"].argmax())
        c, conf = classes[i], f["smooth"][i]
        color = COLORS.get(c, (255, 255, 255)) if f["gate"] is None else GREY
        cv2.rectangle(view, (x, y), (x + w, y + h), color, max(2, round(2 * fs)))
        label = f"{c} {conf:.0%}" if f["gate"] is None else f"{c}? ({GATE_TEXT[f['gate']]})"
        text(view, label, (x, max(y - int(8 * fs), int(18 * fs))), 0.7 * fs, color, max(1, round(2 * fs)))
    clear = [f for f in faces if f["gate"] is None]
    if clear:
        f = max(clear, key=lambda f: f["box"][2] * f["box"][3])
        bw, bh, x0, y0 = int(170 * fs), int(20 * fs), int(12 * fs), int(14 * fs)
        cv2.rectangle(view, (0, 0), (x0 + int(155 * fs) + bw, y0 + len(classes) * (bh + 4) + 6), (25, 25, 25), -1)
        for r, (c, p) in enumerate(zip(classes, f["smooth"])):
            yy = y0 + r * (bh + 4)
            text(view, c, (x0, yy + bh - int(5 * fs)), 0.55 * fs, (235, 235, 235))
            bx = x0 + int(95 * fs)
            cv2.rectangle(view, (bx, yy), (bx + int(bw * p), yy + bh), COLORS.get(c, (255, 255, 255)), -1)
            text(view, f"{p:.0%}", (bx + bw + int(6 * fs), yy + bh - int(5 * fs)), 0.5 * fs, (235, 235, 235))
    text(view, info, (int(10 * fs), view.shape[0] - int(12 * fs)), 0.55 * fs, (235, 235, 235))


def pick_video() -> str:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Chon phim de nhan dien cam xuc", initialdir=str(VIDEO_DIR if VIDEO_DIR.exists() else Path.home()),
        filetypes=[("Video", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v"), ("All files", "*.*")])
    root.destroy()
    return path


def main():
    ap = argparse.ArgumentParser(description="Facial emotion demo on the webcam or a video file.")
    ap.add_argument("--video", help="video file (default: webcam --cam)")
    ap.add_argument("--pick", action="store_true", help="choose the video file in a dialog")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--run", help="run directory with best.pt (default: DEFAULT_RUNS, now runs/transfer_mix/resnet18_s*)")
    ap.add_argument("--start", type=float, default=0, help="start at this second of the video")
    ap.add_argument("--alpha", type=float, default=0.6, help="EMA weight of the new frame")
    ap.add_argument("--det-side", type=int, default=960, help="frames are downscaled to this for detection")
    ap.add_argument("--width", type=int, default=1280, help="display width")
    ap.add_argument("--out", help="also write the annotated video here (.mp4)")
    ap.add_argument("--csv", help="also write one row per face and frame here")
    ap.add_argument("--max-frames", type=int, help="stop after this many frames")
    ap.add_argument("--no-window", action="store_true", help="no display (with --out/--csv)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.pick:
        args.video = pick_video()
        if not args.video:
            print("no video chosen")
            return
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    run = Path(args.run) if args.run else default_run()
    emo = Emotion(run, device, args.det_side)
    classes = emo.classes
    print(f"model {emo.cfg['model']['name']} (seed {emo.cfg['seed']}) from {run} on {device}")

    cap = cv2.VideoCapture(args.video if args.video else args.cam)
    assert cap.isOpened(), f"cannot open {args.video or f'webcam {args.cam}'}"
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if args.video and args.start:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    name = Path(args.video).stem if args.video else f"webcam{args.cam}"
    win = "Facial emotion - q quit, space pause, s screenshot, <- -> 10 s"
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    tracks, writer, rows, counts = Tracks(args.alpha), None, [], dict.fromkeys(classes, 0)
    shots = ROOT / "reports/demo"
    paused, view, n, fps = False, None, 0, 0.0
    while True:
        if not paused or view is None:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if args.video else n / vfps
            faces = emo(frame)
            for f, p in zip(faces, tracks.update([f["box"] for f in faces], [f["probs"] for f in faces])):
                f["smooth"] = p
            k = min(1.0, args.width / frame.shape[1])
            view = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else frame.copy()
            dt = time.perf_counter() - t0
            fps = 1 / dt if fps == 0 else 0.9 * fps + 0.1 / max(dt, 1e-6)
            mm, ss = divmod(int(pos), 60)
            draw(view, faces, k, classes, f"{mm:02d}:{ss:02d}  {len(faces)} face(s)  {fps:.0f} FPS  "
                                          f"{emo.cfg['model']['name']}  grey = outside the training range")
            for j, f in enumerate(faces):
                if f["gate"] is None:
                    counts[classes[int(f["smooth"].argmax())]] += 1
                if args.csv:
                    rows.append({"time_s": round(pos, 2), "face": j, **dict(zip("xywh", np.round(f["box"]).astype(int))),
                                 "label": classes[int(f["smooth"].argmax())], "gate": f["gate"] or "",
                                 **{f"p_{c}": round(float(v), 4) for c, v in zip(classes, f["smooth"])}})
            if args.out:
                if writer is None:
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), vfps,
                                             (view.shape[1], view.shape[0]))
                writer.write(view)
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
        if args.no_window:
            continue
        cv2.imshow(win, view)
        wait = 30 if paused else max(1, int(1000 / vfps - (time.perf_counter() - t0) * 1000)) if args.video else 1
        key = cv2.waitKeyEx(wait)
        if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("s"):
            shots.mkdir(parents=True, exist_ok=True)
            p = shots / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.png"
            cv2.imencode(".png", view)[1].tofile(str(p))   # imwrite cannot write non-ASCII paths on Windows
            print(f"screenshot -> {p}")
        elif args.video and (key in LEFT or key in RIGHT):
            step = -10 if key in LEFT else 10
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, cap.get(cv2.CAP_PROP_POS_MSEC) + step * 1000))
            tracks.reset()
            paused = False
    cap.release()
    if writer is not None:
        writer.release()
        print(f"annotated video -> {args.out}")
    if args.csv and rows:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"{len(rows)} face rows -> {args.csv}")
    cv2.destroyAllWindows()
    total = sum(counts.values())
    if total:
        print(f"{n} frames; clear faces by label: " + ", ".join(f"{c} {v / total:.0%}" for c, v in counts.items()))


if __name__ == "__main__":
    main()
