"""VTV target dataset, step 1: find clear frontal faces, align them like CK+ and sort them for review.

1. Frames: one every ``video_step_s`` seconds of each video, plus the extracted frame images.
2. YuNet (``cv2.FaceDetectorYN``) finds each face and 5 landmarks (eyes, nose tip, mouth corners).
3. The face is aligned to the mean CK+ landmark layout (rotation, scale and shift that best map the
   5 points onto the template), so eyes, nose and mouth sit where they sit in the CK+ 48x48 crops,
   and converted to grayscale.
4. Gates (``configs/data/vtv_faces.yaml``): detector score, eye distance, head tilt, face cut by the
   border, head turn (yaw), up/down (pitch), landmark fit, brightness, contrast, sharpness (Laplacian
   at 48x48 and blur effect at 96x96), then YuNet again on the aligned crop (a real face is found
   again with its landmarks on the template). Head pose and image gates come from the range of CK+.
5. FER+ (ONNX model zoo; none of our trained models) suggests a label.
6. Repeats are dropped per video: within 60 s, NCC >= 0.85, or NCC >= 0.80 and the same suggested
   label (same person and expression in one scene); NCC >= 0.90 with any recent kept face.
7. Faces go to ``review/<suggested class>/`` (``review/_unsure/`` when unsure). A human checks and
   moves the files; step 2 builds the 48x48 dataset from the checked folders.

Writes ``<out>/review/`` (192x192 grayscale PNG + HUONG_DAN.md), ``<out>/faces.csv``,
``<out>/rejected_examples/<gate>/`` and ``<out>/summary.md``. Never overwrites an existing review/
unless ``--overwrite``; ``--append --only <name>`` adds new sources (e.g. new videos in the
source folder) to it, with their own ``faces_<name>.csv`` and ``summary_<name>.md``.

Example:
    python -m src.data.vtv_faces --limit 400 --out data/vtv_test     # quick check of the gates
    python -m src.data.vtv_faces
    python -m src.data.vtv_faces --append --only tao_quan            # a video added later
"""
import argparse
import random
import re
import shutil
import threading
import time
import unicodedata
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.utils.config import load_config

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
FERPLUS = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]  # model output order
GATES = ["score", "small", "roll", "outside", "yaw", "pitch", "fit", "dark_bright", "contrast", "blur",
         "recheck", "duplicate"]
GUIDE = Path(__file__).with_name("vtv_review_guide.md")


def imread(path: Path, flags=cv2.IMREAD_COLOR):
    """cv2.imread that also works with non-ASCII Windows paths."""
    return cv2.imdecode(np.fromfile(str(path), np.uint8), flags)


def imwrite(path: Path, img) -> None:
    """cv2.imwrite that also works with non-ASCII Windows paths."""
    ok, buf = cv2.imencode(path.suffix, img)
    assert ok, path
    buf.tofile(str(path))


def slug(text: str, n: int = 32) -> str:
    """ASCII lowercase name for a file stem (Vietnamese accents removed)."""
    text = unicodedata.normalize("NFKD", text.replace("đ", "d").replace("Đ", "D"))
    text = re.sub(r"[^a-z0-9]+", "_", text.encode("ascii", "ignore").decode().lower()).strip("_")
    return text[:n].rstrip("_")


def similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (Umeyama) mapping points ``src`` onto ``dst``; 2x3 matrix."""
    ms, md = src.mean(0), dst.mean(0)
    s, d = src - ms, dst - md
    u, sv, vt = np.linalg.svd(d.T @ s / len(src))
    e = np.eye(2)
    if np.linalg.det(u @ vt) < 0:
        e[1, 1] = -1
    r = u @ e @ vt
    c = (sv * np.diag(e)).sum() / ((s ** 2).sum() / len(src))
    return np.hstack([c * r, (md - c * r @ ms)[:, None]])


class Detector:
    """YuNet face detector; returns (score, box, 5x2 landmarks) in full-frame pixels."""

    def __init__(self, model: Path, min_score: float, max_side: int):
        self.net = cv2.FaceDetectorYN.create(str(model), "", (320, 320), min_score, 0.3, 50)
        self.max_side = max_side

    def __call__(self, img: np.ndarray):
        h, w = img.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
        self.net.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.net.detect(small)
        if faces is None:
            return []
        return [(float(f[14]), f[:4] / s, f[4:14].reshape(5, 2) / s) for f in faces]


def ckplus_template(ckplus_dir: Path, det: Detector) -> np.ndarray:
    """Mean YuNet landmarks of the CK+ 48x48 crops (5x2, in 48x48 pixels)."""
    pts = []
    for f in sorted(ckplus_dir.glob("*/*.png")):
        g = imread(f, cv2.IMREAD_GRAYSCALE)
        faces = det(cv2.cvtColor(cv2.resize(g, (192, 192), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR))
        if faces:
            pts.append(max(faces, key=lambda x: x[1][2] * x[1][3])[2] / 4.0)
    assert len(pts) > 100, f"CK+ template: only {len(pts)} faces found in {ckplus_dir}"
    return np.mean(pts, 0)


def align(gray: np.ndarray, pts: np.ndarray, tpl48: np.ndarray, size: int):
    """Crop ``size`` x ``size`` with the landmarks mapped onto the CK+ template; also its in-frame mask."""
    t = similarity(pts, tpl48 * size / 48)
    scale = float(np.hypot(t[0, 0], t[0, 1]))
    if scale < 1:  # shrink the frame first (area filter) so the warp does not alias
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        t = similarity(pts * scale, tpl48 * size / 48)
    crop = cv2.warpAffine(gray, t, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    inside = cv2.warpAffine(np.ones(gray.shape, np.uint8), t, (size, size), flags=cv2.INTER_NEAREST, borderValue=0)
    return crop, 1.0 - float(inside.mean())


def blur_effect(img: np.ndarray, h: int = 9) -> float:
    """No-reference blur metric (Crete-Roffet et al., 2007): 0 = sharp, 1 = fully blurred.

    Re-blurs the image and measures how much of its pixel-to-pixel variation is lost: a sharp
    image loses a lot, an already blurred one almost nothing. Independent of brightness and contrast.
    """
    f = img.astype(np.float32)
    out = []
    for axis, ksize in ((0, (1, h)), (1, (h, 1))):
        b = cv2.blur(f, ksize)
        d, db = np.abs(np.diff(f, axis=axis)), np.abs(np.diff(b, axis=axis))
        m1 = d[2:-2, 2:-2].sum()
        m2 = np.maximum(0, d - db)[2:-2, 2:-2].sum()
        out.append((m1 - m2) / (m1 + 1e-6))
    return float(max(out))


def measure(pts: np.ndarray, tpl48: np.ndarray, g48: np.ndarray, g96: np.ndarray) -> dict:
    """Head pose and image measures of one face (landmarks in frame pixels, aligned 48x48 and 96x96 crops)."""
    t = similarity(pts, tpl48)
    a = pts @ t[:, :2].T + t[:, 2]  # landmarks in the aligned 48x48 crop
    eyes, mouth = (a[0] + a[1]) / 2, (a[3] + a[4]) / 2
    ed = float(np.linalg.norm(a[1] - a[0]))
    centre = g48[8:44, 8:40].astype(np.float32)  # face area without the background corners
    return {
        "eye_dist": float(np.linalg.norm(pts[1] - pts[0])),
        "roll": float(np.degrees(np.arctan2(pts[1, 1] - pts[0, 1], pts[1, 0] - pts[0, 0]))),
        "yaw": float((a[2, 0] - eyes[0]) / ed),
        "pitch": float((a[2, 1] - eyes[1]) / max(mouth[1] - eyes[1], 1e-6)),
        "fit": float(np.sqrt(((a - tpl48) ** 2).sum(1).mean()) / ed),
        "brightness": float(centre.mean()),
        "contrast": float(centre.std()),
        "sharpness": float(cv2.Laplacian(g48, cv2.CV_64F).var()),
        "blur": blur_effect(g96),  # at 2x the dataset size: catches soft (upscaled, out of focus) faces
    }


def recheck(det: "Detector", crop: np.ndarray, tpl48: np.ndarray) -> tuple[float, float]:
    """Detect again on the aligned crop: (best score, RMS landmark error vs. the template / eye distance).

    A real face is found again, centred, with its landmarks on the CK+ template; hands, objects
    and half-hidden faces that slipped through the first detection mostly are not.
    """
    size = crop.shape[0]
    faces = det(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))
    if not faces:
        return 0.0, 9.0
    tpl = tpl48 * size / 48
    score, _, pts = max(faces, key=lambda x: x[0])
    err = np.sqrt(((pts - tpl) ** 2).sum(1).mean()) / np.linalg.norm(tpl[1] - tpl[0])
    return score, float(err)


def failed_gate(m: dict, g: dict):
    """Name of the first gate the face fails, or None (order = GATES)."""
    checks = [
        ("score", m["score"] >= g["score"]),
        ("small", m["eye_dist"] >= g["min_eye_dist"]),
        ("roll", abs(m["roll"]) <= g["max_roll"]),
        ("outside", m["outside"] <= g["max_outside"]),
        ("yaw", abs(m["yaw"]) <= g["max_yaw"]),
        ("pitch", g["pitch"][0] <= m["pitch"] <= g["pitch"][1]),
        ("fit", m["fit"] <= g["max_fit"]),
        ("dark_bright", g["brightness"][0] <= m["brightness"] <= g["brightness"][1]),
        ("contrast", m["contrast"] >= g["min_contrast"]),
        ("blur", m["sharpness"] >= g["min_sharpness"] and m["blur"] <= g["max_blur"]),
    ]
    return next((name for name, ok in checks if not ok), None)


class FerPlus:
    """FER+ emotion model (ONNX model zoo): 8-class probabilities, averaged with the mirrored face."""

    def __init__(self, model: Path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # called from several worker threads at once
        self.sess = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name

    def __call__(self, face: np.ndarray) -> np.ndarray:
        x = cv2.resize(face, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
        x = np.stack([x, x[:, ::-1]])[:, None]  # raw 0-255 gray, as the model expects
        z = np.concatenate([self.sess.run(None, {self.name: x[i:i + 1]})[0] for i in range(2)])
        p = np.exp(z - z.max(1, keepdims=True))
        return (p / p.sum(1, keepdims=True)).mean(0)


def ncc_vector(g48: np.ndarray) -> np.ndarray:
    """Blurred 24x24 zero-mean unit vector: its dot product is the NCC, robust to 1-2 px of alignment jitter."""
    x = cv2.resize(cv2.GaussianBlur(g48, (0, 0), 1.0), (24, 24), interpolation=cv2.INTER_AREA)
    v = x.astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


class Dedup:
    """Keep only faces that add something new, per group (video).

    A face is a duplicate when, compared with the faces kept from the same group in the last
    ``recent_s`` seconds of video (``recent_frames`` frames for frame images), it has NCC >=
    ``recent_ncc_any`` (the suggested label of a soft face
    can flip between near-identical frames), or NCC >= ``recent_ncc`` and the same suggested label
    (same person, same expression, same scene); or NCC >= ``group_ncc`` with one of the last
    ``window`` kept faces (nearly the same image, e.g. the same shot after a cut back).
    """

    def __init__(self, cfg: dict):
        self.c = cfg
        self.kept = defaultdict(list)   # group -> [(time or frame position, vector, label)]
        self.pos = Counter()            # group -> frames seen

    def new_frame(self, group: str) -> None:
        self.pos[group] += 1

    def duplicate(self, group: str, vec: np.ndarray, label: str, t: float | None) -> bool:
        """True if the face adds nothing new; otherwise it is recorded as kept. ``t``: video time (s) or None."""
        now, span = (t, self.c["recent_s"]) if t is not None else (self.pos[group], self.c["recent_frames"])
        for when, v, lab in self.kept[group]:
            s = float(vec @ v)
            if s >= self.c["group_ncc"]:
                return True
            if now - when <= span:
                if s >= self.c["recent_ncc_any"] or (s >= self.c["recent_ncc"] and lab == label):
                    return True
        self.kept[group] = (self.kept[group] + [(now, vec, label)])[-self.c["window"]:]
        return False


def ordered_map(fn, items, workers: int):
    """``map(fn, items)`` on a thread pool, results in input order, at most 4 x workers items in flight."""
    pending = deque()
    with ThreadPoolExecutor(workers) as pool:
        for item in items:
            pending.append(pool.submit(fn, item))
            if len(pending) >= 4 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def frame_sources(cfg: dict, limit: int | None, only: list[str] | None = None):
    """Yield (group, source, frame number, time in s, BGR frame) for the videos, then the images.

    ``only``: keep the sources whose ASCII path (``slug``) contains one of these strings.
    """
    root = Path(cfg["source_root"])
    keep = lambda p: not only or any(s in slug(p.relative_to(root).as_posix(), 500) for s in only)
    for video in sorted(p for p in {p for pat in cfg["videos"] for p in root.glob(pat)} if keep(p)):
        print(f"video: {slug(video.stem, 80)}", flush=True)  # ASCII: the log may not be UTF-8
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step, n, done = max(1, round(fps * cfg["video_step_s"])), 0, 0
        group = "vid_" + slug(video.stem)
        while cap.grab():
            if n % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    yield group, video.relative_to(root).as_posix(), n, n / fps, frame
                    done += 1
                    if limit and done >= limit:
                        break
            n += 1
        cap.release()
    files = sorted(p for p in {p for pat in cfg["images"] for p in root.glob(pat)} if keep(p))
    per_group = Counter()
    for f in files:
        m = re.match(r"playlist_(\d+)_(.{11})_(\d+)$", f.stem)
        if m:
            group, n = f"pl{m.group(1)}_{m.group(2)}", int(m.group(3))
        elif re.match(r"frame_\d+$", f.stem):
            n = int(f.stem[6:])
            group = f"frm{n // cfg['frame_block']:02d}"
        else:
            group, n = "img_" + slug(f.stem, 16), 0
        if limit and per_group[group.split("_")[0][:3]] >= limit:
            continue
        per_group[group.split("_")[0][:3]] += 1
        img = imread(f)
        if img is not None:
            yield group, f.relative_to(root).as_posix(), n, None, img


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/data/vtv_faces.yaml")
    ap.add_argument("--out", help="output folder (default: out_dir of the config)")
    ap.add_argument("--limit", type=int, help="frames per video / per image type (quick test)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing review/ (loses manual sorting)")
    ap.add_argument("--only", nargs="+", help="only sources whose ASCII path contains one of these, e.g. tao_quan")
    ap.add_argument("--append", action="store_true",
                    help="add faces to an existing review/ (new sources only): writes faces_<tag>.csv, summary_<tag>.md")
    ap.add_argument("--tag", help="name of an --append run (default: the --only strings)")
    ap.add_argument("--workers", type=int, help="override `workers` (e.g. several runs in parallel)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["workers"] = args.workers or cfg["workers"]
    g, rv = cfg["gates"], cfg["review"]
    out = Path(args.out or cfg["out_dir"])
    review = out / "review"
    tag = (args.tag or "_".join(args.only or ["all"])) if args.append else ""
    if args.append:
        assert args.only, "--append needs --only (the new sources), so no source is processed twice"
    elif review.exists() and any(review.rglob("*.png")):
        assert args.overwrite, f"{review} already has faces (maybe already checked); use --overwrite or --append"
        shutil.rmtree(review)
        shutil.rmtree(out / "rejected_examples", ignore_errors=True)
    folders = CLASSES + ["_unsure", "_reject"] if rv["prelabel"] == "ferplus" else ["_unlabeled", "_reject"]
    for d in folders:
        (review / d).mkdir(parents=True, exist_ok=True)

    cv2.setNumThreads(1)  # parallel over frames instead of inside each OpenCV call
    models, det_args = cfg["models"], (Path(cfg["models"]["yunet"]), cfg["detect"]["min_score"], cfg["detect"]["max_side"])
    tpl = ckplus_template(Path(cfg["ckplus_images"]), Detector(*det_args))
    print("CK+ template (48x48): " + " ".join(f"({x:.1f},{y:.1f})" for x, y in tpl), flush=True)
    fer = FerPlus(Path(models["ferplus"])) if rv["prelabel"] == "ferplus" else None
    local = threading.local()  # one YuNet instance per worker thread (it keeps its input size)

    def analyse(item):
        """All per-frame work that does not depend on other frames (runs in a worker thread)."""
        group, source, n, t, frame = item
        if not hasattr(local, "det"):
            local.det = Detector(*det_args)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []
        for k, (score, box, pts) in enumerate(sorted(local.det(frame), key=lambda x: x[1][0])):
            crop, outside = align(gray, pts, tpl, rv["size"])
            g48 = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
            g96 = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
            m = {"score": score, "outside": outside, **measure(pts, tpl, g48, g96)}
            reason, p = failed_gate(m, g), None
            if reason is None:
                m["recheck_score"], m["recheck_err"] = recheck(local.det, crop, tpl)
                if m["recheck_score"] < g["recheck_score"] or m["recheck_err"] > g["recheck_err"]:
                    reason = "recheck"
            if reason is None and fer is not None:
                p = fer(crop)
            faces.append({"k": k, "box": box, "pts": pts, "m": m, "reason": reason, "p": p, "crop": crop,
                          "vec": ncc_vector(g48) if reason is None else None})
        return (group, source, n, t), faces

    dedup = Dedup(cfg["dedup"])
    rng = random.Random(0)
    rejected, seen = defaultdict(list), Counter()  # reservoir sample of rejected faces per gate
    counts, rows, t0, n_frames = Counter(), [], time.time(), 0

    # an --append run writes its own faces_<tag>.csv / summary_<tag>.md (step 2 reads faces*.csv)
    suffix = f"_{tag}" if tag else ""
    sources = frame_sources(cfg, args.limit, args.only)
    for (group, source, n, t), faces in ordered_map(analyse, sources, cfg["workers"]):
        n_frames += 1
        dedup.new_frame(group)
        for f in faces:
            k, box, pts, m, reason, p, crop = (f[key] for key in ("k", "box", "pts", "m", "reason", "p", "crop"))
            if reason is None:
                label = FERPLUS[int(p.argmax())] if p is not None else ""
                if dedup.duplicate(group, f["vec"], label, t):
                    reason = "duplicate"
            counts[reason or "kept"] += 1
            if reason:
                seen[reason] += 1
                small = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
                if len(rejected[reason]) < rv["reject_examples"]:
                    rejected[reason].append(small)
                elif (j := rng.randrange(seen[reason])) < rv["reject_examples"]:
                    rejected[reason][j] = small
                continue
            face_id = f"{group}_{n:06d}_{k}"
            row = {"id": face_id, "group": group, "source": source, "frame": n,
                   "time_s": None if t is None else round(t, 2),
                   "box": " ".join(f"{v:.0f}" for v in box), "landmarks": " ".join(f"{v:.1f}" for v in pts.ravel()),
                   **{key: round(v, 4) for key, v in m.items()}}
            folder = "_unlabeled"
            if p is not None:
                row.update({f"p_{c}": round(float(v), 4) for c, v in zip(FERPLUS, p)})
                best = FERPLUS[int(p.argmax())]
                row["suggested"] = best
                folder = best if best != "contempt" and p.max() >= rv["min_conf"] else "_unsure"
            row["review_folder"] = folder
            imwrite(review / folder / f"{face_id}.png", crop)
            rows.append(row)
        if n_frames % 5000 == 0:
            print(f"{tag} {n_frames} frames, {time.time() - t0:.0f} s: kept {counts['kept']} | "
                  + " ".join(f"{r} {counts[r]}" for r in GATES if counts[r]), flush=True)
            # save the rows so far: a run stopped midway keeps the metadata of the faces it wrote
            pd.DataFrame(rows).to_csv(out / f"faces{suffix}.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out / f"faces{suffix}.csv", index=False)
    for reason, crops in rejected.items():
        d = out / "rejected_examples" / reason
        d.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(crops):
            imwrite(d / f"{tag + '_' if tag else ''}{i:03d}.png", c)
    if GUIDE.exists():
        try:
            shutil.copy(GUIDE, review / "HUONG_DAN.md")
        except OSError:  # parallel --append runs may copy it at the same moment
            pass

    total = sum(counts.values())
    lines = [f"# VTV faces for review\n", f"{n_frames} frames, {total} YuNet detections (score >= "
             f"{cfg['detect']['min_score']}), {counts['kept']} kept. Time {time.time() - t0:.0f} s.\n",
             "| Gate (first failed) | Faces |", "|---|---|"]
    lines += [f"| {r} | {counts[r]} |" for r in GATES] + [f"| **kept** | {counts['kept']} |", ""]
    if len(df):
        lines += ["| Review folder | Faces |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in df["review_folder"].value_counts().items()] + [""]
        kind = df["group"].str.extract(r"^(vid|pl|frm|img)")[0]
        lines += ["| Source | Groups | Faces |", "|---|---|---|"]
        lines += [f"| {k} | {df.loc[kind == k, 'group'].nunique()} | {(kind == k).sum()} |" for k in sorted(kind.unique())]
    (out / f"summary{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"review folder: {review.resolve()}")


if __name__ == "__main__":
    main()
