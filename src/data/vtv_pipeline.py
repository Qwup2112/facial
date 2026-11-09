"""VTV pipeline runner: the VTV steps one at a time, resumable, and never two runs at once.

Why (2026-11-07): a chain of shell scripts was believed killed and restarted, but the old chain was still alive, so
two vtv_faces runs worked on the same episode (same PNGs, both rewriting faces_<tag>.csv) and the next steps would
have run twice in parallel on 2 GB of free RAM. This runner replaces those scratch scripts:

  extract <tag>  python -m src.data.vtv_faces --config <faces-config> --append --only <tag> --tag <tag>
                 done when ``<out_dir>/<tag>.done`` exists (same marker as the old scripts); a partial run
                 (faces_<tag>.csv or PNGs without the marker) is removed first and redone (vtv_faces is
                 deterministic: ids are <group>_<frame>_<k>, the same faces come back)
  relabel        python -m src.data.vtv_relabel --csv faces_<tag>.csv ... (idempotent: faces already sorted by
                 the ensemble are skipped, faces moved by the user are never touched)
  build          python -m src.data.vtv_build (person split, step 2b balance, target_dataset.zip)

Safety:
  * lock file ``<out_dir>/pipeline.lock``: a second runner exits (code 2) while the first one is alive
  * any other process running a ``src.data.vtv_*`` module or an old ``run_*.sh`` chain (an orphan) stops the start
    (exit 2, listed), or with ``--wait`` the runner waits for it to end
  * ``--detach`` starts the runner as a detached process (not a child of the terminal / Claude Code task that
    launched it), output appended to the log; ``--status`` shows the state without running anything

Examples:
    python -m src.data.vtv_pipeline --status --extract tro_doi_tap_12 tro_doi_tap_15
    python -m src.data.vtv_pipeline --faces-config configs/data/vtv_faces_eye16.yaml \\
        --extract tro_doi_tap_12 tro_doi_tap_15 --relabel --build --detach
    python -m src.data.vtv_pipeline --build --detach            # rebuild after the human review
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil

from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]
FOREIGN = re.compile(r"src\.data\.vtv_(?!pipeline)\w+|run_(?:tro_doi|eye16|vtv)[\w-]*\.sh")  # VTV steps / old chains


def now() -> str:
    return datetime.now().strftime("%H:%M")


class Log:
    """Append lines to the runner log (and stdout)."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def foreign_processes() -> list:
    """(pid, command line) of other processes doing VTV work, excluding this runner and its children."""
    me = psutil.Process()
    mine = {me.pid} | {p.pid for p in me.children(recursive=True)} | {p.pid for p in me.parents()}
    found = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except (psutil.Error, TypeError):
            continue
        if p.info["pid"] not in mine and FOREIGN.search(cmd) and "vtv_pipeline" not in cmd:
            found.append((p.info["pid"], cmd[-160:]))
    return found


def lock_holder(lock: Path):
    """The live runner holding ``lock`` as a dict, or None (no lock or a stale one)."""
    try:
        info = json.loads(lock.read_text(encoding="utf-8"))
        p = psutil.Process(info["pid"])
        if "vtv_pipeline" in " ".join(p.cmdline()) and p.pid != os.getpid():
            return info
    except (OSError, ValueError, KeyError, psutil.Error):
        pass
    return None


def acquire(lock: Path, log: Log) -> bool:
    """Create ``lock`` atomically; a stale lock (its runner is gone) is replaced."""
    info = json.dumps({"pid": os.getpid(), "started": f"{datetime.now():%Y-%m-%d %H:%M}", "cmd": " ".join(sys.argv[1:])})
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = lock_holder(lock)
            if holder:
                log(f"another runner holds {lock} (pid {holder['pid']}, since {holder['started']}): not starting")
                return False
            lock.unlink(missing_ok=True)  # stale
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(info)
        return True
    return False


def release(lock: Path) -> None:
    try:
        if json.loads(lock.read_text(encoding="utf-8"))["pid"] == os.getpid():
            lock.unlink()
    except (OSError, ValueError, KeyError):
        pass


def tag_state(out: Path, tag: str) -> dict:
    """Marker, csv rows and PNGs in review/ of one extraction tag."""
    csv = out / f"faces_{tag}.csv"
    rows = len(pd.read_csv(csv, usecols=["id"])) if csv.exists() and csv.stat().st_size > 1 else 0
    pngs = [f for f in (out / "review").glob("*/*.png") if f"_{tag}_" in f"_{f.stem}"]
    return {"done": (out / f"{tag}.done").exists(), "csv": csv.exists(), "rows": rows, "pngs": pngs}


def clean_partial(out: Path, tag: str, log: Log) -> None:
    """Remove what a stopped run of ``tag`` left (its rows and PNGs), so the redo starts clean."""
    st = tag_state(out, tag)
    csv = out / f"faces_{tag}.csv"
    folders = {}
    if st["rows"]:
        df = pd.read_csv(csv, usecols=["id", "review_folder"])
        folders = dict(zip(df["id"], df["review_folder"]))
        user_moved = [f for f in st["pngs"] if f.stem in folders and f.parent.name != folders[f.stem]]
        assert not user_moved, f"{len(user_moved)} faces of the partial run {tag} were moved by hand; check first"
    ids = set(folders)
    extra = [f for f in (out / "review").glob("*/*.png") if f.stem in ids and f not in st["pngs"]]
    for f in st["pngs"] + extra:
        f.unlink()
    for f in (csv, out / f"summary_{tag}.md"):
        f.unlink(missing_ok=True)
    log(f"{tag}: partial run removed ({st['rows']} csv rows, {len(st['pngs']) + len(extra)} PNGs); redo")


def run(cmd: list, step_log: Path, log: Log) -> int:
    """Run one step as a child process (output to ``step_log``); returns the exit code."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = time.time()
    free = psutil.virtual_memory().available / 2**30
    log(f"start: {' '.join(cmd[2:])} (free RAM {free:.1f} GB, log {step_log.as_posix()})")
    with open(step_log, "w", encoding="utf-8") as f:
        code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"end: {cmd[2]} exit {code}, {time.time() - t0:.0f} s")
    return code


def status(out: Path, tags: list, lock: Path) -> None:
    holder = lock_holder(lock)
    print(f"runner lock: {'pid %s since %s (%s)' % (holder['pid'], holder['started'], holder['cmd']) if holder else 'free'}")
    for pid, cmd in foreign_processes():
        print(f"other VTV process: pid {pid}: {cmd}")
    for t in tags:
        st = tag_state(out, t)
        state = "done" if st["done"] else ("partial" if st["csv"] or st["pngs"] else "to do")
        print(f"{t}: {state}, {st['rows']} csv rows, {len(st['pngs'])} PNGs")
    rv = out / "review"
    counts = {d.name: len(list(d.glob("*.png"))) for d in sorted(rv.iterdir()) if d.is_dir()}
    print("review/: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + f" (total {sum(counts.values())})")
    print(f"free RAM {psutil.virtual_memory().available / 2**30:.1f} GB")


def detach(log: Log) -> None:
    """Start this command again as a detached background process and return.

    The runner writes its log itself; its console output (e.g. a traceback) goes to ``<log>.out``."""
    args = [a for a in sys.argv[1:] if a != "--detach"]
    cmd = [sys.executable, "-m", "src.data.vtv_pipeline", *args]
    out = open(log.path.with_suffix(".out"), "a", encoding="utf-8")
    kw = {"start_new_session": True}
    if os.name == "nt":
        kw = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
    p = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                         env=dict(os.environ, PYTHONIOENCODING="utf-8"), close_fds=True, **kw)
    log(f"detached runner started: pid {p.pid}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/data/vtv_faces.yaml", help="config of relabel and build")
    ap.add_argument("--faces-config", help="config of the extract steps (default: --config)")
    ap.add_argument("--extract", nargs="+", default=[], help="tags (vtv_faces --only strings) to add, in order")
    ap.add_argument("--workers", type=int, default=8, help="vtv_faces worker threads (one process at a time)")
    ap.add_argument("--relabel", action="store_true", help="ensemble relabel of the extracted tags' csvs")
    ap.add_argument("--relabel-csv", nargs="+", default=[], help="more faces*.csv names to relabel")
    ap.add_argument("--build", action="store_true", help="vtv_build at the end (dataset + zip)")
    ap.add_argument("--status", action="store_true", help="show the state and exit")
    ap.add_argument("--wait", action="store_true", help="wait for other VTV processes instead of exiting")
    ap.add_argument("--detach", action="store_true", help="run detached in the background (log only)")
    ap.add_argument("--log", default="runs/logs/vtv_pipeline.log")
    args = ap.parse_args()
    os.chdir(ROOT)
    cfg = load_config(args.config)
    out = Path(cfg["out_dir"])
    lock = out / "pipeline.lock"
    log = Log(Path(args.log))

    if args.status:
        return status(out, args.extract, lock)
    if args.detach:
        return detach(log)
    if not acquire(lock, log):
        sys.exit(2)
    try:
        while others := foreign_processes():
            listing = "; ".join(f"pid {p}: {c}" for p, c in others)
            if not args.wait:
                log(f"other VTV processes running, not starting (use --wait): {listing}")
                sys.exit(2)
            log(f"waiting for other VTV processes: {listing}")
            while foreign_processes():
                time.sleep(60)
        log(f"runner start (pid {os.getpid()}): {' '.join(sys.argv[1:])}")
        py, faces_cfg = sys.executable, args.faces_config or args.config
        failed = []
        for tag in args.extract:
            if (out / f"{tag}.done").exists():
                log(f"{tag}: done before, skip")
                continue
            st = tag_state(out, tag)
            if st["csv"] or st["pngs"]:
                clean_partial(out, tag, log)
            code = run([py, "-m", "src.data.vtv_faces", "--config", faces_cfg, "--append", "--only", tag,
                        "--tag", tag, "--workers", str(args.workers)], Path(f"runs/logs/vtv_faces_{tag}.log"), log)
            summary = out / f"summary_{tag}.md"
            line = next((s for s in summary.read_text(encoding="utf-8").splitlines() if re.match(r"\d+ frames", s)),
                        "") if summary.exists() else ""
            if code == 0 and line and not line.startswith("0 frames"):
                (out / f"{tag}.done").touch()
                log(f"{tag}: {line}")
            else:
                failed.append(tag)
                log(f"{tag}: FAILED (exit {code}, summary: {line or 'none'}); see runs/logs/vtv_faces_{tag}.log")
        csvs = [f"faces_{t}.csv" for t in args.extract if (out / f"{t}.done").exists()] + args.relabel_csv
        if args.relabel and csvs:
            if run([py, "-m", "src.data.vtv_relabel", "--config", args.config, "--csv", *csvs],
                   Path("runs/logs/vtv_relabel.log"), log):
                failed.append("relabel")
        if args.build and "relabel" not in failed:
            if run([py, "-m", "src.data.vtv_build", "--config", args.config], Path("runs/logs/vtv_build.log"), log):
                failed.append("build")
            else:
                for s in Path("runs/logs/vtv_build.log").read_text(encoding="utf-8").splitlines()[-3:]:
                    log(f"build: {s}")
        log(f"ALL DONE{' with failures: ' + ', '.join(failed) if failed else ''} ({now()})")
        sys.exit(1 if failed else 0)
    finally:
        release(lock)


if __name__ == "__main__":
    main()
