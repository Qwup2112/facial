"""Zip what the Colab notebook needs: code, configs and the CK+ stage data.

Writes ``colab/upload/facial_emotion_colab.zip`` (src/, configs/, requirements.txt, data/ckplus/);
upload it to ``MyDrive/facial_emotion/`` and open ``colab/ckplus_colab.ipynb`` in Colab.
Run ``python -m src.data.ckplus_folds`` first if data/ckplus/ is missing. Rebuild the zip after
every code or config change. The zip holds CK+ images: keep it private (licence).

Example:
    python colab/make_bundle.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab" / "upload" / "facial_emotion_colab.zip"
INCLUDE = ["src", "configs", "requirements.txt", "data/ckplus"]


def main():
    assert (ROOT / "data/ckplus/index.csv").exists(), "run `python -m src.data.ckplus_folds` first"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            path = ROOT / item
            files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
            for f in files:
                if "__pycache__" in f.parts:
                    continue
                z.write(f, f.relative_to(ROOT).as_posix())
                n += 1
    print(f"{n} files, {OUT.stat().st_size / 2**20:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
