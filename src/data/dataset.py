"""Dataset over the split CSVs written by src.data.index_splits."""
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def read_split(cfg: dict, split: str) -> pd.DataFrame:
    """Load ``{splits_dir}/{split}.csv``; drop offline-augmented train rows if disabled."""
    df = pd.read_csv(Path(cfg["data"]["splits_dir"]) / f"{split}.csv")
    if split == "train" and not cfg["data"]["use_offline_aug"]:
        assert "source" in df, "train.csv has no 'source' column; re-run src.data.index_splits"
        df = df[df["source"] != "augmented"].reset_index(drop=True)
    return df


def class_names(df: pd.DataFrame, num_classes: int) -> list[str]:
    """Class names ordered by ``label_idx`` (sorted folder names); asserts the class count."""
    pairs = df[["label_idx", "label"]].drop_duplicates().sort_values("label_idx")
    names = pairs["label"].tolist()
    assert pairs["label_idx"].tolist() == list(range(len(names))), "label_idx must be 0..K-1, one name each"
    assert len(names) == num_classes, f"Expected {num_classes} classes, got {names}"
    return names


class FERDataset(Dataset):
    """Face images and integer labels from a split DataFrame, optionally cached in RAM."""

    def __init__(self, df: pd.DataFrame, transform, grayscale: bool = True, cache: bool = False):
        self.paths = df["path"].tolist()
        self.labels = df["label_idx"].to_numpy(dtype=np.int64, copy=True)  # pandas 3 returns read-only views
        self.transform = transform
        self.mode = "L" if grayscale else "RGB"
        self.images = None
        if cache:
            # Decoded uint8 arrays; 48x48 grayscale is ~2.3 KB per image.
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
                self.images = list(pool.map(self._load_array, self.paths))

    def _load_array(self, path: str) -> np.ndarray:
        with Image.open(path) as im:
            return np.asarray(im.convert(self.mode))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        if self.images is not None:
            img = Image.fromarray(self.images[i])
        else:
            with Image.open(self.paths[i]) as im:
                img = im.convert(self.mode)
        return self.transform(img), self.labels[i]


class PrefetchLoader:
    """Iterate a DataLoader in a background thread so batch preparation overlaps GPU work.

    Meant for ``num_workers: 0``: on Windows every DataLoader worker process loads the
    CUDA DLLs of torch (~1.4 GB of commit memory each), which can exhaust the paging file.
    """

    def __init__(self, loader, depth: int = 4):
        self.loader, self.depth = loader, depth

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        q, done = queue.Queue(maxsize=self.depth), object()

        def produce():
            try:
                for batch in self.loader:
                    q.put(batch)
            except BaseException as e:  # re-raised in the consumer thread
                q.put(e)
            q.put(done)

        thread = threading.Thread(target=produce, daemon=True)
        thread.start()
        while (item := q.get()) is not done:
            if isinstance(item, BaseException):
                raise item
            yield item
        thread.join()
