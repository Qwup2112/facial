"""Seeding helpers for reproducible runs."""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed random, numpy, torch and torch.cuda."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive numpy / random seeds from the torch worker seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
