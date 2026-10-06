"""Load YAML configs with `_base_` inheritance."""
import copy
from pathlib import Path

import yaml


def _merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``; lists and scalars are replaced."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path) -> dict:
    """Load a YAML config, resolving ``_base_`` (relative to the file) recursively."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        cfg = _merge(load_config(path.parent / base), cfg)
    return cfg


def save_config(cfg: dict, path) -> None:
    """Write a (merged) config dict to YAML."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
