"""Load experiment configs. A config may `inherit:` another and override keys."""

from __future__ import annotations

import os
from typing import Any

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of `base` with `override` merged in recursively."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config, resolving `inherit:` chains relative to configs/.

    Args:
        path: config file path, absolute or relative to configs/.

    Returns:
        The fully merged configuration dictionary.
    """
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(CONFIG_DIR, path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _deep_merge(load_config(parent), cfg)
    return cfg
