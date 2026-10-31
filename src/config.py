"""
src/config.py
Loads and exposes config.yaml values as a typed dataclass-like object.
"""
from __future__ import annotations
import os
import yaml
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load YAML config; optionally override path (useful in tests)."""
    p = Path(path) if path else _CONFIG_PATH
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ── Convenience accessors ─────────────────────────────────────────────────────
def get_config(path: str | Path | None = None) -> dict:
    cfg = load_config(path)
    # Resolve relative paths to absolute
    cfg["_root"] = str(_ROOT)
    return cfg


def root() -> Path:
    return _ROOT
