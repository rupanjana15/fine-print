"""Repo paths and YAML config loading."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"


def load_config(name: str) -> dict:
    """Load configs/<name>.yaml."""
    with open(CONFIGS / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def data_path(kind: str) -> Path:
    """Absolute path for data/<kind> (raw, interim, processed), created if missing."""
    path = ROOT / load_config("data")["paths"][kind]
    path.mkdir(parents=True, exist_ok=True)
    return path
