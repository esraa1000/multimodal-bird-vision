"""Config system: YAML files + inheritance + command-line overrides.

A config file may inherit from another one with a ``defaults`` key::

    # configs/cls_scratch.yaml
    defaults: base.yaml
    train:
      lr: 3.0e-4

Values from the child file override the parent (deep merge), and values given
on the command line as ``key.subkey=value`` override both::

    python -m src.cli train --config configs/cls_scratch.yaml train.lr=1e-3

The result is a :class:`Config`, a dict that also allows attribute access
(``cfg.train.lr``), so it can be logged to a tracker as-is.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


class Config(dict):
    """A dict with attribute access for nested keys (``cfg.train.lr``)."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = Config(value) if isinstance(value, dict) else value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(f"Config has no key '{name}'") from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Read a nested value by dotted key, e.g. ``cfg.get_path("train.lr")``."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        """Set a nested value by dotted key, creating missing levels."""
        parts = dotted.split(".")
        node: Config = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict:
        """Plain nested dict (for YAML dumping and experiment trackers)."""
        return {k: v.to_dict() if isinstance(v, Config) else copy.deepcopy(v) for k, v in self.items()}

    def __repr__(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml_with_defaults(path: Path, _seen: set[Path] | None = None) -> dict:
    path = path.resolve()
    seen = _seen or set()
    if path in seen:
        raise ValueError(f"Circular 'defaults' chain at {path}")
    seen.add(path)

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    parents = data.pop("defaults", None)
    if parents is None:
        return data
    if isinstance(parents, str):
        parents = [parents]

    merged: dict = {}
    for parent in parents:
        merged = deep_merge(merged, _load_yaml_with_defaults(path.parent / parent, seen))
    return deep_merge(merged, data)


def parse_overrides(overrides: Iterable[str]) -> dict:
    """Turn ``["train.lr=1e-3", "data.augment=none"]`` into a nested dict.

    Values are parsed with YAML, so ``1e-3`` becomes a float, ``true`` a bool,
    ``[1,2]`` a list and ``null`` None.
    """
    result = Config()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key.subkey=value")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        # YAML 1.1 reads "1e-3" as a string; make the common case work.
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        result.set_path(key.strip(), value)
    return result.to_dict()


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Config:
    """Load a YAML config, resolve its ``defaults`` chain, apply CLI overrides."""
    data = _load_yaml_with_defaults(Path(path))
    data = deep_merge(data, parse_overrides(overrides))
    return Config(data)


def save_config(cfg: Config, path: str | Path) -> None:
    """Write the fully resolved config next to a run's outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
