"""Environment-aware paths, so the same command works locally, on Colab and on Kaggle."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def detect_environment() -> str:
    """Return ``"kaggle"``, ``"colab"`` or ``"local"``."""
    # Colab first: recent Colab images also have a /kaggle folder.
    if os.environ.get("COLAB_RELEASE_TAG") or Path("/content").exists():
        return "colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/input").exists():
        return "kaggle"
    return "local"


def resolve_path(path: str | Path) -> Path:
    """Relative paths in configs are relative to the project root."""
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_data_dir(paths_cfg: dict) -> Path:
    """Pick the data directory for the current environment.

    ``paths_cfg`` is the ``paths`` section of the config, e.g.::

        paths:
          data_dir: data                 # local default
          data_dir_colab: /content/data  # optional per-environment override
          data_dir_kaggle: /kaggle/input/cub-cached
    """
    env = detect_environment()
    chosen = paths_cfg.get(f"data_dir_{env}") or paths_cfg.get("data_dir", "data")
    return resolve_path(chosen)


def make_run_dir(output_dir: str | Path, run_name: str) -> Path:
    """Create ``outputs/<run_name>_<timestamp>/`` for one run's files."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = resolve_path(output_dir) / f"{run_name}_{stamp}"
    run_dir, n = base, 1
    while run_dir.exists():          # two runs started in the same second
        n += 1
        run_dir = base.with_name(f"{base.name}-{n}")
    run_dir.mkdir(parents=True)
    return run_dir


def git_commit_hash() -> str:
    """Current commit, logged with each run so results can be traced to code."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
