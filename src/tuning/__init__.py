"""Hyperparameter tuning: grid and random search, split across machines (Milestone 1).

    python -m src.cli tune --config configs/tune_grid.yaml --shard 1/4
    python -m src.cli tune-report --config configs/tune_grid.yaml --config configs/tune_random.yaml
"""

from src.tuning.runner import run_tuning
from src.tuning.search import make_trials, parse_shard, select_shard

__all__ = ["make_trials", "parse_shard", "run_tuning", "select_shard"]
