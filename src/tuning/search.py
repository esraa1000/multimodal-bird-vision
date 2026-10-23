"""Turn a ``tune`` config section into a fixed, numbered list of trials.

Grid search tries every combination of the listed values. Random search
samples ``n_trials`` points from distributions, with ``seed`` so the list is
identical on every machine. Because the full list is fixed before anything
runs, a team can split it with ``--shard k/n`` (member k runs every n-th
trial) and the results still form one search.

    tune:
      method: grid
      space:
        train.lr: [3.0e-4, 1.0e-3, 3.0e-3]
        model.dropout: [0.1, 0.3, 0.5]

    tune:
      method: random
      n_trials: 18
      seed: 42
      space:
        train.lr: {type: loguniform, low: 2.0e-4, high: 5.0e-3}
        model.dropout: {type: uniform, low: 0.0, high: 0.6}
        data.augment: [light, strong]      # a list = pick one value at random

Distribution types: ``uniform``, ``loguniform`` (for values spanning orders of
magnitude, like learning rates), ``int`` (inclusive bounds), ``choice``.
"""

from __future__ import annotations

import itertools
import math

import numpy as np


def grid_points(space: dict) -> list[dict]:
    keys = list(space)
    for key in keys:
        if not isinstance(space[key], list) or not space[key]:
            raise ValueError(f"Grid search needs a non-empty list of values for '{key}'")
    return [dict(zip(keys, combo)) for combo in itertools.product(*(space[k] for k in keys))]


def _sample(rng: np.random.Generator, key: str, spec):
    if isinstance(spec, list):
        return spec[int(rng.integers(len(spec)))]
    kind = spec.get("type")
    if kind == "choice":
        return spec["values"][int(rng.integers(len(spec["values"])))]
    if kind == "int":
        return int(rng.integers(int(spec["low"]), int(spec["high"]) + 1))
    low, high = float(spec["low"]), float(spec["high"])
    if kind == "uniform":
        value = rng.uniform(low, high)
    elif kind == "loguniform":
        value = math.exp(rng.uniform(math.log(low), math.log(high)))
    else:
        raise ValueError(f"Unknown distribution '{kind}' for '{key}'. "
                         "Use uniform, loguniform, int or choice (or a list).")
    return float(f"{value:.4g}")   # readable values, e.g. 0.001234


def random_points(space: dict, n_trials: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{key: _sample(rng, key, spec) for key, spec in space.items()} for _ in range(n_trials)]


def make_trials(tune_cfg: dict) -> list[dict]:
    """``[{"trial_id": 0, "params": {"train.lr": 0.001, ...}}, ...]``"""
    method = tune_cfg.get("method")
    space = dict(tune_cfg.get("space") or {})
    if not space:
        raise ValueError("tune.space is empty")
    if method == "grid":
        points = grid_points(space)
    elif method == "random":
        points = random_points(space, int(tune_cfg.get("n_trials", 10)), int(tune_cfg.get("seed", 0)))
    else:
        raise ValueError(f"Unknown tune.method '{method}'. Use grid or random.")
    return [{"trial_id": i, "params": params} for i, params in enumerate(points)]


def parse_shard(text: str | None) -> tuple[int, int]:
    """``"2/4"`` -> ``(2, 4)``: the 2nd of 4 equal slices. ``None`` -> everything."""
    if not text:
        return 1, 1
    try:
        k, n = (int(part) for part in text.split("/"))
    except ValueError:
        raise ValueError(f"--shard must look like 2/4, got '{text}'") from None
    if not 1 <= k <= n:
        raise ValueError(f"--shard {text}: need 1 <= k <= n")
    return k, n


def select_shard(trials: list[dict], k: int, n: int) -> list[dict]:
    """Every n-th trial starting at the k-th, so slow and fast trials spread evenly."""
    return trials[k - 1::n]
