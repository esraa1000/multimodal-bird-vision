"""Registry pattern: look up classes by the name written in a config file.

Instead of ``if name == "resnet50": ... elif name == "scratch_cnn": ...`` spread
across the code, each implementation registers itself once::

    from src.registry import MODELS

    @MODELS.register("scratch_cnn")
    class ScratchCNN(nn.Module):
        ...

and the training code builds whatever the config asks for::

    model = MODELS.build(cfg.model.name, **cfg.model.params)

Adding a new encoder then means adding one new file; the trainer, CLI and
evaluation code do not change (Open/Closed principle).
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Registry:
    """A named mapping from string keys to classes or factory functions."""

    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, key: str) -> Callable[[T], T]:
        """Decorator that registers a class/function under ``key``."""

        def decorator(obj: T) -> T:
            if key in self._items:
                raise KeyError(f"'{key}' is already registered in {self.name}")
            self._items[key] = obj  # type: ignore[assignment]
            return obj

        return decorator

    def get(self, key: str) -> Callable[..., Any]:
        if key not in self._items:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"Unknown {self.name} '{key}'. Available: {available}")
        return self._items[key]

    def build(self, key: str, *args: Any, **kwargs: Any) -> Any:
        """Instantiate the registered object with the given arguments."""
        return self.get(key)(*args, **kwargs)

    def keys(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __repr__(self) -> str:
        return f"Registry({self.name}: {self.keys()})"


# Shared registries used across the project.
MODELS = Registry("model")
DATASETS = Registry("dataset")
TASKS = Registry("task")
TRACKERS = Registry("tracker")
