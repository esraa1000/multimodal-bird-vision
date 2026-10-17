"""Project tasks. Importing this package registers every task in ``TASKS``."""

from src.tasks import debug, placeholders  # noqa: F401
from src.tasks.base import Task

__all__ = ["Task"]
