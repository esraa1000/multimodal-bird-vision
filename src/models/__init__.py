"""Encoders, classification heads, colorization model, caption decoder (owners: M3, M4).

Every encoder module registers itself with ``@MODELS.register("name")`` so it can be
selected from a config file without touching the training code. Import new
modules here so they register.
"""

from src.models import scratch_cnn  # noqa: F401  (registers "scratch_cnn")
from src.models import vgg_cnn  # noqa: F401
from src.models.base import Classifier, Encoder
from src.models.factory import build_classifier

__all__ = ["Classifier", "Encoder", "build_classifier"]
