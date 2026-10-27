"""Encoders, classification heads, colorization model, caption decoder (owners: M3, M4).

Every encoder module registers itself with ``@MODELS.register("name")`` so it can be
selected from a config file without touching the training code. Import new
modules here so they register.
"""

from src.models import scratch_cnn  # noqa: F401
from src.models import vgg_cnn  # noqa: F401
from src.models import timm_encoder  # noqa: F401

from src.models.base import Classifier, Encoder
from src.models.factory import build_classifier
from src.models.transfer_classifier import TransferClassifier

__all__ = [
    "Classifier",
    "Encoder",
    "TransferClassifier",
    "build_classifier",
]