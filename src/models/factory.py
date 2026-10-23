"""Build a classifier from the ``model`` section of a config.

    model:
      name: scratch_cnn        # encoder registered in MODELS
      num_classes: 200
      dropout: 0.3             # before the linear head
      freeze_backbone: false   # true = train the head only (linear probe)
      params: {...}            # passed to the encoder class as keyword arguments
"""

from __future__ import annotations

from src.models.base import Classifier
from src.registry import MODELS


def build_classifier(model_cfg: dict) -> Classifier:
    encoder = MODELS.build(model_cfg["name"], **dict(model_cfg.get("params") or {}))
    model = Classifier(encoder, int(model_cfg["num_classes"]), float(model_cfg.get("dropout", 0.0)))
    if model_cfg.get("freeze_backbone", False):
        model.freeze_encoder()
    return model
