
"""Build a classifier from the model section of a config."""

from __future__ import annotations

from src.models.base import Classifier
from src.models.transfer_classifier import TransferClassifier
from src.registry import MODELS


def build_classifier(model_cfg: dict):
    encoder = MODELS.build(
        model_cfg["name"],
        **dict(model_cfg.get("params") or {}),
    )

    num_classes = int(model_cfg["num_classes"])
    head_type = model_cfg.get("head_type", "linear")

    if head_type == "mlp":
        model = TransferClassifier(
            encoder=encoder,
            num_classes=num_classes,
            hidden_dim=int(model_cfg.get("head_hidden_dim", 512)),
            dropout=float(model_cfg.get("head_dropout", 0.4)),
        )
    elif head_type == "linear":
        model = Classifier(
            encoder,
            num_classes,
            float(model_cfg.get("dropout", 0.0)),
        )
    else:
        raise ValueError(f"Unknown classification head: {head_type}")

    if model_cfg.get("freeze_backbone", False):
        model.freeze_encoder()

    return model
