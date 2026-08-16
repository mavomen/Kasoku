"""Errors and result types for the training loop."""

from __future__ import annotations

from dataclasses import dataclass, field

import tensorflow as tf


class TrainError(ValueError):
    """Raised when a training run cannot be performed or configured."""


@dataclass
class TrainingResult:
    """Outcome of a :func:`kasoku.train.train` run.

    ``model`` holds the best weights when early stopping ran with a patience
    of at least one; otherwise it holds the final epoch's weights.
    """

    model: tf.keras.Model
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_val_loss: float | None = None
    epochs_trained: int = 0
    stopped_reason: str = "completed"
    checkpoint_path: str | None = None
