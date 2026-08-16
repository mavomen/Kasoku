"""Training loop shared across all models and execution modes."""

from kasoku.train.base import TrainError, TrainingResult
from kasoku.train.trainer import load_checkpoint, train

__all__ = ["TrainError", "TrainingResult", "load_checkpoint", "train"]
