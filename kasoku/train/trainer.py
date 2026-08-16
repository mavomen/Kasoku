"""Training loop shared across all models and execution modes.

The loop trains a single Keras model against windowed inputs. Execution mode
is a runtime choice: the GradientTape train step is wrapped once via
``kasoku.execution.compile_call`` and then runs eagerly, under ``tf.function``,
or under ``tf.function(jit_compile=True)`` without any mode-specific code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from kasoku.execution import compile_call
from kasoku.train.base import TrainError, TrainingResult

__all__ = ["load_checkpoint", "train"]


def train(
    model: tf.keras.Model,
    x_train,
    y_train,
    *,
    x_val=None,
    y_val=None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    mode: str = "eager",
    seed: int | None = None,
    early_stopping_patience: int = 5,
    checkpoint_dir: str | Path | None = None,
    verbose: bool = False,
) -> TrainingResult:
    """Train ``model`` on ``(x_train, y_train)`` and return a :class:`TrainingResult`.

    ``x_train`` has shape ``(n, lookback, 1)`` and ``y_train`` shape
    ``(n, horizon)`` (windowed by ``kasoku.pipeline.window_series``). When
    ``x_val``/``y_val`` are given, validation loss is computed each epoch and
    used for early stopping. ``early_stopping_patience=0`` disables early
    stopping. With ``checkpoint_dir`` set, the best weights are persisted to
    ``model.weights.h5`` inside it on every validation improvement. Validation
    loss drives early stopping when ``early_stopping_patience >= 1`` and a
    validation set is provided; on early stop (or at completion) the best
    weights are restored onto ``model``.
    """
    _validate_training(model, x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate)
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    x_train = _as_tensor(x_train)
    y_train = _as_tensor(y_train)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if seed is not None:
        train_ds = train_ds.shuffle(len(x_train), seed=seed)
    train_ds = train_ds.batch(batch_size, drop_remainder=False)

    has_val = x_val is not None
    if has_val:
        val_ds = tf.data.Dataset.from_tensor_slices((_as_tensor(x_val), _as_tensor(y_val)))
        val_ds = val_ds.batch(batch_size, drop_remainder=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    step = compile_call(_train_step, mode)

    def run_step(x_batch, y_batch):
        return step(model, optimizer, x_batch, y_batch)

    result = TrainingResult(model=model)
    best_val_loss = None
    best_weights = None
    epochs_without_improvement = 0
    checkpoint_path = (
        str(Path(checkpoint_dir) / "model.weights.h5") if checkpoint_dir is not None else None
    )
    early_stopping = early_stopping_patience >= 1 and has_val
    for epoch in range(epochs):
        batch_losses = []
        for x_batch, y_batch in train_ds:
            batch_losses.append(float(run_step(x_batch, y_batch).numpy()))
        train_loss = float(np.mean(batch_losses))
        val_loss = _validation_loss(model, val_ds) if has_val else None
        result.train_losses.append(train_loss)
        if has_val:
            result.val_losses.append(val_loss)
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                result.best_val_loss = val_loss
                best_weights = model.get_weights()
                epochs_without_improvement = 0
                if checkpoint_path is not None:
                    _save_checkpoint(model, checkpoint_path)
            else:
                epochs_without_improvement += 1
        result.epochs_trained = epoch + 1
        if verbose:
            detail = f"val_loss={val_loss:.6f}" if has_val else ""
            print(f"epoch {epoch + 1}/{epochs} train_loss={train_loss:.6f} {detail}")
        if early_stopping and epochs_without_improvement >= early_stopping_patience:
            result.stopped_reason = "early_stopping"
            break
    if best_weights is not None:
        model.set_weights(best_weights)
    result.checkpoint_path = checkpoint_path if best_weights is not None else None
    return result


def _train_step(model: tf.keras.Model, optimizer, x_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        loss = tf.reduce_mean(tf.square(predictions - y_batch))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def _validation_loss(model: tf.keras.Model, val_ds) -> float:
    batch_losses = []
    for x_batch, y_batch in val_ds:
        predictions = model(x_batch, training=False)
        batch_losses.append(float(tf.reduce_mean(tf.square(predictions - y_batch)).numpy()))
    return float(np.mean(batch_losses))


def _save_checkpoint(model: tf.keras.Model, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(path)


def load_checkpoint(model: tf.keras.Model, path: str | Path) -> None:
    """Load weights from a checkpoint produced by :func:`train` into ``model``.

    The model must be built with the same architecture as the one that was
    trained; weights are independent of the execution mode used to train.
    """
    model.load_weights(path)


def _as_tensor(array) -> tf.Tensor:
    return tf.convert_to_tensor(array, dtype=tf.float32)


def _validate_training(model, x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate):
    if not isinstance(model, tf.keras.Model):
        raise TrainError(f"model must be a tf.keras.Model, got {type(model).__name__}")
    x_arr = np.asarray(x_train)
    y_arr = np.asarray(y_train)
    if x_arr.ndim != 3:
        raise TrainError(f"x_train must have shape (n, lookback, 1), got {x_arr.shape}")
    if y_arr.ndim != 2:
        raise TrainError(f"y_train must have shape (n, horizon), got {y_arr.shape}")
    if len(x_arr) != len(y_arr):
        raise TrainError(
            f"x_train and y_train must have the same length, got {len(x_arr)} and {len(y_arr)}"
        )
    if len(x_arr) < 1:
        raise TrainError("x_train must contain at least one sample")
    if epochs < 1:
        raise TrainError(f"epochs must be >= 1, got {epochs}")
    if batch_size < 1:
        raise TrainError(f"batch_size must be >= 1, got {batch_size}")
    if learning_rate < 0:
        raise TrainError(f"learning_rate must be >= 0, got {learning_rate}")
    if (x_val is None) != (y_val is None):
        raise TrainError("x_val and y_val must be provided together")
    if x_val is not None:
        xv = np.asarray(x_val)
        yv = np.asarray(y_val)
        if xv.ndim != 3 or yv.ndim != 2 or len(xv) != len(yv) or len(xv) < 1:
            raise TrainError(
                "x_val and y_val must have shapes (n, lookback, 1) and (n, horizon) "
                "with the same non-zero length"
            )
