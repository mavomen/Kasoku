"""Unit tests for the training loop, checkpointing, and early stopping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kasoku.data import CsvSource
from kasoku.execution import ExecutionError
from kasoku.models import build_gru_by_size
from kasoku.pipeline import window_series
from kasoku.train import TrainError, load_checkpoint, train

FIXTURES = Path(__file__).parent.parent / "fixtures"
SINGLE = FIXTURES / "series_single.csv"


def _windowed():
    df = CsvSource(SINGLE).read()
    x, y = window_series(df, lookback=24, horizon=1)
    return x, y


def _noise(n=96, lookback=24):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(n, lookback, 1)).astype(np.float32)
    y = rng.normal(size=(n, 1)).astype(np.float32)
    return x, y


def _sine(n=96, lookback=24, horizon=1):
    t = np.arange(n + lookback - 1 + horizon)
    values = np.sin(2 * np.pi * t / 12.0).astype(np.float32)
    x = np.stack([values[i : i + lookback] for i in range(n)]).reshape(n, lookback, 1)
    y = np.stack([values[i + lookback : i + lookback + horizon] for i in range(n)]).astype(np.float32)
    return x, y


def test_train_loss_decreases_and_history():
    x, y = _sine()
    model = build_gru_by_size("small", 24, 1, seed=42)
    result = train(model, x, y, epochs=3, batch_size=16, mode="eager")
    assert len(result.train_losses) == 3
    assert result.epochs_trained == 3
    assert result.stopped_reason == "completed"
    assert result.train_losses[-1] < result.train_losses[0]
    assert np.isfinite(result.train_losses).all()


@pytest.mark.parametrize("mode", ["eager", "tf_function", "xla"])
def test_train_runs_in_all_modes(mode):
    x, y = _sine()
    model = build_gru_by_size("small", 24, 1, seed=42)
    result = train(model, x, y, epochs=2, batch_size=16, mode=mode)
    assert result.epochs_trained == 2
    assert len(result.train_losses) == 2
    assert np.isfinite(result.train_losses).all()


def test_train_from_fixture():
    x, y = _windowed()
    model = build_gru_by_size("small", 24, 1, seed=42)
    result = train(model, x, y, epochs=2, batch_size=32, mode="eager")
    assert len(result.train_losses) == 2
    assert np.isfinite(result.train_losses).all()


def test_train_with_validation_tracks_best_loss():
    x, y = _sine(n=120)
    model = build_gru_by_size("small", 24, 1, seed=42)
    result = train(
        model,
        x[:96],
        y[:96],
        x_val=x[96:],
        y_val=y[96:],
        epochs=3,
        batch_size=16,
        mode="eager",
        early_stopping_patience=0,
    )
    assert len(result.val_losses) == 3
    assert len(result.train_losses) == 3
    assert result.best_val_loss == min(result.val_losses)
    assert np.isfinite(result.val_losses).all()


def test_train_deterministic_with_fixed_seed():
    x, y = _sine()
    first = train(build_gru_by_size("small", 24, 1, seed=42), x, y, epochs=2, batch_size=16, seed=7)
    second = train(build_gru_by_size("small", 24, 1, seed=42), x, y, epochs=2, batch_size=16, seed=7)
    np.testing.assert_allclose(first.train_losses, second.train_losses, rtol=1e-6)


def test_train_early_stopping_stops_before_epochs():
    x, y = _noise()
    model = build_gru_by_size("small", 24, 1, seed=42)
    result = train(
        model,
        x[:72],
        y[:72],
        x_val=x[72:],
        y_val=y[72:],
        epochs=50,
        batch_size=16,
        mode="eager",
        early_stopping_patience=1,
    )
    assert result.stopped_reason == "early_stopping"
    assert result.epochs_trained < 50
    assert len(result.val_losses) == result.epochs_trained
    assert result.best_val_loss == min(result.val_losses)


def test_train_patience_zero_completes():
    x, y = _noise()
    model = build_gru_by_size("small", 24, 1, seed=42)
    result = train(
        model,
        x,
        y,
        x_val=x,
        y_val=y,
        epochs=3,
        batch_size=16,
        mode="eager",
        early_stopping_patience=0,
    )
    assert result.stopped_reason == "completed"
    assert result.epochs_trained == 3


def test_train_restores_best_weights_after_early_stop():
    x, y = _noise()
    model = build_gru_by_size("small", 24, 1, seed=42)
    result = train(
        model,
        x[:72],
        y[:72],
        x_val=x[72:],
        y_val=y[72:],
        epochs=50,
        batch_size=16,
        mode="eager",
        early_stopping_patience=1,
    )
    preds = model(x[72:], training=False).numpy()
    loss = float(np.mean(np.square(preds - y[72:])))
    assert loss <= min(result.val_losses) + 1e-6


def test_train_checkpoint_writes_and_loads(tmp_path):
    x, y = _sine(n=120)
    model = build_gru_by_size("small", 24, 1, seed=42)
    result = train(
        model,
        x[:96],
        y[:96],
        x_val=x[96:],
        y_val=y[96:],
        epochs=3,
        batch_size=16,
        mode="xla",
        checkpoint_dir=tmp_path,
        early_stopping_patience=0,
    )
    checkpoint = Path(result.checkpoint_path)
    assert checkpoint.exists()
    assert checkpoint.name == "model.weights.h5"
    expected = model.get_weights()
    rebuilt = build_gru_by_size("small", 24, 1, seed=42)
    load_checkpoint(rebuilt, checkpoint)
    for got, want in zip(rebuilt.get_weights(), expected):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    assert np.isfinite(rebuilt(x, training=False).numpy()).all()


def test_train_without_checkpoint_dir_reports_none():
    x, y = _sine()
    model = build_gru_by_size("small", 24, 1, seed=42)
    result = train(model, x, y, epochs=1, batch_size=16, mode="eager")
    assert result.checkpoint_path is None


def test_train_verbose_prints(capsys):
    x, y = _sine(n=48)
    model = build_gru_by_size("small", 24, 1, seed=42)
    train(model, x, y, epochs=1, batch_size=16, mode="eager", verbose=True)
    captured = capsys.readouterr()
    assert "epoch 1/1" in captured.out


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"epochs": 0}, "epochs"),
        ({"batch_size": 0}, "batch_size"),
        ({"learning_rate": -1.0}, "learning_rate"),
    ],
)
def test_train_validates_hyperparameters(kwargs, message):
    x, y = _sine()
    model = build_gru_by_size("small", 24, 1, seed=42)
    with pytest.raises(TrainError, match=message):
        train(model, x, y, **kwargs)


def test_train_validates_shapes():
    x, y = _sine()
    model = build_gru_by_size("small", 24, 1, seed=42)
    with pytest.raises(TrainError, match="lookback"):
        train(model, x.reshape(-1), y)
    with pytest.raises(TrainError, match="horizon"):
        train(model, x, y.reshape(-1))
    with pytest.raises(TrainError, match="same length"):
        train(model, x[:-1], y)
    with pytest.raises(TrainError, match="together"):
        train(model, x, y, x_val=x)
    with pytest.raises(TrainError, match="shapes"):
        train(model, x, y, x_val=x, y_val=y[:0])
    with pytest.raises(TrainError, match="at least one sample"):
        train(model, x[:0], y[:0])


def test_train_validates_model_type():
    with pytest.raises(TrainError, match="keras.Model"):
        train(object(), np.zeros((4, 24, 1)), np.zeros((4, 1)))


def test_train_bad_mode_raises():
    x, y = _sine(n=48)
    model = build_gru_by_size("small", 24, 1, seed=42)
    with pytest.raises(ExecutionError, match="unknown execution mode"):
        train(model, x, y, epochs=1, batch_size=16, mode="quantum")
