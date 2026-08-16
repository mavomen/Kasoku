"""Forecasting model architectures (GRU, LSTM, temporal conv)."""

from kasoku.models.base import ModelError
from kasoku.models.gru import GRU_SIZES, build_gru, build_gru_by_size
from kasoku.models.lstm import LSTM_SIZES, build_lstm, build_lstm_by_size
from kasoku.models.tcn import TCN_SIZES, build_tcn, build_tcn_by_size

__all__ = [
    "GRU_SIZES",
    "LSTM_SIZES",
    "TCN_SIZES",
    "ModelError",
    "build_gru",
    "build_gru_by_size",
    "build_lstm",
    "build_lstm_by_size",
    "build_tcn",
    "build_tcn_by_size",
]
