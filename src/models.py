"""模型：naive 基準、線性迴歸、隨機森林、LSTM。"""
import random

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from torch import nn

import config


def set_seed(seed=config.SEED):
    """固定 random、numpy、torch 的隨機種子，確保結果可重現。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def naive_predict(today_close):
    """基準（非模型）：預測隔天收盤價 = 今天（視窗最後一天）的收盤價。"""
    return np.asarray(today_close, dtype=float).copy()


def flatten(X):
    """將 (N, window, n_features) 攤平成 (N, window * n_features)，供 scikit-learn 使用。"""
    return X.reshape(len(X), -1)


def fit_linear(X, y):
    """以攤平視窗訓練線性迴歸。"""
    return LinearRegression().fit(flatten(X), y)


def fit_rf(X, y, n_estimators=config.RF_N_ESTIMATORS):
    """以攤平視窗訓練隨機森林。限制：無法預測高於訓練目標最大值的價格。"""
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=config.SEED, n_jobs=-1)
    return model.fit(flatten(X), y)


def predict_flat(model, X):
    """scikit-learn 模型對視窗資料預測（輸出為縮放後的數值）。"""
    return model.predict(flatten(X))


class LSTMRegressor(nn.Module):
    """LSTM → Dropout → Linear，以最後一個時間步的輸出預測隔天收盤價。"""

    def __init__(self, n_features):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=config.LSTM_HIDDEN,
            num_layers=config.LSTM_LAYERS,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.LSTM_DROPOUT)
        self.fc = nn.Linear(config.LSTM_HIDDEN, 1)

    def forward(self, x):
        out, _ = self.lstm(x)  # out: (batch, window, hidden)
        last = self.dropout(out[:, -1, :])
        return self.fc(last).squeeze(-1)  # (batch,)
