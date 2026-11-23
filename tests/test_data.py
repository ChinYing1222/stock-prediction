"""資料處理測試：視窗形狀、標準化只用訓練資料。"""
import numpy as np
import pytest

import config
from src.data import load_clean_features, make_windows, prepare_dataset


@pytest.fixture(scope="module")
def df():
    return load_clean_features(verbose=False)


def test_make_windows_shape_and_alignment():
    values = np.arange(20, dtype=float).reshape(10, 2)  # 10 天、2 個特徵
    target = np.arange(10, dtype=float)
    X, y = make_windows(values, target, window=3)
    assert X.shape == (7, 3, 2)
    assert y.shape == (7,)
    # 第 0 個樣本：輸入第 0~2 天，目標第 3 天
    np.testing.assert_array_equal(X[0], values[0:3])
    assert y[0] == target[3]


@pytest.mark.parametrize("feature_set", ["A", "B"])
def test_prepare_dataset_shapes(df, feature_set):
    d = prepare_dataset(df, feature_set)
    n_feat = len(config.FEATURE_SETS[feature_set])
    n_total = len(d["y_train"]) + len(d["y_val"]) + len(d["y_test"])
    assert n_total == len(df) - config.WINDOW
    for split in ["train", "val", "test"]:
        X = d[f"X_{split}"]
        assert X.shape[1:] == (config.WINDOW, n_feat)
        assert len(X) == len(d[f"y_{split}"])
    assert len(d["test_dates"]) == len(d["test_actual"]) == len(d["test_today_close"]) == len(d["y_test"])


def test_scaler_fitted_on_train_only(df):
    d = prepare_dataset(df, "B")
    # scaler 只用訓練資料列 fit：縮放後訓練期收盤價最大值為 1、最小值為 0
    train_close = d["X_train"][:, :, 0]
    assert train_close.max() == pytest.approx(1.0)
    assert d["y_train"].max() == pytest.approx(1.0)
    assert train_close.min() == pytest.approx(0.0)
    # 測試期價格高於訓練期，縮放後會大於 1（證明 scaler 沒有看到測試資料）
    assert d["y_test"].max() > 1.0
    # target scaler 的範圍等於訓練資料列的收盤價範圍
    train_rows_close = df["close"].to_numpy()[: d["train_end_row"]]
    assert d["target_scaler"].data_max_[0] == pytest.approx(train_rows_close.max())
