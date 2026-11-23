"""模型測試：naive 基準等於視窗最後一天的收盤價。"""
import numpy as np

from src.data import load_clean_features, prepare_dataset
from src.models import naive_predict


def test_naive_equals_last_close_in_window():
    d = prepare_dataset(load_clean_features(verbose=False), "A")
    pred = naive_predict(d["test_today_close"])
    # 由縮放後視窗的最後一天反轉換回 USD，應等於 naive 預測
    last_scaled = d["X_test"][:, -1, 0].reshape(-1, 1)
    last_close = d["feature_scaler"].inverse_transform(last_scaled).ravel()
    np.testing.assert_allclose(pred, last_close, rtol=1e-5)


def test_naive_simple_values():
    np.testing.assert_array_equal(naive_predict([10.0, 11.0, 12.5]), [10.0, 11.0, 12.5])
