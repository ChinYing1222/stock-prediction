"""實驗二測試：報酬率目標對齊、價格還原、測試日期與實驗一相同。"""
import numpy as np
import pytest

import config
from src.data import load_clean_features, prepare_dataset, prepare_return_dataset, return_to_price


@pytest.fixture(scope="module")
def df():
    return load_clean_features(verbose=False)


@pytest.mark.parametrize("feature_set", ["A_ret", "B_ret"])
def test_return_target_is_next_day_return(df, feature_set):
    d = prepare_return_dataset(df, feature_set)
    y_test = d["target_scaler"].inverse_transform(d["y_test"].reshape(-1, 1)).ravel()
    # 目標 = 隔天報酬率 = close(t+1) / close(t) - 1
    np.testing.assert_allclose(y_test, d["test_actual"] / d["test_today_close"] - 1, atol=1e-6)
    # 也等於 df 中目標日（隔天）那一列的 ret
    close = df["close"].to_numpy()
    first_target_row = config.WINDOW + len(d["y_train"]) + len(d["y_val"])
    expected = close[first_target_row] / close[first_target_row - 1] - 1
    assert y_test[0] == pytest.approx(expected, abs=1e-6)
    assert d["test_actual_return"][0] == pytest.approx(expected)


def test_return_to_price_handmade():
    np.testing.assert_allclose(return_to_price([100.0, 200.0], [0.01, -0.05]), [101.0, 190.0])
    # 報酬率 0（naive）→ 價格不變
    np.testing.assert_allclose(return_to_price([150.0], [0.0]), [150.0])


def test_return_to_price_reconstructs_actual(df):
    d = prepare_return_dataset(df, "A_ret")
    rebuilt = return_to_price(d["test_today_close"], d["test_actual_return"])
    np.testing.assert_allclose(rebuilt, d["test_actual"], rtol=1e-10)


@pytest.mark.parametrize("feature_set", ["A_ret", "B_ret"])
def test_return_dates_match_price_experiment(df, feature_set):
    price = prepare_dataset(df, "A")
    ret = prepare_return_dataset(df, feature_set)
    np.testing.assert_array_equal(ret["test_dates"], price["test_dates"])
    assert len(ret["test_dates"]) == 182
    for split in ["train", "val", "test"]:
        assert len(ret[f"y_{split}"]) == len(price[f"y_{split}"])
    np.testing.assert_allclose(ret["test_today_close"], price["test_today_close"])
