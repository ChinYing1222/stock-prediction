"""評估指標測試：以手算數字驗證。"""
import math

import pytest

from src.evaluate import compute_metrics


def test_metrics_handmade_numbers():
    actual = [100.0, 102.0, 98.0, 101.0]
    pred = [101.0, 100.0, 99.0, 103.0]
    today = [99.0, 101.0, 100.0, 100.0]
    m = compute_metrics(actual, pred, today)
    # 誤差 = [1, -2, 1, 2]
    assert m["rmse"] == pytest.approx(math.sqrt((1 + 4 + 1 + 4) / 4))
    assert m["mae"] == pytest.approx(1.5)
    assert m["mape"] == pytest.approx((1 / 100 + 2 / 102 + 1 / 98 + 2 / 101) / 4 * 100)
    # 方向：實際 [+, +, -, +]、預測 [+, -, -, +] → 3/4 正確
    assert m["dir_acc"] == pytest.approx(75.0)


def test_naive_direction_is_na():
    m = compute_metrics([100.0, 101.0], [99.0, 100.0], [99.0, 100.0], is_naive=True)
    assert math.isnan(m["dir_acc"])
    assert m["mae"] == pytest.approx(1.0)
