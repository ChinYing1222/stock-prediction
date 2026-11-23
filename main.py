"""主程式：資料 → EDA → 訓練 → 評估 → 印出比較表。

用法：
    python main.py                        執行實驗一與實驗二（預設 --experiment both）
    python main.py --experiment price     只執行實驗一：預測價格
    python main.py --experiment return    只執行實驗二：預測漲跌幅
    python main.py --quick                快速測試（LSTM 最多 3 回合、隨機森林 20 棵樹），
                                          輸出只寫到 results/quick/，不會覆寫完整實驗結果
"""
import argparse
import time

import numpy as np

import config
from src import evaluate
from src.data import load_clean_features, prepare_dataset, prepare_return_dataset, return_to_price
from src.eda import plot_returns, run_eda
from src.models import set_seed
from src.train import train_all


def run_price(df, quick):
    """實驗一：預測隔天收盤價（行為與輸出與原本相同）。"""
    datasets = {fs: prepare_dataset(df, fs) for fs in config.FEATURE_SETS}
    first = next(iter(datasets.values()))
    print(f"樣本數：訓練 {len(first['y_train'])}／驗證 {len(first['y_val'])}／測試 {len(first['y_test'])}")
    run_eda(df, first["train_end_row"], first["val_end_row"])

    rows = []
    for fs, data in datasets.items():
        print(f"\n===== 實驗一 特徵組 {fs}：{data['features']} =====")
        preds, _ = train_all(data, quick=quick)
        evaluate.plot_predictions(data, preds)
        rows.extend(evaluate.evaluate_all(data, preds))

    metrics = evaluate.save_metrics(rows)
    evaluate.plot_metrics_bar(metrics)
    evaluate.print_table(metrics, "實驗一：預測價格 測試集評估結果（USD）")


def run_return(df, quick):
    """實驗二：預測隔天報酬率，換回價格後用與實驗一相同的方式評估。"""
    plot_returns(df)
    price_test_dates = prepare_dataset(df, "A")["test_dates"]  # 實驗一的測試日期

    rows = []
    for fs in config.RETURN_FEATURE_SETS:
        data = prepare_return_dataset(df, fs)
        assert np.array_equal(data["test_dates"], price_test_dates), "實驗二的測試日期與實驗一不同"
        print(f"\n===== 實驗二 特徵組 {fs}：{data['features']} =====")
        pred_returns, _ = train_all(data, quick=quick)
        preds = {name: return_to_price(data["test_today_close"], r) for name, r in pred_returns.items()}
        evaluate.plot_predictions(data, preds)
        rows.extend(evaluate.evaluate_all(data, preds))

    suffix = config.RETURN_SUFFIX
    metrics = evaluate.save_metrics(rows, name=f"metrics{suffix}.csv")
    evaluate.plot_metrics_bar(metrics, name=f"metrics_bar{suffix}.png",
                              title="實驗二：預測漲跌幅 測試集 RMSE 比較（越低越好）")
    evaluate.print_table(metrics, "實驗二：預測漲跌幅 測試集評估結果（USD）")


def main():
    parser = argparse.ArgumentParser(description="LSTM 股價預測")
    parser.add_argument("--quick", action="store_true",
                        help="快速測試模式（僅檢查流程，結果不具參考價值；輸出寫到 results/quick/）")
    parser.add_argument("--experiment", choices=["price", "return", "both"], default="both",
                        help="price＝實驗一：預測價格；return＝實驗二：預測漲跌幅；both＝兩者（預設）")
    args = parser.parse_args()

    if args.quick:
        config.RESULTS_DIR = config.QUICK_RESULTS_DIR  # 快速測試不覆寫完整實驗的結果

    start = time.time()
    set_seed(config.SEED)
    print(f"裝置：{config.DEVICE}　模式：{'快速測試' if args.quick else '完整實驗'}　實驗：{args.experiment}")

    df = load_clean_features()
    if args.experiment in ("price", "both"):
        run_price(df, args.quick)
    if args.experiment in ("return", "both"):
        run_return(df, args.quick)

    evaluate.plot_compare_price_vs_return()
    print(f"\n輸出目錄：{config.RESULTS_DIR}")
    print(f"總耗時：{time.time() - start:.1f} 秒")


if __name__ == "__main__":
    main()
