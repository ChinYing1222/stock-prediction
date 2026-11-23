"""主程式：資料 → EDA → 訓練 → 評估 → 印出比較表。

用法：
    python main.py          完整實驗
    python main.py --quick  快速測試（LSTM 最多 3 回合、隨機森林 20 棵樹）
"""
import argparse
import time

import config
from src import evaluate
from src.data import load_clean_features, prepare_dataset
from src.eda import run_eda
from src.models import set_seed
from src.train import train_all


def main():
    parser = argparse.ArgumentParser(description="LSTM 股價預測")
    parser.add_argument("--quick", action="store_true", help="快速測試模式（僅檢查流程，結果不具參考價值）")
    args = parser.parse_args()

    start = time.time()
    set_seed(config.SEED)
    print(f"裝置：{config.DEVICE}　模式：{'快速測試' if args.quick else '完整實驗'}")

    # 1. 資料載入、清理與特徵
    df = load_clean_features()

    # 2. 各特徵組的資料集
    datasets = {fs: prepare_dataset(df, fs) for fs in config.FEATURE_SETS}
    first = next(iter(datasets.values()))
    print(f"樣本數：訓練 {len(first['y_train'])}／驗證 {len(first['y_val'])}／測試 {len(first['y_test'])}")

    # 3. EDA 圖表
    run_eda(df, first["train_end_row"], first["val_end_row"])

    # 4. 訓練與評估
    rows = []
    for fs, data in datasets.items():
        print(f"\n===== 特徵組 {fs}：{data['features']} =====")
        preds, _ = train_all(data, quick=args.quick)
        evaluate.plot_predictions(data, preds)
        rows.extend(evaluate.evaluate_all(data, preds))

    # 5. 比較表與比較圖
    metrics = evaluate.save_metrics(rows)
    evaluate.plot_metrics_bar(metrics)
    evaluate.print_table(metrics)
    print(f"\n輸出目錄：{config.RESULTS_DIR}")
    print(f"總耗時：{time.time() - start:.1f} 秒")


if __name__ == "__main__":
    main()
