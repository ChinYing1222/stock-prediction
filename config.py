"""專案設定：所有路徑與參數集中於此，修改參數只改這個檔案。"""
from pathlib import Path

import torch

# ===== 路徑 =====
BASE_DIR = Path(__file__).resolve().parent  # 專案根目錄
DATA_PATH = BASE_DIR / "data" / "AAPL_data.csv"  # 原始資料（唯讀，不可修改）
RESULTS_DIR = BASE_DIR / "results"  # 圖表與 metrics.csv 輸出目錄

# ===== 資料 =====
TICKER = "AAPL"  # 股票代號；若 CSV 有 Name 欄位則只保留此代號
WINDOW = 30  # 用過去 30 個交易日預測隔天收盤價
TRAIN_RATIO = 0.70  # 訓練集比例（依目標日時間順序）
VAL_RATIO = 0.15  # 驗證集比例；剩餘 15% 為測試集
MA_SHORT = 5  # 短期移動平均天數（ma5）
MA_LONG = 20  # 長期移動平均天數（ma20），前 MA_LONG-1 筆會被刪除
SPLIT_ALERT_PCT = 0.30  # 單日收盤價變動超過 30% 視為可能的股票分割並回報
TARGET_COL = "close"  # 預測目標欄位

# 特徵組：A 只用收盤價；B 加入成交量與移動平均
FEATURE_SETS = {
    "A": ["close"],
    "B": ["close", "volume", "ma5", "ma20"],
}

# ===== 模型參數 =====
SEED = 42  # 隨機種子（random、numpy、torch 共用）
RF_N_ESTIMATORS = 200  # 隨機森林樹的數量
LSTM_HIDDEN = 64  # LSTM 隱藏層大小
LSTM_LAYERS = 1  # LSTM 層數
LSTM_DROPOUT = 0.2  # LSTM 輸出後的 Dropout 比例
LSTM_LR = 1e-3  # Adam 學習率
LSTM_BATCH_SIZE = 32  # 批次大小
LSTM_MAX_EPOCHS = 100  # 最大訓練回合數
LSTM_PATIENCE = 10  # Early stopping：驗證損失連續幾回合未改善就停止

# ===== 快速測試（--quick）=====
QUICK_LSTM_EPOCHS = 3  # 快速測試時 LSTM 最多訓練 3 回合
QUICK_RF_N_ESTIMATORS = 20  # 快速測試時隨機森林只用 20 棵樹

# ===== 運算裝置 =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 有 GPU 用 cuda，否則用 cpu

# ===== 繪圖 =====
FONT_FAMILY = ["Microsoft JhengHei", "Noto Sans CJK TC", "DejaVu Sans"]  # 中文字型優先順序
DPI = 150  # 圖檔解析度
