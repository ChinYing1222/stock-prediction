# 開發紀錄 DEV_LOG

## 決策總表
| 編號 | 日期 | 決策 | 理由 |
|---|---|---|---|
| D01 | 2026-11-23 | 以 Kaggle S&P 500 Stock Data 的 AAPL 為資料 | 公開資料、知名度高、欄位完整 |
| D02 | 2026-11-23 | 預測隔天收盤價（迴歸） | 最常見、評估指標直觀 |
| D03 | 2026-11-23 | 模型：LSTM、線性迴歸、隨機森林，另加「明天＝今天」基準 | 作業要求至少兩種模型；基準用來檢驗模型是否真的有學到東西 |
| D04 | 2026-11-23 | 特徵組 A（收盤價）與 B（收盤價＋成交量＋MA5＋MA20） | 比較特徵選擇的影響 |
| D05 | 2026-11-23 | 視窗 30 天 | 約一個半月的交易日，兼顧短期趨勢與樣本數 |
| D06 | 2026-11-23 | 依時間切分 70／15／15，不打亂 | 避免用未來資料預測過去 |
| D07 | 2026-11-23 | 標準化只用訓練資料計算範圍 | 避免資訊洩漏 |
| D08 | 2026-11-23 | 隨機森林維持預測價格，限制寫入分析 | 保持簡單；改預測漲跌幅列為改善方向 |
| D09 | 2026-11-23 | LSTM 使用 PyTorch | 沿用上一個專案熟悉的套件 |
| D10 | 2026-11-23 | 完整實驗由組員手動執行，Claude Code 只做快速測試 | 可自行掌控與觀察訓練 |
| D11 | 2026-11-23 | 只有 AAPL_data.csv 進 Git，GitHub 由組員最後自行建立與上傳 | 檔案小，clone 後即可執行 |

## 實驗數據總表
| 特徵組 | 模型 | RMSE | MAE | MAPE | 方向正確率 |
|---|---|---|---|---|---|

## 紀錄範本
```
### 日期／階段
- 目標：
- 架構決策與理由：
- 實作內容：
- 問題與排解：
- 實驗數據：
- 下一步：
```

---

### 2026-11-23／P0 環境與骨架
- 目標：建立 Python 3.11 虛擬環境、安裝套件、建立專案骨架與文件。
- 架構決策與理由：依決策總表 D01–D11；參數集中於 `config.py`，模組單向依賴 config → data → models → train/evaluate → main。
- 實作內容：`.venv`（Python 3.11.9）；安裝並固定版本 torch 2.15.0（CPU）、numpy 2.4.6、pandas 3.1.1、scikit-learn 1.9.1、matplotlib 3.11.2、pytest 9.2.0；`git init -b main`；建立 `config.py`、`CLAUDE.md`、`DEV_LOG.md`、`README.md`、`.gitignore`、`requirements.txt`、`src/__init__.py`、`results/.gitkeep`。
- 問題與排解：無。本機無 CUDA，DEVICE 自動為 cpu。
- 實驗數據：無。
- 下一步：P1 資料處理與 EDA。

### 2026-11-23／P1 資料處理與視覺化
- 目標：完成資料載入、清理、特徵工程、時間切分、標準化與視窗化，並產生 EDA 圖表。
- 架構決策與理由：`prepare_dataset()` 一次回傳某特徵組所需的全部陣列與 scaler，讓訓練與評估模組不必重複處理；scaler 只用「最後一個訓練目標日（含）以前」的資料列 fit（D07）。第 i 個樣本以第 i ~ i+29 列為輸入、第 i+30 列收盤價為目標。
- 實作內容：`src/data.py`（load_data、clean、add_features、load_clean_features、split_indices、make_windows、prepare_dataset）；`src/eda.py`（eda_close、eda_ma、eda_volume、eda_corr、eda_split 五張圖，Agg 後端、存檔後關閉）。
- 問題與排解：清理報告：1259 筆、期間 2013-02-08 ~ 2018-02-07、無缺失值、無重複日期、無單日變動 > 30%（資料已為分割調整後價格）。加入 MA 後剩 1240 筆、1210 個樣本。
- 實驗數據：樣本數 訓練 847／驗證 181／測試 182；視窗形狀 A=(N, 30, 1)、B=(N, 30, 4)；測試期 2017-05-19 ~ 2018-02-07。縮放後訓練目標最大值 = 1.0。觀察：測試期收盤價（約 145–180 USD）高於訓練期最高價（約 133 USD），預期隨機森林無法外推（D08），報告需討論。
- 下一步：P2 模型與訓練。

### 2026-11-23／P2 模型與訓練
- 目標：實作 naive 基準、線性迴歸、隨機森林、LSTM 及訓練流程。
- 架構決策與理由：`train_all()` 對單一特徵組訓練全部模型並回傳 USD 預測值，評估模組只需處理 USD 數值；naive 直接使用「今天收盤價」（USD），不經縮放；LSTM 取最後時間步輸出 → Dropout → Linear；DataLoader 只打亂視窗之間的順序（視窗內部仍為時間順序），並以 SEED 固定 generator；early stopping 依驗證損失、保留最佳權重。
- 實作內容：`src/models.py`（set_seed、naive_predict、flatten、fit_linear、fit_rf、predict_flat、LSTMRegressor）；`src/train.py`（train_lstm、lstm_predict、plot_loss、train_all），輸出 `lstm_loss_{A,B}.png`。
- 問題與排解：無。
- 實驗數據：檢查（特徵組 A、quick、LSTM 2 回合、RF 20 棵）：四個模型皆輸出 182 筆測試預測。隨機森林預測值全部約為 129.73 USD，證實無法外推至訓練期最高價以上（D08，報告需討論）。
- 下一步：P3 評估與主程式。

### 2026-11-23／P3 評估與主程式
- 目標：實作評估指標、預測圖、比較表／圖，以及串接全部流程的 `main.py`。
- 架構決策與理由：指標一律以 USD 在測試集計算；方向正確率以「今天收盤價」為比較基準，naive 基準記為 N/A；`metrics.csv` 中 mape 與 dir_acc 單位皆為 %；`--quick` 只減少 LSTM 回合數與樹的數量，流程與正式實驗完全相同。
- 實作內容：`src/evaluate.py`（compute_metrics、evaluate_all、save_metrics、format_table、print_table、plot_predictions、plot_metrics_bar）；`main.py`（argparse `--quick`）。
- 問題與排解：無。
- 實驗數據：`python main.py --quick` 約 5 秒完成，產出 11 個檔案（5 張 EDA 圖、lstm_loss_A/B、pred_A/B、metrics_bar、metrics.csv）。快速測試的數字僅用來確認流程，不列入實驗數據總表：Naive RMSE 2.056、線性迴歸 A 2.067／B 2.092、隨機森林 A 32.62／B 33.77（預測值固定在約 129.7 USD）、LSTM（僅 3 回合）A 5.89／B 8.27。
- 下一步：P4 測試、文件與清理。
