# LSTM 股價預測（類神經網路商業應用 課程專案）

組員：M11523001 李季錞、M11523002 洪親穎

## 專案目的
以 AAPL 歷史股價資料，使用過去 30 個交易日預測隔天收盤價，比較 LSTM、線性迴歸、隨機森林與「明天＝今天」基準的表現，並分析特徵選擇（只用收盤價 vs. 加入成交量與移動平均）的影響。

## 環境建置（Windows PowerShell）
需要 Python 3.11。

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

若執行 `Activate.ps1` 時出現「無法載入，因為這個系統上已停用指令碼執行」，請先執行：

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## 資料來源
Kaggle「S&P 500 Stock Data」（camnugent/sandp500）中的 `AAPL_data.csv`，期間 2013-02 至 2018-02，欄位：date、open、high、low、close、volume、Name。
請將檔案放在 `data/AAPL_data.csv`（本專案已附上）。

## 執行方式
先啟用虛擬環境（`.venv\Scripts\Activate.ps1`），在專案根目錄執行：

```powershell
python main.py                                # 執行實驗一與實驗二（預設 --experiment both）
python main.py --experiment price             # 只執行實驗一：預測價格
python main.py --experiment return            # 只執行實驗二：預測漲跌幅
python main.py --quick                        # 快速測試：LSTM 最多 3 回合、隨機森林 20 棵樹
python main.py --quick --experiment return    # 只對實驗二做快速測試
python -m pytest -v                           # 單元測試
```

- `--quick` 只用來確認流程可以跑完，結果不具參考價值；輸出寫到 `results/quick/`（不進 Git），**不會覆寫** `results/` 中的完整實驗結果。
- 所有參數（視窗長度、切分比例、特徵組、模型參數、SEED）都在 `config.py` 中修改。

## 方法摘要
共同設定：
- 用過去 30 個交易日預測隔天，依時間順序切成訓練 70%／驗證 15%／測試 15%（不打亂）。兩個實驗使用相同的 182 個測試日（2017-05-19 ~ 2018-02-07）。
- 模型：「明天＝今天」基準、線性迴歸、隨機森林（200 棵）、LSTM（PyTorch，64 個單元）。兩個實驗的超參數相同。
- 指標（測試集，USD）：RMSE、MAE、MAPE(%)、方向正確率(%)。

| | 實驗一：預測價格 | 實驗二：預測漲跌幅 |
|---|---|---|
| 預測目標 | 隔天收盤價 | 隔天報酬率 r(t+1) = close(t+1) / close(t) − 1 |
| 特徵組 A | close | A_ret：ret（日報酬率） |
| 特徵組 B | close、volume、ma5、ma20 | B_ret：ret、vol_chg = log(volume / 前一天 volume)、ma5_gap = close / ma5 − 1、ma20_gap = close / ma20 − 1 |
| 標準化 | MinMaxScaler（只用訓練資料 fit） | StandardScaler（只用訓練資料 fit） |
| 基準 | 預測 = 今天收盤價 | 預測報酬率 = 0（與實驗一的基準相同） |
| 換回價格 | 不需要 | 預測價格 = 今天收盤價 × (1 + 預測報酬率) |

為什麼要做實驗二：實驗一的測試期價格高於訓練期的最高價，隨機森林無法外推，LSTM 的輸入也超出 MinMax 的範圍，因此兩者都失準。變化率不受價格水準影響，可以避免這個問題（詳見 `DEV_LOG.md`）。

## 輸出結果（`results/`）
| 檔案 | 內容 |
|---|---|
| `eda_close.png`、`eda_ma.png`、`eda_volume.png` | 收盤價、移動平均、成交量 |
| `eda_corr.png` | 特徵相關係數熱圖 |
| `eda_split.png` | 訓練／驗證／測試期間 |
| `eda_returns.png` | 日報酬率時間序列與分布直方圖 |
| `lstm_loss_A.png`、`lstm_loss_B.png`／`lstm_loss_A_ret.png`、`lstm_loss_B_ret.png` | LSTM 訓練與驗證損失曲線（實驗一／實驗二） |
| `pred_A.png`、`pred_B.png`／`pred_A_ret.png`、`pred_B_ret.png` | 測試期實際值與各模型預測（實驗一／實驗二） |
| `metrics_bar.png`／`metrics_bar_ret.png` | 各模型與特徵組的 RMSE 比較（實驗一／實驗二） |
| `metrics.csv`／`metrics_ret.csv` | 欄位：feature_set、model、rmse、mae、mape(%)、dir_acc(%)；naive 的 dir_acc 為 N/A |
| `compare_price_vs_return.png` | 實驗一 vs 實驗二的測試集 RMSE（兩個 CSV 都存在時才會產生） |

## 專案結構
```
config.py        所有路徑與參數
main.py          主程式（--experiment、--quick）
src/data.py      載入、清理、特徵（含變化率）、切分、標準化、視窗、報酬率換回價格
src/eda.py       EDA 圖表
src/models.py    naive 基準、線性迴歸、隨機森林、LSTM
src/train.py     訓練所有模型
src/evaluate.py  指標、預測圖、比較表與圖
tests/           pytest 測試
```
