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
python main.py           # 完整實驗：資料 → EDA → 訓練 → 評估 → 印出比較表
python main.py --quick   # 快速測試：LSTM 最多 3 回合、隨機森林 20 棵樹，只用來確認流程可以跑完
python -m pytest -v      # 單元測試
```

所有參數（視窗長度、切分比例、特徵組、模型參數、SEED）都在 `config.py` 中修改。

## 方法摘要
- 任務：用過去 30 個交易日預測隔天收盤價（迴歸）。
- 特徵組：A = 收盤價；B = 收盤價、成交量、MA5、MA20。
- 切分：依時間順序分為訓練 70%／驗證 15%／測試 15%，不打亂；MinMaxScaler 只用訓練資料 fit。
- 模型：「明天＝今天」基準、線性迴歸、隨機森林、LSTM（PyTorch）。
- 指標（測試集，USD）：RMSE、MAE、MAPE(%)、方向正確率(%)。

## 輸出結果（`results/`）
| 檔案 | 內容 |
|---|---|
| `eda_close.png` | 收盤價走勢 |
| `eda_ma.png` | 收盤價與 MA5／MA20 |
| `eda_volume.png` | 成交量 |
| `eda_corr.png` | 特徵相關係數熱圖 |
| `eda_split.png` | 訓練／驗證／測試期間 |
| `lstm_loss_A.png`、`lstm_loss_B.png` | LSTM 訓練與驗證損失曲線 |
| `pred_A.png`、`pred_B.png` | 測試期實際值與各模型預測 |
| `metrics_bar.png` | 各模型與特徵組的 RMSE 比較 |
| `metrics.csv` | 欄位：feature_set、model、rmse、mae、mape(%)、dir_acc(%)；naive 的 dir_acc 為 N/A |

## 專案結構
```
config.py        所有路徑與參數
main.py          主程式
src/data.py      載入、清理、特徵、切分、標準化、視窗
src/eda.py       EDA 圖表
src/models.py    naive 基準、線性迴歸、隨機森林、LSTM
src/train.py     訓練所有模型
src/evaluate.py  指標、預測圖、比較表與圖
tests/           pytest 測試
```
