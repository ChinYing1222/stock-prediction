# CLAUDE.md

## 專案概述
「類神經網路商業應用」課程專案：以 LSTM 預測股價。使用 AAPL 過去 30 個交易日的資料，預測隔天收盤價（迴歸），並與線性迴歸、隨機森林及「明天＝今天」基準比較。
組員：M11523001 李季錞、M11523002 洪親穎。

## 作業要求對照
| 作業要求 | 對應實作 |
|---|---|
| (1) 資料蒐集與前處理（清理、轉換、特徵選擇、視覺化） | `src/data.py`、`src/eda.py`、特徵組 A／B |
| (2) 至少兩種 AI 模型並說明 | `src/models.py`：LSTM、線性迴歸、隨機森林（＋naive 基準） |
| (3) 完整可執行的 Python 程式並顯示結果 | `main.py` |
| (4) 訓練、測試與評估，以表格／圖表比較 | `src/train.py`、`src/evaluate.py`、`results/` |
| (5) 結果分析與具體改善方向 | `DEV_LOG.md`、書面報告 |

## 技術棧
Python 3.11、venv、PyTorch（CPU，自動偵測 cuda）、NumPy、pandas、scikit-learn、matplotlib、pytest。版本固定於 `requirements.txt`。

## 目錄結構與模組職責
```
stock-prediction/
├── CLAUDE.md, DEV_LOG.md, README.md, requirements.txt, .gitignore
├── config.py        所有路徑與參數
├── main.py          執行完整流程：資料 → EDA → 訓練 → 評估 → 印出比較表；--quick 快速測試
├── src/
│   ├── data.py      載入、清理、特徵、切分、標準化、建立視窗
│   ├── eda.py       原始資料視覺化
│   ├── models.py    naive 基準、線性迴歸、隨機森林、LSTM 類別
│   ├── train.py     針對一個特徵組訓練所有模型
│   └── evaluate.py  評估指標、預測圖、比較表與圖
├── tests/           pytest 基本測試
├── data/            AAPL_data.csv（唯讀）
└── results/         PNG 圖表與 metrics.csv
```

## 模組依賴規則
單向依賴：`config → data → models → train/evaluate → main`。下游模組不可被上游模組匯入。

## 固定決策
- 資料：Kaggle「S&P 500 Stock Data」的 `data/AAPL_data.csv`，唯讀；有 Name 欄位時篩選 `TICKER`。
- 任務：過去 30 個交易日 → 隔天收盤價。
- 清理：依日期排序、檢查缺失值與重複日期、標記單日變動 > 30% 的日子。
- 特徵：ma5、ma20，刪除前 19 筆；特徵組 A = close，B = close、volume、ma5、ma20。
- 依目標日時間切分 70／15／15，不跨時間打亂。
- MinMaxScaler 只用訓練資料 fit；收盤價目標使用獨立 scaler 以反轉回 USD。
- 模型：naive 基準、線性迴歸、隨機森林（200 棵，不改預測目標）、LSTM(64)→Dropout(0.2)→Linear；Adam 1e-3、MSE、batch 32、最多 100 回合、patience 10。
- 指標（測試集，USD）：RMSE、MAE、MAPE(%)、方向正確率（naive 為 N/A）。
- SEED = 42。
- 繪圖使用 Agg 後端、不呼叫 `plt.show()`、存檔後 `plt.close(fig)`；中文字型、dpi 150。

## 常用指令
```powershell
# 建立環境（第一次）
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# 完整實驗（組員執行）
.venv\Scripts\python main.py

# 快速測試（只檢查流程；結果不具參考價值，輸出不可 commit）
.venv\Scripts\python main.py --quick

# 單元測試
.venv\Scripts\python -m pytest -v
```

## 工作規則
- 一律以繁體中文回覆；文件與註解用繁中，程式識別字用英文。
- 先規劃後動工，等組員核准後才修改檔案。
- 每個階段結束更新 `DEV_LOG.md` 並 commit。
- 參數只改 `config.py`。
- 不得修改或刪除 `data/AAPL_data.csv`。
- 保持簡單（MVP）。
- Claude Code 只執行 `python main.py --quick`；完整實驗由組員執行。

## 開發階段
- [x] P0 環境與骨架
- [x] P1 資料處理與視覺化
- [x] P2 模型與訓練
- [x] P3 評估與主程式
- [x] P4 測試、文件與清理
