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
（P4 補齊）
