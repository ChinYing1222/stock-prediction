"""資料處理：載入、清理、特徵、切分、標準化、建立視窗。"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import config


def load_data(path=config.DATA_PATH):
    """讀取 CSV；若有 Name 欄位則只保留 config.TICKER，並將 date 轉為日期型別。"""
    df = pd.read_csv(path)
    if "Name" in df.columns:
        df = df[df["Name"] == config.TICKER]
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.reset_index(drop=True)


def clean(df, verbose=True):
    """依日期排序、檢查缺失值與重複日期、標記單日收盤價大幅變動，並印出清理報告。"""
    df = df.sort_values("date").reset_index(drop=True)
    n_before = len(df)

    # 缺失值：回報各欄缺失數，並刪除價格或成交量缺失的列
    missing = df.isna().sum()
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    # 重複日期：保留第一筆
    n_dup = int(df["date"].duplicated().sum())
    df = df.drop_duplicates(subset="date", keep="first").reset_index(drop=True)

    # 單日收盤價變動超過門檻：可能是股票分割，只回報不修改
    pct = df["close"].pct_change().abs()
    alerts = df.loc[pct > config.SPLIT_ALERT_PCT, ["date", "close"]]

    if verbose:
        print("===== 資料清理報告 =====")
        print(f"原始筆數：{n_before}，清理後筆數：{len(df)}")
        print(f"期間：{df['date'].min().date()} ~ {df['date'].max().date()}")
        missing_nonzero = missing[missing > 0]
        if missing_nonzero.empty:
            print("缺失值：無")
        else:
            print("缺失值（已刪除該列）：" + "、".join(f"{k}={v}" for k, v in missing_nonzero.items()))
        print(f"重複日期：{n_dup} 筆" + ("（已保留第一筆）" if n_dup else ""))
        if alerts.empty:
            print(f"單日收盤價變動 > {config.SPLIT_ALERT_PCT:.0%}：無（未發現疑似股票分割）")
        else:
            print(f"單日收盤價變動 > {config.SPLIT_ALERT_PCT:.0%}（疑似股票分割）：")
            for _, row in alerts.iterrows():
                print(f"  {row['date'].date()} close={row['close']:.2f}")
    return df


def add_features(df):
    """新增 ma5、ma20 與實驗二的變化率特徵，並刪除前 MA_LONG-1 筆（移動平均不完整的列）。

    變化率特徵在刪除前計算，所以保留下來的第一列也有 ret 與 vol_chg。
    ma20_gap 同樣只在前 MA_LONG-1 筆未定義，因此實驗一與實驗二使用完全相同的資料列。
    """
    df = df.copy()
    df["ma5"] = df["close"].rolling(config.MA_SHORT).mean()
    df["ma20"] = df["close"].rolling(config.MA_LONG).mean()
    # 實驗二：變化率特徵（實驗一不使用，不影響實驗一）
    df["ret"] = df["close"].pct_change()
    df["vol_chg"] = np.log(df["volume"] / df["volume"].shift(1))
    df["ma5_gap"] = df["close"] / df["ma5"] - 1
    df["ma20_gap"] = df["close"] / df["ma20"] - 1
    return df.iloc[config.MA_LONG - 1:].reset_index(drop=True)


def load_clean_features(verbose=True):
    """一次完成載入、清理與特徵工程。"""
    return add_features(clean(load_data(), verbose=verbose))


def split_indices(n_samples):
    """依目標日時間順序切分樣本，回傳 (訓練樣本數, 驗證樣本數)；剩餘為測試。"""
    n_train = int(n_samples * config.TRAIN_RATIO)
    n_val = int(n_samples * config.VAL_RATIO)
    return n_train, n_val


def make_windows(values, target, window=config.WINDOW):
    """以過去 window 天的特徵預測下一天的目標。

    values: (n_rows, n_features)；target: (n_rows,)
    回傳 X: (n_rows - window, window, n_features)，y: (n_rows - window,)
    第 i 個樣本的輸入為第 i ~ i+window-1 列，目標為第 i+window 列。
    """
    X = np.stack([values[i:i + window] for i in range(len(values) - window)])
    y = np.asarray(target[window:])
    return X.astype(np.float32), y.astype(np.float32)


def prepare_dataset(df, feature_set, window=config.WINDOW):
    """針對一個特徵組，回傳訓練／驗證／測試的 X、y，scaler，以及測試期的日期與 USD 價格。

    標準化只用「最後一個訓練目標日（含）以前」的資料列 fit，避免資訊洩漏。
    """
    cols = config.FEATURE_SETS[feature_set]
    values = df[cols].to_numpy(dtype=float)
    close = df[config.TARGET_COL].to_numpy(dtype=float)

    n_samples = len(df) - window
    n_train, n_val = split_indices(n_samples)
    train_end_row = window + n_train  # 訓練資料列的結束位置（不含）
    val_end_row = train_end_row + n_val  # 驗證資料列的結束位置（不含）

    feature_scaler = MinMaxScaler().fit(values[:train_end_row])
    target_scaler = MinMaxScaler().fit(close[:train_end_row].reshape(-1, 1))
    scaled_values = feature_scaler.transform(values)
    scaled_close = target_scaler.transform(close.reshape(-1, 1)).ravel()

    X, y = make_windows(scaled_values, scaled_close, window)
    tr = slice(0, n_train)
    va = slice(n_train, n_train + n_val)
    te = slice(n_train + n_val, n_samples)

    today_close = close[window - 1:-1]  # 每個樣本「今天」（視窗最後一天）的收盤價
    actual = close[window:]  # 每個樣本隔天的實際收盤價
    dates = df["date"].to_numpy()[window:]

    return {
        "feature_set": feature_set,
        "target": "price",
        "features": cols,
        "X_train": X[tr], "y_train": y[tr],
        "X_val": X[va], "y_val": y[va],
        "X_test": X[te], "y_test": y[te],
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "test_dates": dates[te],
        "test_today_close": today_close[te],
        "test_actual": actual[te],
        "train_end_row": train_end_row,
        "val_end_row": val_end_row,
    }


def prepare_return_dataset(df, feature_set, window=config.WINDOW):
    """實驗二：輸入為變化率特徵，目標為隔天報酬率 r(t+1) = close(t+1) / close(t) - 1。

    使用與實驗一相同的資料列、視窗與切分，所以目標日期（含 182 個測試日）完全相同。
    StandardScaler 只用「最後一個訓練目標日（含）以前」的資料列 fit；目標另用一個 scaler。
    回傳的鍵與 prepare_dataset 相同（test_actual、test_today_close 仍為 USD），
    另加 test_actual_return（隔天實際報酬率）。
    """
    cols = config.RETURN_FEATURE_SETS[feature_set]
    values = df[cols].to_numpy(dtype=float)
    ret = df[config.RETURN_COL].to_numpy(dtype=float)
    close = df[config.TARGET_COL].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"特徵組 {feature_set} 含有未定義的值（NaN 或 inf）")

    n_samples = len(df) - window
    n_train, n_val = split_indices(n_samples)
    train_end_row = window + n_train
    val_end_row = train_end_row + n_val

    feature_scaler = StandardScaler().fit(values[:train_end_row])
    target_scaler = StandardScaler().fit(ret[:train_end_row].reshape(-1, 1))
    scaled_values = feature_scaler.transform(values)
    scaled_ret = target_scaler.transform(ret.reshape(-1, 1)).ravel()

    X, y = make_windows(scaled_values, scaled_ret, window)
    tr = slice(0, n_train)
    va = slice(n_train, n_train + n_val)
    te = slice(n_train + n_val, n_samples)

    today_close = close[window - 1:-1]
    actual = close[window:]
    dates = df["date"].to_numpy()[window:]

    return {
        "feature_set": feature_set,
        "target": "return",
        "features": cols,
        "X_train": X[tr], "y_train": y[tr],
        "X_val": X[va], "y_val": y[va],
        "X_test": X[te], "y_test": y[te],
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "test_dates": dates[te],
        "test_today_close": today_close[te],
        "test_actual": actual[te],
        "test_actual_return": ret[window:][te],
        "train_end_row": train_end_row,
        "val_end_row": val_end_row,
    }


def return_to_price(today_close, ret):
    """把預測的報酬率換回價格：pred_price = today_close * (1 + ret)。"""
    return np.asarray(today_close, dtype=float) * (1 + np.asarray(ret, dtype=float))
