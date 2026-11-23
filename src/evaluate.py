"""評估：指標計算、預測圖、比較表與比較圖。"""
import matplotlib

matplotlib.use("Agg")  # 非互動式後端，不開視窗
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402

plt.rcParams["font.sans-serif"] = config.FONT_FAMILY
plt.rcParams["axes.unicode_minus"] = False

METRIC_COLUMNS = ["feature_set", "model", "rmse", "mae", "mape", "dir_acc"]


def compute_metrics(actual, pred, today_close, is_naive=False):
    """計算 RMSE、MAE、MAPE(%)、方向正確率(%)，皆以 USD 計算。

    方向正確率 = sign(預測 - 今天收盤) 與 sign(實際 - 今天收盤) 相同的天數比例；
    naive 基準預測「不變」，方向正確率無意義，回傳 NaN（顯示為 N/A）。
    """
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    today_close = np.asarray(today_close, dtype=float)
    err = pred - actual
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / actual)) * 100)
    if is_naive:
        dir_acc = float("nan")
    else:
        dir_acc = float(np.mean(np.sign(pred - today_close) == np.sign(actual - today_close)) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape, "dir_acc": dir_acc}


def evaluate_all(data, preds):
    """對一個特徵組的所有模型計算指標，回傳 list[dict]。"""
    rows = []
    for name, pred in preds.items():
        m = compute_metrics(data["test_actual"], pred, data["test_today_close"], is_naive=(name == "Naive"))
        rows.append({"feature_set": data["feature_set"], "model": name, **m})
    return rows


def save_metrics(rows):
    """將所有指標存成 results/metrics.csv，回傳 DataFrame。"""
    df = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_DIR / "metrics.csv", index=False, float_format="%.4f", na_rep="N/A")
    return df


def format_table(df):
    """回傳格式化的比較表字串。"""
    out = df.copy()
    out["rmse"] = out["rmse"].map("{:.3f}".format)
    out["mae"] = out["mae"].map("{:.3f}".format)
    out["mape"] = out["mape"].map("{:.2f}%".format)
    out["dir_acc"] = out["dir_acc"].map(lambda v: "N/A" if pd.isna(v) else f"{v:.2f}%")
    out.columns = ["特徵組", "模型", "RMSE", "MAE", "MAPE", "方向正確率"]
    return out.to_string(index=False)


def print_table(df):
    """印出比較表。"""
    print("\n===== 測試集評估結果（USD）=====")
    print(format_table(df))


def _save(fig, name):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=config.DPI)
    plt.close(fig)
    return path


def plot_predictions(data, preds):
    """測試期實際收盤價與各模型預測，存成 pred_{feature_set}.png。"""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(data["test_dates"], data["test_actual"], color="black", linewidth=2, label="實際")
    for name, pred in preds.items():
        style = "--" if name == "Naive" else "-"
        ax.plot(data["test_dates"], pred, linestyle=style, linewidth=1, label=name)
    ax.set_title(f"測試期預測比較（特徵組 {data['feature_set']}）")
    ax.set_xlabel("日期")
    ax.set_ylabel("收盤價（USD）")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, f"pred_{data['feature_set']}.png")


def plot_metrics_bar(df):
    """各模型在各特徵組的 RMSE 長條圖，存成 metrics_bar.png。"""
    model_names = list(dict.fromkeys(df["model"]))
    feature_sets = list(dict.fromkeys(df["feature_set"]))
    x = np.arange(len(model_names))
    width = 0.8 / len(feature_sets)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, fs in enumerate(feature_sets):
        sub = df[df["feature_set"] == fs].set_index("model").reindex(model_names)
        bars = ax.bar(x + (i - (len(feature_sets) - 1) / 2) * width, sub["rmse"], width, label=f"特徵組 {fs}")
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, labels=model_names)
    ax.set_title("測試集 RMSE 比較（越低越好）")
    ax.set_ylabel("RMSE（USD）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "metrics_bar.png")
