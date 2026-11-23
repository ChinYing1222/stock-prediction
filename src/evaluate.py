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
    """對一個特徵組的所有模型計算指標，回傳 list[dict]。

    preds 必須是 USD 價格。實驗二先用 return_to_price 換回價格；因為 today_close > 0，
    sign(預測價格 - 今天收盤) = sign(預測報酬率)，所以方向正確率與「報酬率正負號」的定義相同。
    """
    rows = []
    for name, pred in preds.items():
        m = compute_metrics(data["test_actual"], pred, data["test_today_close"], is_naive=(name == "Naive"))
        rows.append({"feature_set": data["feature_set"], "model": name, **m})
    return rows


def save_metrics(rows, name="metrics.csv"):
    """將所有指標存成 results/{name}（預設 metrics.csv），回傳 DataFrame。"""
    df = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_DIR / name, index=False, float_format="%.4f", na_rep="N/A")
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


def print_table(df, title="測試集評估結果（USD）"):
    """印出比較表。"""
    print(f"\n===== {title} =====")
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


def plot_metrics_bar(df, name="metrics_bar.png", title="測試集 RMSE 比較（越低越好）"):
    """各模型在各特徵組的 RMSE 長條圖，存成 results/{name}（預設 metrics_bar.png）。"""
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
    ax.set_title(title)
    ax.set_ylabel("RMSE（USD）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, name)


def plot_compare_price_vs_return():
    """實驗一（預測價格）vs 實驗二（預測漲跌幅）的測試集 RMSE 比較，存成 compare_price_vs_return.png。

    讀取 results/ 下的 metrics.csv 與 metrics_ret.csv；任一個不存在就印出提示並跳過，回傳 None。
    """
    path_price = config.RESULTS_DIR / "metrics.csv"
    path_return = config.RESULTS_DIR / f"metrics{config.RETURN_SUFFIX}.csv"
    missing = [p.name for p in (path_price, path_return) if not p.exists()]
    if missing:
        print(f"略過 compare_price_vs_return.png：找不到 {'、'.join(missing)}")
        return None

    price = pd.read_csv(path_price)
    ret = pd.read_csv(path_return)
    ret["feature_set"] = ret["feature_set"].str.removesuffix(config.RETURN_SUFFIX)  # A_ret → A
    feature_sets = list(dict.fromkeys(price["feature_set"]))
    model_names = list(dict.fromkeys(price["model"]))
    x = np.arange(len(model_names))
    width = 0.4

    fig, axes = plt.subplots(1, len(feature_sets), figsize=(12, 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fs in zip(axes, feature_sets):
        for i, (label, df) in enumerate([("實驗一：預測價格", price), ("實驗二：預測漲跌幅", ret)]):
            sub = df[df["feature_set"] == fs].set_index("model").reindex(model_names)
            bars = ax.bar(x + (i - 0.5) * width, sub["rmse"], width, label=label)
            ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.set_xticks(x, labels=model_names)
        ax.set_title(f"特徵組 {fs}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("RMSE（USD）")
    axes[0].legend()
    fig.suptitle("測試集 RMSE：預測價格 vs 預測漲跌幅（越低越好）")
    return _save(fig, "compare_price_vs_return.png")
