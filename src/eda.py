"""探索性資料分析（EDA）：原始資料視覺化。"""
import matplotlib

matplotlib.use("Agg")  # 非互動式後端，不開視窗
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402

plt.rcParams["font.sans-serif"] = config.FONT_FAMILY
plt.rcParams["axes.unicode_minus"] = False


def _save(fig, name):
    """存檔並關閉圖表，回傳檔案路徑。"""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=config.DPI)
    plt.close(fig)
    return path


def plot_close(df):
    """收盤價走勢。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["date"], df["close"], color="tab:blue")
    ax.set_title(f"{config.TICKER} 收盤價")
    ax.set_xlabel("日期")
    ax.set_ylabel("收盤價（USD）")
    ax.grid(alpha=0.3)
    return _save(fig, "eda_close.png")


def plot_ma(df):
    """收盤價與 MA5、MA20。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["date"], df["close"], label="收盤價", linewidth=1)
    ax.plot(df["date"], df["ma5"], label="MA5", linewidth=1)
    ax.plot(df["date"], df["ma20"], label="MA20", linewidth=1)
    ax.set_title(f"{config.TICKER} 收盤價與移動平均")
    ax.set_xlabel("日期")
    ax.set_ylabel("價格（USD）")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, "eda_ma.png")


def plot_volume(df):
    """每日成交量。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["date"], df["volume"], width=1.0, color="tab:gray")
    ax.set_title(f"{config.TICKER} 成交量")
    ax.set_xlabel("日期")
    ax.set_ylabel("成交量（股）")
    ax.grid(alpha=0.3)
    return _save(fig, "eda_volume.png")


def plot_corr(df):
    """特徵組 B 各特徵的相關係數熱圖。"""
    cols = config.FEATURE_SETS["B"]
    corr = df[cols].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(cols)), labels=cols)
    ax.set_yticks(np.arange(len(cols)), labels=cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    ax.set_title("特徵相關係數")
    return _save(fig, "eda_corr.png")


def plot_split(df, train_end_row, val_end_row):
    """以顏色標示訓練／驗證／測試期間的收盤價。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    parts = [
        ("訓練", slice(0, train_end_row), "tab:blue"),
        ("驗證", slice(train_end_row, val_end_row), "tab:orange"),
        ("測試", slice(val_end_row, len(df)), "tab:green"),
    ]
    for label, sl, color in parts:
        part = df.iloc[sl]
        ax.plot(part["date"], part["close"], color=color, label=f"{label}（{len(part)} 筆）")
    ax.set_title("資料切分（依時間順序）")
    ax.set_xlabel("日期")
    ax.set_ylabel("收盤價（USD）")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, "eda_split.png")


def run_eda(df, train_end_row, val_end_row):
    """產生所有 EDA 圖表，回傳檔案路徑清單。"""
    return [
        plot_close(df),
        plot_ma(df),
        plot_volume(df),
        plot_corr(df),
        plot_split(df, train_end_row, val_end_row),
    ]
