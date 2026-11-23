"""訓練：針對一個特徵組訓練所有模型，回傳測試集的 USD 預測值。"""
import copy

import matplotlib

matplotlib.use("Agg")  # 非互動式後端，不開視窗
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import config  # noqa: E402
from src import models  # noqa: E402

plt.rcParams["font.sans-serif"] = config.FONT_FAMILY
plt.rcParams["axes.unicode_minus"] = False


def _to_tensor(a):
    return torch.as_tensor(a, dtype=torch.float32, device=config.DEVICE)


def train_lstm(data, max_epochs=config.LSTM_MAX_EPOCHS, patience=config.LSTM_PATIENCE, verbose=True):
    """訓練 LSTM，使用驗證損失做 early stopping，保留最佳權重。

    回傳 (model, history)，history = {"train": [...], "val": [...], "best_epoch": int}。
    訓練時只打亂「視窗之間」的順序，每個視窗內部仍是時間順序，不會造成資訊洩漏。
    """
    models.set_seed()
    n_features = data["X_train"].shape[2]
    model = models.LSTMRegressor(n_features).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LSTM_LR)
    loss_fn = nn.MSELoss()

    generator = torch.Generator().manual_seed(config.SEED)
    train_ds = TensorDataset(torch.as_tensor(data["X_train"]), torch.as_tensor(data["y_train"]))
    loader = DataLoader(train_ds, batch_size=config.LSTM_BATCH_SIZE, shuffle=True, generator=generator)
    X_val, y_val = _to_tensor(data["X_val"]), _to_tensor(data["y_val"])

    history = {"train": [], "val": [], "best_epoch": 0}
    best_val, best_state, wait = float("inf"), None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        train_loss = total / len(train_ds)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val, wait = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
        else:
            wait += 1

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"  [LSTM {data['feature_set']}] epoch {epoch:3d}  train={train_loss:.6f}  val={val_loss:.6f}")
        if wait >= patience:
            if verbose:
                print(f"  [LSTM {data['feature_set']}] early stopping 於 epoch {epoch}")
            break

    model.load_state_dict(best_state)
    if verbose:
        print(f"  [LSTM {data['feature_set']}] 最佳 epoch {history['best_epoch']}，val loss={best_val:.6f}")
    return model, history


def lstm_predict(model, X):
    """LSTM 對視窗資料預測（輸出為縮放後的數值）。"""
    model.eval()
    with torch.no_grad():
        return model(_to_tensor(X)).cpu().numpy()


def plot_loss(history, feature_set):
    """畫出 LSTM 每回合的訓練／驗證損失，存成 lstm_loss_{feature_set}.png。"""
    epochs = np.arange(1, len(history["train"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train"], label="訓練損失")
    ax.plot(epochs, history["val"], label="驗證損失")
    ax.axvline(history["best_epoch"], color="gray", linestyle="--", label=f"最佳 epoch {history['best_epoch']}")
    ax.set_title(f"LSTM 損失曲線（特徵組 {feature_set}）")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE（縮放後）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"lstm_loss_{feature_set}.png"
    fig.savefig(path, dpi=config.DPI)
    plt.close(fig)
    return path


def train_all(data, quick=False, verbose=True):
    """針對一個特徵組訓練所有模型，回傳 (預測 dict, LSTM history)。

    預測 dict 的鍵為模型名稱，值為測試集預測值（已反轉換回目標單位）：
    實驗一（data["target"] == "price"）為 USD 價格；實驗二（"return"）為隔天報酬率。
    quick=True 時 LSTM 最多 3 回合、隨機森林 20 棵樹，僅供快速測試。
    """
    def inverse(pred_scaled):
        """縮放後的預測值反轉換回 USD。"""
        return data["target_scaler"].inverse_transform(np.asarray(pred_scaled).reshape(-1, 1)).ravel()

    X_tr, y_tr, X_te = data["X_train"], data["y_train"], data["X_test"]
    n_trees = config.QUICK_RF_N_ESTIMATORS if quick else config.RF_N_ESTIMATORS
    max_epochs = config.QUICK_LSTM_EPOCHS if quick else config.LSTM_MAX_EPOCHS

    if data["target"] == "return":
        preds = {"Naive": models.naive_return_predict(len(X_te))}
    else:
        preds = {"Naive": models.naive_predict(data["test_today_close"])}

    if verbose:
        print(f"[{data['feature_set']}] 訓練線性迴歸")
    preds["LinearRegression"] = inverse(models.predict_flat(models.fit_linear(X_tr, y_tr), X_te))

    if verbose:
        print(f"[{data['feature_set']}] 訓練隨機森林（{n_trees} 棵樹）")
    preds["RandomForest"] = inverse(models.predict_flat(models.fit_rf(X_tr, y_tr, n_trees), X_te))

    if verbose:
        print(f"[{data['feature_set']}] 訓練 LSTM（最多 {max_epochs} 回合，裝置 {config.DEVICE}）")
    lstm, history = train_lstm(data, max_epochs=max_epochs, verbose=verbose)
    preds["LSTM"] = inverse(lstm_predict(lstm, X_te))
    plot_loss(history, data["feature_set"])

    return preds, history
