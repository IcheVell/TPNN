import os
import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE_PATH = "Steel_industry_data.csv"
TARGET_COLUMN = "Usage_kWh"

TIME_STEPS = 24
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1

EPOCHS = 30
BATCH_SIZE = 64
HIDDEN_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 5
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
    df = df.sort_values("date").reset_index(drop=True)

    categorical_cols = ["WeekStatus", "Day_of_week", "Load_Type"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    return df


def split_train_test_by_time(df: pd.DataFrame, train_size: float = 0.8):
    split_index = int(len(df) * train_size)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


class MinMaxScalerCustom:
    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.range_ = None

    def fit(self, x: np.ndarray):
        self.min_ = x.min(axis=0)
        self.max_ = x.max(axis=0)
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ == 0] = 1.0

    def transform(self, x: np.ndarray):
        return (x - self.min_) / self.range_

    def fit_transform(self, x: np.ndarray):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x: np.ndarray):
        return x * self.range_ + self.min_


def fit_transform_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str):
    feature_columns = [col for col in train_df.columns if col != "date"]

    scaler = MinMaxScalerCustom()
    train_scaled = scaler.fit_transform(train_df[feature_columns].values.astype(np.float64))
    test_scaled = scaler.transform(test_df[feature_columns].values.astype(np.float64))

    target_index = feature_columns.index(target_column)

    return train_scaled, test_scaled, scaler, feature_columns, target_index


def create_train_sequences(train_scaled: np.ndarray, target_index: int, time_steps: int):
    x_train, y_train = [], []
    for i in range(time_steps, len(train_scaled)):
        x_train.append(train_scaled[i - time_steps:i, :])
        y_train.append(train_scaled[i, target_index])
    return np.array(x_train), np.array(y_train).reshape(-1, 1)


def create_test_sequences(train_scaled: np.ndarray, test_scaled: np.ndarray, target_index: int, time_steps: int):
    combined = np.vstack([train_scaled[-time_steps:], test_scaled])
    x_test, y_test = [], []
    for i in range(time_steps, len(combined)):
        x_test.append(combined[i - time_steps:i, :])
        y_test.append(combined[i, target_index])
    return np.array(x_test), np.array(y_test).reshape(-1, 1)


def train_val_split_sequences(x_train: np.ndarray, y_train: np.ndarray, val_size: float = 0.1):
    split_index = int(len(x_train) * (1 - val_size))
    x_tr = x_train[:split_index]
    y_tr = y_train[:split_index]
    x_val = x_train[split_index:]
    y_val = y_train[split_index:]
    return x_tr, y_tr, x_val, y_val

def mse_loss(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def calculate_metrics(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    mask = y_true != 0
    if np.any(mask):
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float("nan")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "R2": r2
    }


def inverse_transform_target(y_scaled, scaler: MinMaxScalerCustom, feature_count: int, target_index: int):
    temp = np.zeros((len(y_scaled), feature_count), dtype=np.float64)
    temp[:, target_index] = y_scaled.reshape(-1)
    restored = scaler.inverse_transform(temp)
    return restored[:, target_index].reshape(-1, 1)


class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        for key in params.keys():
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class BaseSequenceRegressor:
    def __init__(self, input_size, hidden_size, learning_rate=0.001, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)
        self.params = {}
        self.optimizer = AdamOptimizer(lr=learning_rate)
        self._init_params()

    def _xavier(self, fan_in, fan_out):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return self.rng.uniform(-limit, limit, size=(fan_in, fan_out))

    def _init_params(self):
        raise NotImplementedError

    def forward(self, x_batch):
        raise NotImplementedError

    def backward(self, cache, y_true):
        raise NotImplementedError

    def predict_scaled(self, x):
        y_pred, _ = self.forward(x)
        return y_pred

    def _iterate_minibatches(self, x, y, batch_size, shuffle=True):
        indices = np.arange(len(x))
        if shuffle:
            self.rng.shuffle(indices)

        for start in range(0, len(x), batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield x[batch_idx], y[batch_idx]

    def fit(self, x_train, y_train, x_val, y_val, epochs=30, batch_size=64, patience=5, verbose=1):
        history = {"loss": [], "val_loss": []}
        best_val = float("inf")
        best_params = copy.deepcopy(self.params)
        no_improve = 0

        for epoch in range(1, epochs + 1):
            batch_losses = []

            for x_batch, y_batch in self._iterate_minibatches(x_train, y_train, batch_size=batch_size, shuffle=True):
                y_pred, cache = self.forward(x_batch)
                loss = mse_loss(y_batch, y_pred)
                grads = self.backward(cache, y_batch)
                self.optimizer.update(self.params, grads)
                batch_losses.append(loss)

            train_pred = self.predict_scaled(x_train)
            val_pred = self.predict_scaled(x_val)

            train_loss = mse_loss(y_train, train_pred)
            val_loss = mse_loss(y_val, val_pred)

            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if verbose:
                print(f"Epoch {epoch}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                best_params = copy.deepcopy(self.params)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        self.params = best_params
        return history

class SimpleRNNRegressor(BaseSequenceRegressor):
    def _init_params(self):
        h = self.hidden_size
        d = self.input_size

        self.params = {
            "W_xh": self._xavier(d, h),
            "W_hh": self._xavier(h, h),
            "b_h": np.zeros((1, h)),
            "W_hy": self._xavier(h, 1),
            "b_y": np.zeros((1, 1)),
        }

    def forward(self, x_batch):
        batch_size, time_steps, _ = x_batch.shape
        h = self.hidden_size

        hs = np.zeros((batch_size, time_steps + 1, h), dtype=np.float64)

        for t in range(time_steps):
            x_t = x_batch[:, t, :]
            hs[:, t + 1, :] = np.tanh(
                x_t @ self.params["W_xh"] +
                hs[:, t, :] @ self.params["W_hh"] +
                self.params["b_h"]
            )

        h_last = hs[:, -1, :]
        y_pred = h_last @ self.params["W_hy"] + self.params["b_y"]

        cache = {
            "x": x_batch,
            "hs": hs,
            "y_pred": y_pred
        }
        return y_pred, cache

    def backward(self, cache, y_true):
        x = cache["x"]
        hs = cache["hs"]
        y_pred = cache["y_pred"]

        batch_size, time_steps, _ = x.shape

        grads = {k: np.zeros_like(v) for k, v in self.params.items()}

        dy = 2.0 * (y_pred - y_true) / batch_size

        grads["W_hy"] += hs[:, -1, :].T @ dy
        grads["b_y"] += np.sum(dy, axis=0, keepdims=True)

        dh_next = dy @ self.params["W_hy"].T

        for t in reversed(range(time_steps)):
            h_t = hs[:, t + 1, :]
            h_prev = hs[:, t, :]
            x_t = x[:, t, :]

            da = dh_next * (1 - h_t ** 2)

            grads["W_xh"] += x_t.T @ da
            grads["W_hh"] += h_prev.T @ da
            grads["b_h"] += np.sum(da, axis=0, keepdims=True)

            dh_next = da @ self.params["W_hh"].T

        return grads

class GRURegressor(BaseSequenceRegressor):
    def _init_params(self):
        h = self.hidden_size
        d = self.input_size

        self.params = {
            "W_xz": self._xavier(d, h),
            "W_hz": self._xavier(h, h),
            "b_z": np.zeros((1, h)),

            "W_xr": self._xavier(d, h),
            "W_hr": self._xavier(h, h),
            "b_r": np.zeros((1, h)),

            "W_xh": self._xavier(d, h),
            "W_hh": self._xavier(h, h),
            "b_h": np.zeros((1, h)),

            "W_hy": self._xavier(h, 1),
            "b_y": np.zeros((1, 1)),
        }

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def forward(self, x_batch):
        batch_size, time_steps, _ = x_batch.shape
        h = self.hidden_size

        hs = np.zeros((batch_size, time_steps + 1, h), dtype=np.float64)
        zs = np.zeros((batch_size, time_steps, h), dtype=np.float64)
        rs = np.zeros((batch_size, time_steps, h), dtype=np.float64)
        h_tildes = np.zeros((batch_size, time_steps, h), dtype=np.float64)

        for t in range(time_steps):
            x_t = x_batch[:, t, :]
            h_prev = hs[:, t, :]

            z = self.sigmoid(x_t @ self.params["W_xz"] + h_prev @ self.params["W_hz"] + self.params["b_z"])
            r = self.sigmoid(x_t @ self.params["W_xr"] + h_prev @ self.params["W_hr"] + self.params["b_r"])
            h_tilde = np.tanh(x_t @ self.params["W_xh"] + (r * h_prev) @ self.params["W_hh"] + self.params["b_h"])
            h_new = (1 - z) * h_prev + z * h_tilde

            zs[:, t, :] = z
            rs[:, t, :] = r
            h_tildes[:, t, :] = h_tilde
            hs[:, t + 1, :] = h_new

        y_pred = hs[:, -1, :] @ self.params["W_hy"] + self.params["b_y"]

        cache = {
            "x": x_batch,
            "hs": hs,
            "zs": zs,
            "rs": rs,
            "h_tildes": h_tildes,
            "y_pred": y_pred
        }
        return y_pred, cache

    def backward(self, cache, y_true):
        x = cache["x"]
        hs = cache["hs"]
        zs = cache["zs"]
        rs = cache["rs"]
        h_tildes = cache["h_tildes"]
        y_pred = cache["y_pred"]

        batch_size, time_steps, _ = x.shape
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}

        dy = 2.0 * (y_pred - y_true) / batch_size
        grads["W_hy"] += hs[:, -1, :].T @ dy
        grads["b_y"] += np.sum(dy, axis=0, keepdims=True)

        dh_next = dy @ self.params["W_hy"].T

        for t in reversed(range(time_steps)):
            x_t = x[:, t, :]
            h_prev = hs[:, t, :]
            z = zs[:, t, :]
            r = rs[:, t, :]
            h_tilde = h_tildes[:, t, :]

            dh = dh_next

            dz = dh * (h_tilde - h_prev)
            dh_tilde = dh * z
            dh_prev = dh * (1 - z)

            da_h = dh_tilde * (1 - h_tilde ** 2)
            grads["W_xh"] += x_t.T @ da_h
            grads["W_hh"] += (r * h_prev).T @ da_h
            grads["b_h"] += np.sum(da_h, axis=0, keepdims=True)

            d_rhprev = da_h @ self.params["W_hh"].T
            dr = d_rhprev * h_prev
            dh_prev += d_rhprev * r

            da_r = dr * r * (1 - r)
            grads["W_xr"] += x_t.T @ da_r
            grads["W_hr"] += h_prev.T @ da_r
            grads["b_r"] += np.sum(da_r, axis=0, keepdims=True)
            dh_prev += da_r @ self.params["W_hr"].T

            da_z = dz * z * (1 - z)
            grads["W_xz"] += x_t.T @ da_z
            grads["W_hz"] += h_prev.T @ da_z
            grads["b_z"] += np.sum(da_z, axis=0, keepdims=True)
            dh_prev += da_z @ self.params["W_hz"].T

            dh_next = dh_prev

        return grads

class LSTMRegressor(BaseSequenceRegressor):
    def _init_params(self):
        h = self.hidden_size
        d = self.input_size

        self.params = {
            "W_xf": self._xavier(d, h),
            "W_hf": self._xavier(h, h),
            "b_f": np.zeros((1, h)),

            "W_xi": self._xavier(d, h),
            "W_hi": self._xavier(h, h),
            "b_i": np.zeros((1, h)),

            "W_xo": self._xavier(d, h),
            "W_ho": self._xavier(h, h),
            "b_o": np.zeros((1, h)),

            "W_xg": self._xavier(d, h),
            "W_hg": self._xavier(h, h),
            "b_g": np.zeros((1, h)),

            "W_hy": self._xavier(h, 1),
            "b_y": np.zeros((1, 1)),
        }

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def forward(self, x_batch):
        batch_size, time_steps, _ = x_batch.shape
        h = self.hidden_size

        hs = np.zeros((batch_size, time_steps + 1, h), dtype=np.float64)
        cs = np.zeros((batch_size, time_steps + 1, h), dtype=np.float64)

        fs = np.zeros((batch_size, time_steps, h), dtype=np.float64)
        ins = np.zeros((batch_size, time_steps, h), dtype=np.float64)
        os_ = np.zeros((batch_size, time_steps, h), dtype=np.float64)
        gs = np.zeros((batch_size, time_steps, h), dtype=np.float64)

        for t in range(time_steps):
            x_t = x_batch[:, t, :]
            h_prev = hs[:, t, :]
            c_prev = cs[:, t, :]

            f = self.sigmoid(x_t @ self.params["W_xf"] + h_prev @ self.params["W_hf"] + self.params["b_f"])
            i = self.sigmoid(x_t @ self.params["W_xi"] + h_prev @ self.params["W_hi"] + self.params["b_i"])
            o = self.sigmoid(x_t @ self.params["W_xo"] + h_prev @ self.params["W_ho"] + self.params["b_o"])
            g = np.tanh(x_t @ self.params["W_xg"] + h_prev @ self.params["W_hg"] + self.params["b_g"])

            c = f * c_prev + i * g
            h_new = o * np.tanh(c)

            fs[:, t, :] = f
            ins[:, t, :] = i
            os_[:, t, :] = o
            gs[:, t, :] = g
            cs[:, t + 1, :] = c
            hs[:, t + 1, :] = h_new

        y_pred = hs[:, -1, :] @ self.params["W_hy"] + self.params["b_y"]

        cache = {
            "x": x_batch,
            "hs": hs,
            "cs": cs,
            "fs": fs,
            "ins": ins,
            "os": os_,
            "gs": gs,
            "y_pred": y_pred
        }
        return y_pred, cache

    def backward(self, cache, y_true):
        x = cache["x"]
        hs = cache["hs"]
        cs = cache["cs"]
        fs = cache["fs"]
        ins = cache["ins"]
        os_ = cache["os"]
        gs = cache["gs"]
        y_pred = cache["y_pred"]

        batch_size, time_steps, _ = x.shape
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}

        dy = 2.0 * (y_pred - y_true) / batch_size
        grads["W_hy"] += hs[:, -1, :].T @ dy
        grads["b_y"] += np.sum(dy, axis=0, keepdims=True)

        dh_next = dy @ self.params["W_hy"].T
        dc_next = np.zeros_like(hs[:, 0, :])

        for t in reversed(range(time_steps)):
            x_t = x[:, t, :]
            h_prev = hs[:, t, :]
            c_prev = cs[:, t, :]
            c_t = cs[:, t + 1, :]

            f = fs[:, t, :]
            i = ins[:, t, :]
            o = os_[:, t, :]
            g = gs[:, t, :]

            tanh_c = np.tanh(c_t)

            dh = dh_next
            do = dh * tanh_c
            dc = dh * o * (1 - tanh_c ** 2) + dc_next

            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc_prev = dc * f

            da_f = df * f * (1 - f)
            da_i = di * i * (1 - i)
            da_o = do * o * (1 - o)
            da_g = dg * (1 - g ** 2)

            grads["W_xf"] += x_t.T @ da_f
            grads["W_hf"] += h_prev.T @ da_f
            grads["b_f"] += np.sum(da_f, axis=0, keepdims=True)

            grads["W_xi"] += x_t.T @ da_i
            grads["W_hi"] += h_prev.T @ da_i
            grads["b_i"] += np.sum(da_i, axis=0, keepdims=True)

            grads["W_xo"] += x_t.T @ da_o
            grads["W_ho"] += h_prev.T @ da_o
            grads["b_o"] += np.sum(da_o, axis=0, keepdims=True)

            grads["W_xg"] += x_t.T @ da_g
            grads["W_hg"] += h_prev.T @ da_g
            grads["b_g"] += np.sum(da_g, axis=0, keepdims=True)

            dh_prev = (
                da_f @ self.params["W_hf"].T +
                da_i @ self.params["W_hi"].T +
                da_o @ self.params["W_ho"].T +
                da_g @ self.params["W_hg"].T
            )

            dh_next = dh_prev
            dc_next = dc_prev

        return grads

def train_and_evaluate(model, model_name, x_train, y_train, x_val, y_val, x_test, y_test,
                       scaler, feature_count, target_index):
    print("\n" + "=" * 60)
    print(f"Обучение модели: {model_name}")
    print("=" * 60)

    history = model.fit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        verbose=1
    )

    y_pred_scaled = model.predict_scaled(x_test)
    y_true_scaled = y_test

    y_pred = inverse_transform_target(y_pred_scaled, scaler, feature_count, target_index)
    y_true = inverse_transform_target(y_true_scaled, scaler, feature_count, target_index)

    metrics = calculate_metrics(y_true, y_pred)
    return history, y_true, y_pred, metrics

def plot_loss(histories):
    plt.figure(figsize=(12, 6))
    for model_name, history in histories.items():
        plt.plot(history["loss"], label=f"{model_name} train")
        plt.plot(history["val_loss"], linestyle="--", label=f"{model_name} val")

    plt.title("Сравнение кривых обучения")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_predictions(y_true, predictions_dict, points=300):
    plt.figure(figsize=(14, 7))
    plt.plot(y_true[:points], label="Реальные значения", linewidth=2)

    for model_name, y_pred in predictions_dict.items():
        plt.plot(y_pred[:points], label=model_name)

    plt.title("Сравнение прогнозов моделей")
    plt.xlabel("Наблюдение")
    plt.ylabel("Usage_kWh")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_metrics(metrics_df):
    metrics_to_plot = ["RMSE", "MAE", "MAPE (%)"]

    for metric in metrics_to_plot:
        plt.figure(figsize=(8, 5))
        plt.bar(metrics_df["Model"], metrics_df[metric])
        plt.title(f"Сравнение моделей по метрике {metric}")
        plt.ylabel(metric)
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()

def main():
    df = load_and_prepare_data(FILE_PATH)

    print("\nРазмер датасета:", df.shape)

    train_df, test_df = split_train_test_by_time(df, TRAIN_SIZE)

    print("\nРазмер train:", train_df.shape)
    print("Размер test :", test_df.shape)

    train_scaled, test_scaled, scaler, feature_columns, target_index = fit_transform_train_test(
        train_df, test_df, TARGET_COLUMN
    )

    x_train_full, y_train_full = create_train_sequences(train_scaled, target_index, TIME_STEPS)
    x_test, y_test = create_test_sequences(train_scaled, test_scaled, target_index, TIME_STEPS)

    x_train, y_train, x_val, y_val = train_val_split_sequences(x_train_full, y_train_full, VAL_SIZE)

    print("\nФормы массивов:")
    print("X_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("X_val  :", x_val.shape)
    print("y_val  :", y_val.shape)
    print("X_test :", x_test.shape)
    print("y_test :", y_test.shape)

    input_size = x_train.shape[2]
    feature_count = len(feature_columns)

    models = {
        "RNN": SimpleRNNRegressor(input_size=input_size, hidden_size=HIDDEN_SIZE, learning_rate=LEARNING_RATE, seed=RANDOM_SEED),
        "GRU": GRURegressor(input_size=input_size, hidden_size=HIDDEN_SIZE, learning_rate=LEARNING_RATE, seed=RANDOM_SEED),
        "LSTM": LSTMRegressor(input_size=input_size, hidden_size=HIDDEN_SIZE, learning_rate=LEARNING_RATE, seed=RANDOM_SEED),
    }

    histories = {}
    predictions = {}
    metrics_rows = []
    true_values_reference = None

    for model_name, model in models.items():
        history, y_true, y_pred, metrics = train_and_evaluate(
            model=model,
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            scaler=scaler,
            feature_count=feature_count,
            target_index=target_index
        )

        histories[model_name] = history
        predictions[model_name] = y_pred.reshape(-1)

        if true_values_reference is None:
            true_values_reference = y_true.reshape(-1)

        row = {"Model": model_name}
        row.update(metrics)
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("RMSE").reset_index(drop=True)

    print("\nИтоговые метрики:")
    print(metrics_df)

    plot_loss(histories)
    plot_predictions(true_values_reference, predictions, points=300)
    plot_metrics(metrics_df)


if __name__ == "__main__":
    main()