import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def corr_heatmap(df, title, save_path):
    corr = df.corr(numeric_only=True, method="pearson")
    plt.figure(figsize=(9, 7))
    plt.imshow(corr.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return corr


def split_df(df, test_size=0.2, random_state=0):
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(df))
    rng.shuffle(idx)

    test_n = int(round(len(df) * test_size))
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def standardize_fit(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma


def standardize_transform(X, mu, sigma):
    return (X - mu) / sigma


def load_and_merge_wines(red_path, white_path):
    red = pd.read_csv(red_path, sep=";")
    white = pd.read_csv(white_path, sep=";")

    red.columns = red.columns.str.strip()
    white.columns = white.columns.str.strip()

    red["color"] = 1
    white["color"] = 0

    df = pd.concat([red, white], ignore_index=True)
    df.columns = df.columns.str.strip()
    return df


def fit_fillna_means(df):
    return df.mean(numeric_only=True)


def apply_fillna_means(df, means):
    out = df.copy()
    for c in out.columns:
        if c in means.index:
            out[c] = out[c].fillna(means[c])
    return out


def fit_iqr_bounds(df, cols, k=1.5):
    bounds = {}
    for c in cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        bounds[c] = (lo, hi)
    return bounds


def apply_iqr_clip(df, bounds):
    out = df.copy()
    for c, (lo, hi) in bounds.items():
        out[c] = out[c].clip(lower=lo, upper=hi)
    return out


def drop_highly_correlated_features(df, threshold=0.85):
    corr = df.corr(numeric_only=True).abs()
    cols = list(corr.columns)

    to_drop = set()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr.iloc[i, j] >= threshold:
                to_drop.add(cols[j])

    kept = [c for c in cols if c not in to_drop]
    return df[kept].copy(), sorted(to_drop)


def prepare_wine_regression_data(
    red_path="winequality-red.csv",
    white_path="winequality-white.csv",
    test_size=0.2,
    random_state=0,
    corr_threshold=0.85,
    heatmap_path="corr_mlp_regression.png"
):
    df = load_and_merge_wines(red_path, white_path)
    df = df.drop_duplicates().reset_index(drop=True)

    train_df, test_df = split_df(df, test_size=test_size, random_state=random_state)

    y_train_raw = train_df["quality"].to_numpy(dtype=float).reshape(-1, 1)
    y_test_raw = test_df["quality"].to_numpy(dtype=float).reshape(-1, 1)

    X_train_df = train_df.drop(columns=["quality"]).copy()
    X_test_df = test_df.drop(columns=["quality"]).copy()

    fill_means = fit_fillna_means(X_train_df)
    X_train_df = apply_fillna_means(X_train_df, fill_means)
    X_test_df = apply_fillna_means(X_test_df, fill_means)

    iqr_bounds = fit_iqr_bounds(X_train_df, cols=X_train_df.columns.tolist(), k=1.5)
    X_train_df = apply_iqr_clip(X_train_df, iqr_bounds)
    X_test_df = apply_iqr_clip(X_test_df, iqr_bounds)

    X_train_df, dropped = drop_highly_correlated_features(X_train_df, threshold=corr_threshold)
    X_test_df = X_test_df[X_train_df.columns].copy()

    print(f"Удалены по корреляции (|corr| >= {corr_threshold}): {dropped}")

    corr_heatmap(
        X_train_df,
        "Wine correlation heatmap after preprocessing (train)",
        heatmap_path
    )

    feature_names = X_train_df.columns.tolist()

    X_train = X_train_df.to_numpy(dtype=float)
    X_test = X_test_df.to_numpy(dtype=float)

    x_mu, x_sigma = standardize_fit(X_train)
    X_train_s = standardize_transform(X_train, x_mu, x_sigma)
    X_test_s = standardize_transform(X_test, x_mu, x_sigma)

    y_mu = y_train_raw.mean(axis=0)
    y_sigma = y_train_raw.std(axis=0, ddof=0)
    y_sigma[y_sigma == 0] = 1.0

    y_train_s = (y_train_raw - y_mu) / y_sigma
    y_test_s = (y_test_raw - y_mu) / y_sigma

    return {
        "X_train": X_train_s,
        "X_test": X_test_s,
        "y_train": y_train_s,
        "y_test": y_test_s,
        "y_train_raw": y_train_raw,
        "y_test_raw": y_test_raw,
        "x_mu": x_mu,
        "x_sigma": x_sigma,
        "y_mu": y_mu,
        "y_sigma": y_sigma,
        "feature_names": feature_names,
        "dropped_features": dropped
    }

class MLPRegressorScratch:
    def __init__(
        self,
        input_dim,
        hidden_layers=(32, 16),
        learning_rate=0.005,
        epochs=500,
        batch_size=64,
        random_state=0,
        l2=0.0
    ):
        self.input_dim = input_dim
        self.hidden_layers = tuple(hidden_layers)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.l2 = l2

        self.weights = []
        self.biases = []
        self.loss_history = []

        self._init_params()

    def _init_params(self):
        rng = np.random.default_rng(self.random_state)
        layer_sizes = [self.input_dim] + list(self.hidden_layers) + [1]

        self.weights = []
        self.biases = []

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            W = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
            b = np.zeros((1, fan_out))

            self.weights.append(W)
            self.biases.append(b)

    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def relu_grad(z):
        return (z > 0).astype(float)

    def forward(self, X):
        activations = [X]
        pre_activations = []

        A = X
        last_idx = len(self.weights) - 1

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = A @ W + b
            pre_activations.append(Z)

            if i == last_idx:
                A = Z
            else:
                A = self.relu(Z)

            activations.append(A)

        return activations, pre_activations

    def predict(self, X):
        A = X
        last_idx = len(self.weights) - 1

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = A @ W + b
            if i == last_idx:
                A = Z
            else:
                A = self.relu(Z)

        return A

    def compute_loss(self, y_true, y_pred):
        n = y_true.shape[0]

        data_loss = 0.5 * np.mean((y_pred - y_true) ** 2)

        reg_loss = 0.0
        if self.l2 > 0:
            reg_loss = (self.l2 / (2.0 * n)) * sum(np.sum(W ** 2) for W in self.weights)

        return data_loss + reg_loss

    def backward(self, activations, pre_activations, y_true):
        m = y_true.shape[0]
        dW = [None] * len(self.weights)
        db = [None] * len(self.biases)

        dA = (activations[-1] - y_true) / m

        for layer in reversed(range(len(self.weights))):
            A_prev = activations[layer]

            if layer == len(self.weights) - 1:
                dZ = dA
            else:
                dZ = dA * self.relu_grad(pre_activations[layer])

            dW[layer] = A_prev.T @ dZ
            db[layer] = np.sum(dZ, axis=0, keepdims=True)

            if self.l2 > 0:
                dW[layer] += (self.l2 / m) * self.weights[layer]

            dA = dZ @ self.weights[layer].T

        return dW, db

    def update_params(self, dW, db):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def fit(self, X, y, verbose=True):
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]

        for epoch in range(1, self.epochs + 1):
            idx = np.arange(n)
            rng.shuffle(idx)

            X_sh = X[idx]
            y_sh = y[idx]

            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                xb = X_sh[start:end]
                yb = y_sh[start:end]

                activations, pre_activations = self.forward(xb)
                dW, db = self.backward(activations, pre_activations, yb)
                self.update_params(dW, db)

            train_pred = self.predict(X)
            loss = self.compute_loss(y, train_pred)
            self.loss_history.append(loss)

            if verbose and (epoch == 1 or epoch % 50 == 0 or epoch == self.epochs):
                print(f"Epoch {epoch:4d}/{self.epochs} | loss = {loss:.6f}")

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def rounded_accuracy(y_true, y_pred, low=3, high=9):
    y_hat_round = np.rint(y_pred).clip(low, high)
    return float(np.mean(y_true == y_hat_round))

def run_mlp_wine_regression(
    red_path="winequality-red.csv",
    white_path="winequality-white.csv",
    test_size=0.2,
    random_state=0,
    corr_threshold=0.85,
    hidden_layers=(64, 32),
    learning_rate=0.005,
    epochs=700,
    batch_size=64,
    l2=1e-4
):
    data = prepare_wine_regression_data(
        red_path=red_path,
        white_path=white_path,
        test_size=test_size,
        random_state=random_state,
        corr_threshold=corr_threshold,
        heatmap_path="corr_mlp_regression.png"
    )

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]

    model = MLPRegressorScratch(
        input_dim=X_train.shape[1],
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
        l2=l2
    )

    model.fit(X_train, y_train, verbose=True)

    y_pred_train_s = model.predict(X_train)
    y_pred_test_s = model.predict(X_test)

    y_mu = data["y_mu"]
    y_sigma = data["y_sigma"]

    y_train_true = data["y_train_raw"]
    y_test_true = data["y_test_raw"]

    y_pred_train = y_pred_train_s * y_sigma + y_mu
    y_pred_test = y_pred_test_s * y_sigma + y_mu

    print("\nИспользованные признаки:")
    print(data["feature_names"])

    print("\nМетрики на train:")
    print(f"MAE  = {mae(y_train_true, y_pred_train):.4f}")
    print(f"RMSE = {rmse(y_train_true, y_pred_train):.4f}")
    print(f"R2   = {r2_score_np(y_train_true, y_pred_train):.4f}")
    print(f"Rounded accuracy = {rounded_accuracy(y_train_true, y_pred_train):.4f}")

    print("\nМетрики на test:")
    print(f"MAE  = {mae(y_test_true, y_pred_test):.4f}")
    print(f"RMSE = {rmse(y_test_true, y_pred_test):.4f}")
    print(f"R2   = {r2_score_np(y_test_true, y_pred_test):.4f}")
    print(f"Rounded accuracy = {rounded_accuracy(y_test_true, y_pred_test):.4f}")

    history_df = pd.DataFrame({
        "epoch": np.arange(1, len(model.loss_history) + 1),
        "loss": model.loss_history
    })

    history_df.to_csv("mlp_regression_loss_history.csv", index=False, encoding="utf-8-sig")

    pred_df = pd.DataFrame({
        "y_true": y_test_true.ravel(),
        "y_pred": y_pred_test.ravel(),
        "y_pred_round": np.rint(y_pred_test.ravel()).clip(3, 9)
    })
    pred_df.to_csv("mlp_regression_predictions.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(model.loss_history) + 1), model.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP regression training loss (sklearn-style loss)")
    plt.tight_layout()
    plt.savefig("mlp_regression_loss.png", dpi=200, bbox_inches="tight")
    plt.close()

    return model, data, pred_df


if __name__ == "__main__":
    model, data, pred_df = run_mlp_wine_regression(
        red_path="winequality-red.csv",
        white_path="winequality-white.csv",
        test_size=0.2,
        random_state=0,
        corr_threshold=0.85,
        hidden_layers=(64, 32),
        learning_rate=0.005,
        epochs=700,
        batch_size=64,
        l2=1e-4
    )