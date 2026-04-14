import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

FILE_PATH = "Steel_industry_data.csv"
TARGET_COLUMN = "Usage_kWh"
TIME_STEPS = 24
TRAIN_SIZE = 0.8
BATCH_SIZE = 32
EPOCHS = 30
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

    return df

def split_train_test_by_time(df: pd.DataFrame, train_size: float = 0.8):
    split_index = int(len(df) * train_size)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df

def fit_transform_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str):
    feature_columns = [col for col in train_df.columns if col != "date"]

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_df[feature_columns])
    test_scaled = scaler.transform(test_df[feature_columns])

    train_scaled_df = pd.DataFrame(train_scaled, columns=feature_columns, index=train_df.index)
    test_scaled_df = pd.DataFrame(test_scaled, columns=feature_columns, index=test_df.index)

    target_index = feature_columns.index(target_column)

    return train_scaled_df, test_scaled_df, scaler, feature_columns, target_index

def create_train_sequences(train_scaled: np.ndarray, target_index: int, time_steps: int):
    x_train, y_train = [], []

    for i in range(time_steps, len(train_scaled)):
        x_train.append(train_scaled[i - time_steps:i, :])
        y_train.append(train_scaled[i, target_index])

    return np.array(x_train), np.array(y_train)


def create_test_sequences(train_scaled: np.ndarray, test_scaled: np.ndarray, target_index: int, time_steps: int):
    combined = np.vstack([train_scaled[-time_steps:], test_scaled])

    x_test, y_test = [], []

    for i in range(time_steps, len(combined)):
        x_test.append(combined[i - time_steps:i, :])
        y_test.append(combined[i, target_index])

    return np.array(x_test), np.array(y_test)

def build_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_gru_model(input_shape):
    model = Sequential([
        GRU(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def inverse_transform_target(y_scaled, scaler: MinMaxScaler, feature_count: int, target_index: int):
    temp = np.zeros((len(y_scaled), feature_count))
    temp[:, target_index] = y_scaled.reshape(-1)
    restored = scaler.inverse_transform(temp)
    return restored[:, target_index]

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "R2": r2
    }

def train_and_evaluate(model, model_name, x_train, y_train, x_test, y_test, scaler, feature_count, target_index):
    print(f"\n{'=' * 60}")
    print(f"Обучение модели: {model_name}")
    print(f"{'=' * 60}")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred_scaled = model.predict(x_test, verbose=0).reshape(-1)
    y_test_scaled = y_test.reshape(-1)

    y_pred = inverse_transform_target(y_pred_scaled, scaler, feature_count, target_index)
    y_true = inverse_transform_target(y_test_scaled, scaler, feature_count, target_index)

    metrics = calculate_metrics(y_true, y_pred)

    return history, y_true, y_pred, metrics


def plot_loss(histories):
    plt.figure(figsize=(12, 6))

    for model_name, history in histories.items():
        plt.plot(history.history["loss"], label=f"{model_name} train")
        plt.plot(history.history["val_loss"], linestyle="--", label=f"{model_name} val")

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

    train_scaled_df, test_scaled_df, scaler, feature_columns, target_index = fit_transform_train_test(
        train_df=train_df,
        test_df=test_df,
        target_column=TARGET_COLUMN
    )

    x_train, y_train = create_train_sequences(
        train_scaled=train_scaled_df.values,
        target_index=target_index,
        time_steps=TIME_STEPS
    )

    x_test, y_test = create_test_sequences(
        train_scaled=train_scaled_df.values,
        test_scaled=test_scaled_df.values,
        target_index=target_index,
        time_steps=TIME_STEPS
    )

    print("\nФормы массивов:")
    print("X_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("X_test :", x_test.shape)
    print("y_test :", y_test.shape)

    input_shape = (x_train.shape[1], x_train.shape[2])
    feature_count = len(feature_columns)

    models = {
        "RNN": build_rnn_model(input_shape),
        "GRU": build_gru_model(input_shape),
        "LSTM": build_lstm_model(input_shape)
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
            x_test=x_test,
            y_test=y_test,
            scaler=scaler,
            feature_count=feature_count,
            target_index=target_index
        )

        histories[model_name] = history
        predictions[model_name] = y_pred

        if true_values_reference is None:
            true_values_reference = y_true

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