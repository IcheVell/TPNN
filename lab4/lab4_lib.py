import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 20
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_and_prepare_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("Исходные формы:")
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_test :", x_test.shape)
    print("y_test :", y_test.shape)

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), mode="constant")
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), mode="constant")

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)

    print("\nПосле подготовки:")
    print("x_train:", x_train.shape)
    print("y_train_cat:", y_train_cat.shape)
    print("x_test :", x_test.shape)
    print("y_test_cat :", y_test_cat.shape)

    return x_train, y_train, y_train_cat, x_test, y_test, y_test_cat


def build_lenet5():
    model = models.Sequential(
        [
            layers.Input(shape=(32, 32, 1)),

            layers.Conv2D(
                filters=6,
                kernel_size=(5, 5),
                activation="tanh",
                padding="valid"
            ),

            layers.AveragePooling2D(pool_size=(2, 2), strides=2),

            layers.Conv2D(
                filters=16,
                kernel_size=(5, 5),
                activation="tanh",
                padding="valid"
            ),

            layers.AveragePooling2D(pool_size=(2, 2), strides=2),

            layers.Flatten(),
            layers.Dense(120, activation="tanh"),
            layers.Dense(84, activation="tanh"),
            layers.Dense(NUM_CLASSES, activation="softmax")
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss", linestyle="--")
    plt.title("Кривые обучения: Loss")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history["accuracy"], label="train accuracy")
    plt.plot(history.history["val_accuracy"], label="val accuracy", linestyle="--")
    plt.title("Кривые обучения: Accuracy")
    plt.xlabel("Эпоха")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_predictions(x_test, y_test, y_pred, count=15):
    plt.figure(figsize=(15, 5))
    for i in range(count):
        plt.subplot(3, 5, i + 1)
        plt.imshow(x_test[i].squeeze(), cmap="gray")
        plt.title(f"True: {y_test[i]}\nPred: {y_pred[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    x_train, y_train, y_train_cat, x_test, y_test, y_test_cat = load_and_prepare_mnist()

    model = build_lenet5()

    print("\nСводка модели:")
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    print("\nНачало обучения...")
    history = model.fit(
        x_train,
        y_train_cat,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stop],
        verbose=1
    )

    print("\nОценка на тестовой выборке:")
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"Test loss     : {test_loss:.4f}")
    print(f"Test accuracy : {test_acc:.4f}")

    y_pred_proba = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(NUM_CLASSES))
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Матрица ошибок")
    plt.show()

    plot_training_history(history)

    show_predictions(x_test, y_test, y_pred, count=15)


if __name__ == "__main__":
    main()