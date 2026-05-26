import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.datasets import mnist

NUM_CLASSES = 10
EPOCHS = 8
BATCH_SIZE = 32
LEARNING_RATE = 0.01
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42

TRAIN_LIMIT = 500
TEST_LIMIT = 200

np.random.seed(RANDOM_SEED)

def one_hot_encode(y, num_classes):
    result = np.zeros((len(y), num_classes), dtype=np.float32)
    result[np.arange(len(y)), y] = 1.0
    return result


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def compute_accuracy(y_true, y_pred_probs):
    y_pred = np.argmax(y_pred_probs, axis=1)
    return np.mean(y_true == y_pred)


def tanh(x):
    return np.tanh(x)


def tanh_backward(grad_output, activated):
    return grad_output * (1.0 - activated ** 2)

def load_and_prepare_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("Исходные формы:")
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_test :", x_test.shape)
    print("y_test :", y_test.shape)

    if TRAIN_LIMIT is not None:
        x_train = x_train[:TRAIN_LIMIT]
        y_train = y_train[:TRAIN_LIMIT]

    if TEST_LIMIT is not None:
        x_test = x_test[:TEST_LIMIT]
        y_test = y_test[:TEST_LIMIT]

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), mode="constant")
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), mode="constant")

    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)

    y_train_oh = one_hot_encode(y_train, NUM_CLASSES)
    y_test_oh = one_hot_encode(y_test, NUM_CLASSES)

    split_index = int(len(x_train) * (1 - VALIDATION_SPLIT))

    x_tr = x_train[:split_index]
    y_tr = y_train[:split_index]
    y_tr_oh = y_train_oh[:split_index]

    x_val = x_train[split_index:]
    y_val = y_train[split_index:]
    y_val_oh = y_train_oh[split_index:]

    print("\nПосле подготовки:")
    print("X_train:", x_tr.shape)
    print("y_train:", y_tr.shape)
    print("X_val  :", x_val.shape)
    print("y_val  :", y_val.shape)
    print("X_test :", x_test.shape)
    print("y_test :", y_test.shape)

    return x_tr, y_tr, y_tr_oh, x_val, y_val, y_val_oh, x_test, y_test, y_test_oh

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        limit = np.sqrt(6.0 / (in_channels * kernel_size * kernel_size + out_channels))
        self.W = np.random.uniform(
            -limit, limit,
            size=(out_channels, in_channels, kernel_size, kernel_size)
        ).astype(np.float32)
        self.b = np.zeros(out_channels, dtype=np.float32)

        self.kernel_size = kernel_size
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        F, _, K, _ = self.W.shape

        H_out = H - K + 1
        W_out = W - K + 1

        out = np.zeros((N, F, H_out, W_out), dtype=np.float32)

        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        region = x[n, :, i:i + K, j:j + K]
                        out[n, f, i, j] = np.sum(region * self.W[f]) + self.b[f]

        return out

    def backward(self, grad_output):
        x = self.x
        N, C, H, W = x.shape
        F, _, K, _ = self.W.shape
        _, _, H_out, W_out = grad_output.shape

        grad_input = np.zeros_like(x, dtype=np.float32)
        self.dW = np.zeros_like(self.W, dtype=np.float32)
        self.db = np.zeros_like(self.b, dtype=np.float32)

        for n in range(N):
            for f in range(F):
                self.db[f] += np.sum(grad_output[n, f])
                for i in range(H_out):
                    for j in range(W_out):
                        region = x[n, :, i:i + K, j:j + K]
                        self.dW[f] += grad_output[n, f, i, j] * region
                        grad_input[n, :, i:i + K, j:j + K] += grad_output[n, f, i, j] * self.W[f]

        return grad_input

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class AvgPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.x = None

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        P = self.pool_size
        S = self.stride

        H_out = (H - P) // S + 1
        W_out = (W - P) // S + 1

        out = np.zeros((N, C, H_out, W_out), dtype=np.float32)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h0 = i * S
                        w0 = j * S
                        region = x[n, c, h0:h0 + P, w0:w0 + P]
                        out[n, c, i, j] = np.mean(region)

        return out

    def backward(self, grad_output):
        x = self.x
        N, C, H, W = x.shape
        P = self.pool_size
        S = self.stride
        _, _, H_out, W_out = grad_output.shape

        grad_input = np.zeros_like(x, dtype=np.float32)
        coeff = 1.0 / (P * P)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h0 = i * S
                        w0 = j * S
                        grad_input[n, c, h0:h0 + P, w0:w0 + P] += grad_output[n, c, i, j] * coeff

        return grad_input

    def step(self, lr):
        pass


class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)

    def step(self, lr):
        pass


class Dense:
    def __init__(self, in_features, out_features):
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, size=(in_features, out_features)).astype(np.float32)
        self.b = np.zeros((1, out_features), dtype=np.float32)

        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        self.dW = self.x.T @ grad_output
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.W.T

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

class SimpleLeNetLike:
    def __init__(self):
        self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5)
        self.pool1 = AvgPool2D(pool_size=2, stride=2)  # 28 -> 14

        self.flatten = Flatten()                       # 6*14*14 = 1176
        self.fc1 = Dense(6 * 14 * 14, 64)
        self.fc2 = Dense(64, NUM_CLASSES)

        self.cache = {}

    def forward(self, x):
        z1 = self.conv1.forward(x)
        a1 = tanh(z1)
        p1 = self.pool1.forward(a1)

        f = self.flatten.forward(p1)
        z2 = self.fc1.forward(f)
        a2 = tanh(z2)

        z3 = self.fc2.forward(a2)
        y_pred = softmax(z3)

        self.cache = {
            "a1": a1,
            "a2": a2,
            "y_pred": y_pred
        }

        return y_pred

    def backward(self, y_true):
        y_pred = self.cache["y_pred"]
        a1 = self.cache["a1"]
        a2 = self.cache["a2"]

        batch_size = y_true.shape[0]
        grad = (y_pred - y_true) / batch_size

        grad = self.fc2.backward(grad)
        grad = tanh_backward(grad, a2)

        grad = self.fc1.backward(grad)
        grad = self.flatten.backward(grad)

        grad = self.pool1.backward(grad)
        grad = tanh_backward(grad, a1)

        _ = self.conv1.backward(grad)

    def step(self, lr):
        self.conv1.step(lr)
        self.fc1.step(lr)
        self.fc2.step(lr)

    def predict_probs(self, x):
        return self.forward(x)

def iterate_minibatches(x, y, y_oh, batch_size, shuffle=True):
    indices = np.arange(len(x))
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(x), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield x[batch_idx], y[batch_idx], y_oh[batch_idx]


def evaluate_model(model, x, y, y_oh):
    probs = model.predict_probs(x)
    loss = cross_entropy_loss(y_oh, probs)
    acc = compute_accuracy(y, probs)
    return loss, acc, probs


def train_model(model, x_train, y_train, y_train_oh, x_val, y_val, y_val_oh, epochs, batch_size, lr):
    history = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_accuracy": []
    }

    for epoch in range(1, epochs + 1):
        for x_batch, _, y_batch_oh in iterate_minibatches(x_train, y_train, y_train_oh, batch_size, shuffle=True):
            _ = model.forward(x_batch)
            model.backward(y_batch_oh)
            model.step(lr)

        train_loss, train_acc, _ = evaluate_model(model, x_train, y_train, y_train_oh)
        val_loss, val_acc, _ = evaluate_model(model, x_val, y_val, y_val_oh)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch}/{epochs} - "
            f"loss: {train_loss:.4f} - "
            f"val_loss: {val_loss:.4f} - "
            f"accuracy: {train_acc:.4f} - "
            f"val_accuracy: {val_acc:.4f}"
        )

    return history

def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history["loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss", linestyle="--")
    plt.title("Кривые обучения: Loss")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history["accuracy"], label="train accuracy")
    plt.plot(history["val_accuracy"], label="val accuracy", linestyle="--")
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
        plt.imshow(x_test[i, 0], cmap="gray")
        plt.title(f"True: {y_test[i]}\nPred: {y_pred[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    x_train, y_train, y_train_oh, x_val, y_val, y_val_oh, x_test, y_test, y_test_oh = load_and_prepare_mnist()

    model = SimpleLeNetLike()

    print("\nНачало обучения...")
    history = train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        y_train_oh=y_train_oh,
        x_val=x_val,
        y_val=y_val,
        y_val_oh=y_val_oh,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE
    )

    print("\nОценка на тестовой выборке:")
    test_loss, test_acc, test_probs = evaluate_model(model, x_test, y_test, y_test_oh)
    print(f"Test loss     : {test_loss:.4f}")
    print(f"Test accuracy : {test_acc:.4f}")

    y_pred = np.argmax(test_probs, axis=1)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(NUM_CLASSES))
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title("Матрица ошибок")
    plt.show()

    plot_training_history(history)
    show_predictions(x_test, y_test, y_pred, count=15)


if __name__ == "__main__":
    main()