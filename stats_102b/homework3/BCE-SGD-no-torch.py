import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


seed_parameter = 123
n = 10_000
p = 100
corr_factor = 0.5
flip_factor = 0.15

train_idx = slice(0, int(0.6 * n))
val_idx = slice(int(0.6 * n), int(0.8 * n))
test_idx = slice(int(0.8 * n), n)

epochs = 50
batch_sizes = [4]
hidden_sizes = [16, 64, 256]
base_step_size = 0.1
decay_rate = 0.7

rng = np.random.default_rng(seed_parameter)


def generate_correlated_data(n_samples=n, n_features=p, rho=corr_factor, flip_f=flip_factor):
    sigma = np.full((n_features, n_features), rho, dtype=float)
    np.fill_diagonal(sigma, 1.0)

    x = rng.multivariate_normal(mean=np.zeros(n_features), cov=sigma, size=n_samples)
    beta = rng.uniform(-1, 1, size=n_features)
    logits = x @ beta
    probs = 1 / (1 + np.exp(-logits))

    y = rng.binomial(1, probs)
    flip_count = int(flip_f * n_samples)
    flip_idx = rng.choice(n_samples, size=flip_count, replace=False)
    y[flip_idx] = 1 - y[flip_idx]

    return x, y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def bce_loss(y_true, y_pred, eps=1e-8):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def xavier_init(in_dim, out_dim):
    limit = math.sqrt(6 / (in_dim + out_dim))
    return rng.uniform(-limit, limit, size=(in_dim, out_dim))


def he_init(in_dim, out_dim):
    stddev = math.sqrt(2 / in_dim)
    return rng.normal(0, stddev, size=(in_dim, out_dim))


X, y = generate_correlated_data()
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

results = {}
best_model = None
best_val_loss = float("inf")

for batch_size in batch_sizes:
    for hidden_size in hidden_sizes:
        W1 = he_init(p, hidden_size)
        b1 = np.zeros(hidden_size)
        W2 = xavier_init(hidden_size, 1).reshape(hidden_size, 1)
        b2 = np.zeros(1)

        train_loss_history = []
        val_loss_history = []

        for epoch in range(epochs):
            lr = base_step_size * (decay_rate ** epoch)
            perm = rng.permutation(len(X_train))

            for start in range(0, len(X_train), batch_size):
                end = start + batch_size
                batch_idx = perm[start:end]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx][:, None]

                A = X_batch @ W1 + b1
                H = relu(A)
                Z = H @ W2 + b2
                y_hat = sigmoid(Z)

                dZ = y_hat - y_batch
                batch_size_eff = len(X_batch)
                dW2 = H.T @ dZ / batch_size_eff
                db2 = dZ.mean(axis=0)
                dH = dZ @ W2.T
                dA = dH * (A > 0)
                dW1 = X_batch.T @ dA / batch_size_eff
                db1 = dA.mean(axis=0)

                W1 -= lr * dW1
                b1 -= lr * db1
                W2 -= lr * dW2
                b2 -= lr * db2

            H_train = relu(X_train @ W1 + b1)
            y_train_pred = sigmoid(H_train @ W2 + b2)
            train_loss = bce_loss(y_train, y_train_pred)

            H_val = relu(X_val @ W1 + b1)
            y_val_pred = sigmoid(H_val @ W2 + b2)
            val_loss = bce_loss(y_val, y_val_pred)

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)

        key = f"Batch_{batch_size}_Hidden_{hidden_size}"
        results[key] = {"train": train_loss_history, "val": val_loss_history}

        if min(val_loss_history) < best_val_loss:
            best_val_loss = min(val_loss_history)
            best_model = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

print(f"Best validation loss: {best_val_loss:.4f}")

train_fig = Path(__file__).with_name("BCE-SGD-no-torch_training.png")
val_fig = Path(__file__).with_name("BCE-SGD-no-torch_validation.png")

plt.figure(figsize=(10, 6))
for key, history in results.items():
    plt.plot(range(1, epochs + 1), history["train"], label=key)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.tight_layout()
plt.savefig(train_fig)

plt.figure(figsize=(10, 6))
for key, history in results.items():
    plt.plot(range(1, epochs + 1), history["val"], label=key, linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss over Epochs")
plt.legend()
plt.tight_layout()
plt.savefig(val_fig)

print(f"Saved training loss plot to {train_fig}")
print(f"Saved validation loss plot to {val_fig}")
