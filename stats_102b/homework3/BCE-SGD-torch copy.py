from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


torch.manual_seed(123)
rng = np.random.default_rng(123)

n = 10_000
p = 100
corr_factor = 0.5
flip_factor = 0.15

train_idx = slice(0, int(0.6 * n))
val_idx = slice(int(0.6 * n), int(0.8 * n))
test_idx = slice(int(0.8 * n), n)

epochs = 30
batch_sizes = [64, 128, 256]
hidden_sizes = [64, 128, 256]
base_lr = 0.1
decay_rate = 0.7


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

    return x.astype(np.float32), y.astype(np.float32)


X, y = generate_correlated_data()
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

x_train = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train).unsqueeze(1)
x_val = torch.tensor(X_val)
y_val_tensor = torch.tensor(y_val).unsqueeze(1)
x_test = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test).unsqueeze(1)


class SingleLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()
        self.out_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.out_activation(self.output(x))
        return x


results = {}
best_val_loss = float("inf")
best_state = None
best_config = None

loss_fn = nn.BCELoss()

for batch_size in batch_sizes:
    for hidden_dim in hidden_sizes:
        print(f"Training model with batch size={batch_size}, hidden dim={hidden_dim}")
        model = SingleLayerMLP(p, hidden_dim)
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)

        train_loss_history = []
        val_loss_history = []

        for epoch in range(epochs):
            lr = base_lr * (decay_rate ** epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            perm = torch.randperm(x_train.shape[0])
            for start in range(0, x_train.shape[0], batch_size):
                end = start + batch_size
                idx = perm[start:end]
                x_batch = x_train[idx]
                y_batch = y_train_tensor[idx]

                optimizer.zero_grad()
                preds = model(x_batch)
                loss = loss_fn(preds, y_batch)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                train_pred = model(x_train)
                val_pred = model(x_val)
                train_loss = loss_fn(train_pred, y_train_tensor).item()
                val_loss = loss_fn(val_pred, y_val_tensor).item()

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)

        key = f"Batch_{batch_size}_Hidden_{hidden_dim}"
        results[key] = {"train": train_loss_history, "val": val_loss_history}

        if min(val_loss_history) < best_val_loss:
            best_val_loss = min(val_loss_history)
            best_state = model.state_dict()
            best_config = key

print(f"Best model config: {best_config} with val loss {best_val_loss:.4f}")

train_fig = Path(__file__).with_name("BCE-SGD-torch_copy_training.png")
val_fig = Path(__file__).with_name("BCE-SGD-torch_copy_validation.png")

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
