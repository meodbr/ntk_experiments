import numpy as np
import torch
import pandas as pd
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

# Data (MNIST-like digits)
digits = load_digits()
X = digits.data
y = digits.target

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Simple MLP
class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, width),
            nn.ReLU(),
            nn.Linear(width, 10)
        )

    def forward(self, x):
        return self.net(x)

# Training function
def train_model(width, epochs=50, lr=1e-3):
    model = MLP(width)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        logits = model(X_train)
        loss = loss_fn(logits, y_train)

        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        acc = (preds == y_test).float().mean().item()

    return acc

def compute_mlp_ntk(x, x_prime, width):
    # For simplicity, we compute the NTK for a single layer MLP with ReLU activation
    # NTK for a single layer MLP with ReLU is given by:
    # K(x, x') = (1/width) * (x @ x'.T) * (ReLU(x) @ ReLU(x').T)

    # Compute activations
    relu_x = torch.relu(x)
    relu_x_prime = torch.relu(x_prime)

    # Compute NTK
    ntk = (1 / width) * (x @ x_prime.T) * (relu_x @ relu_x_prime.T)
    return ntk

# Compute Neural Tangent Kernel (NTK) for a given width
def compute_ntk_gram(width):
    model = MLP(width)
    params = list(model.parameters())
    p = sum(p.numel() for p in params)

    # Shape of ntk gram matrix: (n_train, n_train)
    ntk_gram_matrix = torch.zeros((len(X_train), len(X_train)))




# Experiment over widths
widths = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
runs = 5

results = []

for w in widths:
    accs = [train_model(w) for _ in range(runs)]
    for a in accs:
        results.append((w, a))

# Plot with uncertainty

df = pd.DataFrame(results, columns=["width", "accuracy"])

plt.figure(figsize=(8,5))
sns.lineplot(data=df, x="width", y="accuracy", errorbar="sd", marker="o")

plt.xscale("log")
plt.title("MLP Accuracy vs Width (mean ± std over runs)")
plt.xlabel("Width (log scale)")
plt.ylabel("Test Accuracy")

plt.show()