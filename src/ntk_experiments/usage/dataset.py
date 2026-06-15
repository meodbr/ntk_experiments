import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from ntk_experiments.config import config

def get_dataset(name, seed=config.SEED):
    if name == 'synthetic':
        return get_synthetic_data(input_dim=config.INPUT_DIM, output_dim=config.OUTPUT_DIM, seed=seed)
    elif name == 'mnist':
        return get_raw_mnist_data(seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def get_synthetic_data(num_samples=1000, input_dim=64, output_dim=1, seed=42):
    # Synthetic regression data: y = Xw + noise
    torch.manual_seed(seed)
    X = torch.randn(num_samples, input_dim)
    w = torch.randn(input_dim, output_dim)
    y = X @ w + 0.1 * torch.randn(num_samples, output_dim)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    return X_train, X_test, y_train, y_test


def get_lowres_mnist_data(seed=42):
    digits = load_digits()
    X = digits.data
    y = digits.target

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )

    X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 8, 8)  # Reshape to (N, C, H, W)
    X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, 8, 8)  # Reshape to (N, C, H, W)
    y_train = torch.tensor(y_train, dtype=torch.long).view(-1)  # Ensure y is of shape (N,)
    y_test = torch.tensor(y_test, dtype=torch.long).view(-1)  # Ensure y is of shape (N,)

    y_train = F.one_hot(y_train, num_classes=10).float()  # Convert to one-hot encoding
    y_test = F.one_hot(y_test, num_classes=10).float()  # Convert to one-hot encoding

    return X_train, X_test, y_train, y_test

def get_raw_lowres_mnist_data(seed=42):
    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )

    return X_train, X_test, y_train, y_test

def plot_lowres_mnist_sample(X, y, index=0):
    plt.imshow(X[index].reshape(8, 8), cmap='gray')
    plt.title(f"Label: {y[index].item()}")
    plt.axis('off')
    plt.show()

def get_raw_mnist_data(seed=42, n_train=200, n_test=40):
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    g = torch.Generator().manual_seed(seed)

    train_idx = torch.randperm(len(train_dataset), generator=g)[:n_train]
    test_idx = torch.randperm(len(test_dataset), generator=g)[:n_test]

    X_train = torch.stack([
        train_dataset[i.item()][0]
        for i in train_idx
    ])

    y_train = torch.tensor([
        train_dataset[i.item()][1]
        for i in train_idx
    ], dtype=torch.long)

    X_test = torch.stack([
        test_dataset[i.item()][0]
        for i in test_idx
    ])

    y_test = torch.tensor([
        test_dataset[i.item()][1]
        for i in test_idx
    ], dtype=torch.long)

    return X_train, X_test, y_train, y_test

def plot_mnist_sample(X, y, index=0):
    plt.imshow(X[index].numpy().squeeze(), cmap='gray')
    plt.title(f"Label: {y[index].item()}")
    plt.axis('off')
    plt.show()