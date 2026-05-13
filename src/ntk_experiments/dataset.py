import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ntk_experiments.config import config

def get_dataset(name, seed=config.SEED):
    if name == 'synthetic':
        return get_synthetic_data(input_dim=config.INPUT_DIM, output_dim=config.OUTPUT_DIM, seed=seed)
    elif name == 'mnist':
        return get_mnist_data(seed=seed)
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


def get_mnist_data(seed=42):
    digits = load_digits()
    X = digits.data
    y = digits.target

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test