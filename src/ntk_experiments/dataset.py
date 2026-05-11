import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# old implementation
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


def get_dataset(name):
    if name == 'synthetic':
        return get_synthetic_data()
    elif name == 'mnist':
        return get_mnist_data()
    else:
        raise ValueError(f"Unknown dataset: {name}")

def get_synthetic_data(num_samples=1000, input_dim=64, num_classes=10, seed=42):
    digits = load_digits(seed=seed)
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

def get_mnist_data():
    # Placeholder for MNIST loading logic
    # You can use torchvision.datasets.MNIST to load the dataset
    # and apply necessary transformations (e.g., flattening, normalization)
    pass