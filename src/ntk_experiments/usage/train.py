import torch
import torch.nn as nn

from ntk_experiments.usage.dataset import get_dataset
from ntk_experiments.MLP.practical.ntkmlp_model import NTKMLP

# Training function
def basic_train_model(model, dataset='synthetic', epochs=50, lr=1e-3):
    if isinstance(dataset, str):
        dataset = get_dataset(dataset)

    X_train, X_test, y_train, y_test = dataset
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        logits = model(X_train)
        loss = loss_fn(logits, y_train)

        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = model(X_test)
        acc = (preds - y_test).abs().mean().item()  # Simple accuracy for regression

    return acc

def train_model(model, dataset, epochs=50, lr=1e-3):
    if isinstance(dataset, str):
        dataset = get_dataset(dataset)

    X_train, X_test, y_train, y_test = dataset

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # m

    for _ in range(epochs):
        logits = model(X_train)
        loss = loss_fn(logits, y_train)

        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = model(X_test)
        acc = (preds - y_test).abs().mean().item()  # Simple accuracy for regression

    return acc

def train_step(model, X, y, opt, loss_fn=None, lr=1e-3):
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    
    opt.zero_grad()

    logits = model(X)
    loss = loss_fn(logits, y)

    loss.backward()
    opt.step()

    return loss.item()

def batched_training(model, dataset, max_steps, lr=1e-3, batch_size=32):
    if isinstance(dataset, str):
        dataset = get_dataset(dataset)

    X_train, X_test, y_train, y_test = dataset

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    num_samples = X_train.size(0)
    steps = 0
    while steps < max_steps:
        perm = torch.randperm(num_samples)
        for i in range(0, num_samples, batch_size):
            indices = perm[i:i + batch_size]
            X_batch, y_batch = X_train[indices], y_train[indices]

            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        preds = model(X_test)
        acc = (preds - y_test).abs().mean().item()  # Simple accuracy for regression

    return acc