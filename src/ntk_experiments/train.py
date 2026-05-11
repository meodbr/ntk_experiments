import torch
import torch.nn as nn

from ntk_experiments.dataset import get_dataset
from ntk_experiments.ntkmlp_model import NTKMLP

# Training function
def train_model(model, dataset='synthetic', epochs=50, lr=1e-3):
    if isinstance(dataset, str):
        dataset = get_dataset(dataset)

    X_train, X_test, y_train, y_test = dataset
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
