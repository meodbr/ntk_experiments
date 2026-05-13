from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

from ntk_experiments.config import config
from ntk_experiments.dataset import get_dataset
from ntk_experiments.theoretical_ntk import infinite_width_ntk, simulate_batched_infinite_width_ntk
from ntk_experiments.inference import predict_infinite
from ntk_experiments.ntkmlp_model import NTKMLP
from ntk_experiments.train import train_model

def plot_infinite_width_predictions():
    """
    plot mean and variance of predictions around some of the points of X_test for the infinite width limit of the network.
    This is done by sampling from the Gaussian process defined by the NTK.
    """

    kernel_func = partial(
        simulate_batched_infinite_width_ntk,
        depth=4, 
        implemented_sigma='relu',
        sigma_w=1.0,
        beta=config.BETA,
        n_gh=40,
    )

    dataset = get_dataset(config.DATASET)
    X_train, X_test, y_train, y_test = dataset
    X_train = X_train[:5]
    y_train = y_train[:5]
    gram_matrix = None

    # Make x a zoomed window around some test points
    x = X_test[:10]  # Shape: (10, 64)
    x_max = x.max().item()
    x_min = x.min().item()
    x_resolution = 100
    x = np.zeros((x_resolution, config.INPUT_DIM))  # Shape: (100, 1)
    x[:, 0] = np.linspace(x_min - 1, x_max + 1, x_resolution)  # Vary only the first feature for visualization

    y_pred, y_var, gram_matrix = predict_infinite(x, gram_matrix, kernel_func, X_train, y_train)
    y_pred = y_pred.squeeze()  # Shape: (100,)
    print("y_pred shape:", y_pred.shape)
    print("y_var shape:", y_var.shape)
    print("x shape:", x.shape)
    print("(y_pred - 2 * torch.sqrt(y_var)).shape:", (y_pred - 2 * torch.sqrt(y_var)).shape)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_pred, label='Mean Prediction', color='blue')
    plt.fill_between(x[:, 0].flatten(), (y_pred - 2 * torch.sqrt(y_var)).flatten(), (y_pred + 2 * torch.sqrt(y_var)).flatten(), color='blue', alpha=0.2, label='Confidence Interval (±2 std)')
    plt.scatter(X_train[:, 0].numpy(), y_train.numpy(), color='red', label='Training Data', alpha=0.5)
    plt.title('Infinite-width NTK Predictions with Confidence Intervals')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    plt.show()

def compare_finite_infinite_predictions():
    """
    Compare predictions of a finite width network with the infinite width limit.
    """
    kernel_func = partial(
        simulate_batched_infinite_width_ntk,
        depth=4, 
        implemented_sigma='relu',
        sigma_w=1.0,
        beta=config.BETA,
        n_gh=40,
    )

    dataset = get_dataset(config.DATASET)
    X_train, X_test, y_train, y_test = dataset
    X_train = X_train[:5]
    y_train = y_train[:5]
    gram_matrix = None

    # Make x a zoomed window around some test points
    x = X_test[:10]  # Shape: (10, 64)
    x_max = x.max().item()
    x_min = x.min().item()
    x_resolution = 100
    x = np.zeros((x_resolution, config.INPUT_DIM))  # Shape: (100, 1)
    x[:, 0] = np.linspace(x_min - 1, x_max + 1, x_resolution)  # Vary only the first feature for visualization

    y_pred, y_var, gram_matrix = predict_infinite(x, gram_matrix, kernel_func, X_train, y_train)
    y_pred = y_pred.squeeze()  # Shape: (100,)
    print("y_pred shape:", y_pred.shape)
    print("y_var shape:", y_var.shape)
    print("x shape:", x.shape)
    print("(y_pred - 2 * torch.sqrt(y_var)).shape:", (y_pred - 2 * torch.sqrt(y_var)).shape)

    # TRAIN FINITE WIDTH NETWORKs AND GET ITS PREDICTIONS
    n_models = 5
    models = [NTKMLP(input_dim=config.INPUT_DIM, width=1000, depth=4, beta=config.BETA) for _ in range(n_models)]
    ds_train = (X_train, X_test, y_train.squeeze(), y_test.squeeze())
    finite_preds_list = []
    for model in models:
        acc = train_model(model, dataset=ds_train, epochs=100, lr=.1)
        print(f"Finite width model accuracy: {acc}")
        with torch.no_grad():
            print("x shape, dtype:", x.shape, "dtype:", x.dtype)
            x = torch.tensor(x, dtype=X_train.dtype)  # Ensure x is a torch tensor with the same dtype as X_train
            finite_preds = model(x).squeeze()  # Shape: (100,)
            finite_preds_list.append(finite_preds)

    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_pred, label='Infinite-width Mean Prediction', color='blue')
    plt.fill_between(x[:, 0].flatten(), (y_pred - 2 * torch.sqrt(y_var)).flatten(), (y_pred + 2 * torch.sqrt(y_var)).flatten(), color='blue', alpha=0.2, label='Infinite-width Confidence Interval (±2 std)')
    for i, finite_preds in enumerate(finite_preds_list):
        plt.plot(x, finite_preds, label=f'Finite-width Prediction {i+1}')
    plt.scatter(X_train[:, 0].numpy(), y_train.numpy(), color='red', label='Training Data', alpha=0.5)
    plt.title('Finite vs Infinite-width NTK Predictions')
    plt.xlabel('Input')
    y_min = y_pred.min().item()
    y_max = y_pred.max().item()
    y_range = y_max - y_min
    plt.ylim((y_min - y_range * 0.4, y_max + y_range * 0.4))  # Set y-limits to better visualize the predictions
    plt.ylabel('Output')
    plt.legend()
    plt.show()



def plot_gram_matrix():
    kernel_func = partial(
        simulate_batched_infinite_width_ntk,
        depth=4, 
        implemented_sigma='relu',
        sigma_w=1.0,
        beta=config.BETA,
        n_gh=40,
    )

    dataset = get_dataset(config.DATASET)
    X_train, X_test, y_train, y_test = dataset
    X_train = X_train[:5]
    y_train = y_train[:5]
    gram_matrix = None

    # Make x a zoomed window around some test points
    x = X_test[:10]  # Shape: (10, 64)
    x_max = x.max().item()
    x_min = x.min().item()
    x_resolution = 100
    x = np.zeros((x_resolution, config.INPUT_DIM))  # Shape: (100, 1)
    x[:, 0] = np.linspace(x_min - 1, x_max + 1, x_resolution)  # Vary only the first feature for visualization

    _, _, gram_matrix = predict_infinite(x, gram_matrix, kernel_func, X_train, y_train)

    plt.figure(figsize=(8, 6))
    sns.heatmap(gram_matrix.numpy(), cmap='viridis')
    plt.title('Gram Matrix of the NTK')
    plt.xlabel('Train Points')
    plt.ylabel('Train Points')
    plt.show()

if __name__ == "__main__":
    sns.set_theme()
    # plot_infinite_width_predictions()
    # plot_gram_matrix()
    compare_finite_infinite_predictions()
