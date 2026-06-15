"""
Goal: To investigate the spectral properties of the NTK for different widths of a CNN model.
Compute Eigenvalues of the NTK for different widths and visualize the distribution of eigenvalues.
Note: eigenvalues are also singular values of the NTK matrix, since the NTK is symmetric and positive semi-definite.
"""

from matplotlib.pylab import eigvals
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import seaborn as sns
from tqdm import tqdm

from ntk_experiments.config import config
from ntk_experiments.usage.common_ntk import (
    get_mlp_theoretical_ntk_function,
    get_mlp_practical_ntk_function,
    get_cnn_theoretical_ntk_function,
    get_cnn_practical_ntk_function,
    get_practical_model_ntk_function,
)

from ntk_experiments.CNN.practical.cnn_model import NTKCNN
from ntk_experiments.MLP.practical.ntkmlp_model import NTKMLP

from ntk_experiments.usage.train import train_step
from ntk_experiments.usage.dataset import get_raw_mnist_data, plot_mnist_sample, get_lowres_mnist_data, plot_lowres_mnist_sample
from ntk_experiments.utils.norms import frobenius_relative_change
from ntk_experiments.utils.gram import compute_gram_matrix

sns.set_theme()

def center_kernel(K):
    n = K.shape[0]
    H = torch.eye(n, device=K.device, dtype=K.dtype) - torch.ones(n, n, device=K.device, dtype=K.dtype) / n
    return H @ K @ H

def within_between_kernel_stats(K, y, normalize=False, center=False):
    K = 0.5 * (K + K.T)

    if center:
        n = K.shape[0]
        H = torch.eye(n, device=K.device, dtype=K.dtype) - torch.ones(n, n, device=K.device, dtype=K.dtype) / n
        K = H @ K @ H

    if normalize:
        diag = K.diag().clamp_min(1e-12)
        K = K / torch.sqrt(diag[:, None] * diag[None, :])

    n = K.shape[0]
    same = y[:, None] == y[None, :]
    diag_mask = torch.eye(n, device=K.device, dtype=torch.bool)

    within = K[same & ~diag_mask]
    between = K[~same]

    return {
        "within_mean": within.mean().item(),
        "between_mean": between.mean().item(),
        "gap": (within.mean() - between.mean()).item(),
        "within_std": within.std().item(),
        "between_std": between.std().item(),
    }

def plot_sorted_kernel(K, y, title="Kernel sorted by class", center=True, normalize=True):
    K = 0.5 * (K + K.T)

    if center:
        n = K.shape[0]
        H = torch.eye(n, device=K.device, dtype=K.dtype) - torch.ones(n, n, device=K.device, dtype=K.dtype) / n
        K = H @ K @ H

    if normalize:
        diag = K.diag().clamp_min(1e-12)
        K = K / torch.sqrt(diag[:, None] * diag[None, :])

    idx = torch.argsort(y)
    K_sorted = K[idx][:, idx].detach().cpu()
    y_sorted = y[idx].detach().cpu()

    plt.figure(figsize=(6, 5))
    plt.imshow(K_sorted, cmap="viridis")
    plt.colorbar(label="kernel similarity")
    plt.title(title)
    plt.xlabel("samples sorted by class")
    plt.ylabel("samples sorted by class")

    plt.tight_layout()
    plt.show()

def plot_sorted_kernel_by_class(
    K,
    y,
    title="Kernel sorted by class",
    center=True,
    normalize=True,
    cmap="viridis",
):
    """
    K: tensor of shape (n, n)
    y: tensor of shape (n,), integer class labels
    """

    K = 0.5 * (K + K.T)
    y = y.reshape(-1)

    n = K.shape[0]
    assert K.shape == (n, n)
    assert y.shape == (n,)

    if center:
        H = (
            torch.eye(n, device=K.device, dtype=K.dtype)
            - torch.ones(n, n, device=K.device, dtype=K.dtype) / n
        )
        K = H @ K @ H

    if normalize:
        diag = K.diag().clamp_min(1e-12)
        K = K / torch.sqrt(diag[:, None] * diag[None, :])

    # Sort samples by class
    idx = torch.argsort(y)
    K_sorted = K[idx][:, idx].detach().cpu()
    y_sorted = y[idx].detach().cpu()

    # Find class labels, counts, boundaries, and tick positions
    classes, counts = torch.unique_consecutive(y_sorted, return_counts=True)

    boundaries = torch.cumsum(counts, dim=0)
    starts = torch.cat([torch.tensor([0]), boundaries[:-1]])
    centers = (starts + boundaries - 1) / 2

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(K_sorted, cmap=cmap, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Kernel similarity")

    # Ticks at the center of each class block
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels([str(c.item()) for c in classes])
    ax.set_yticklabels([str(c.item()) for c in classes])

    ax.set_xlabel("Class")
    ax.set_ylabel("Class")
    ax.set_title(title)

    # Draw grid lines at class boundaries
    ax.grid(False)  # Disable default grid
    for i, b in enumerate(boundaries[:-1]):
        b = b.item() - 0.5
        # if i % 2 == 0:
        ax.axhline(b, color="white", linewidth=0.1)
        ax.axvline(b, color="white", linewidth=0.1)

    # Optional: minor grid around every sample
    ax.set_xticks(torch.arange(n) - 0.5, minor=True)
    ax.set_yticks(torch.arange(n) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.15, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.show()

    return K_sorted, y_sorted


def classes_equilibrated_samples(X, y, samples_per_class=5):
    """
    Select a balanced subset of samples from the dataset, ensuring an equal number of samples per class.
    """
    classes = torch.unique(y)
    selected_indices = []

    for cls in classes:
        cls_indices = torch.where(y == cls)[0]
        if len(cls_indices) < samples_per_class:
            raise ValueError(f"Not enough samples for class {cls.item()}. Requested {samples_per_class}, but only {len(cls_indices)} available.")
        selected_cls_indices = cls_indices[torch.randperm(len(cls_indices))[:samples_per_class]]
        selected_indices.append(selected_cls_indices)

    selected_indices = torch.cat(selected_indices)
    return X[selected_indices], y[selected_indices]


def explore_spectral_properties(
    model_type: str = "CNN",
    subset_size: int = 10,
    subset_samples_per_class: int = 0,
    seed: int = 42,
):
    """
    Explore the spectral properties of the NTK for different widths of a CNN model.
    Compute Eigenvalues of the NTK for different widths and visualize the distribution of eigenvalues.
    Note: eigenvalues are also singular values of the NTK matrix, since the NTK is symmetric and positive semi-definite.
    """
    
    # dataset = get_raw_mnist_data()
    dataset = get_lowres_mnist_data(seed=seed)

    X_train, X_test, y_train, y_test = dataset
    y_train = y_train.argmax(dim=1)
    y_test = y_test.argmax(dim=1)

    if subset_samples_per_class > 0: 
        sub_X, sub_y = classes_equilibrated_samples(X_train, y_train, samples_per_class=subset_samples_per_class)
    else:
        sub_X = X_train[:subset_size] 
        sub_y = y_train[:subset_size]

    sub_X = sub_X - sub_X.mean(dim=0, keepdim=True)  # Center the data
    print(f"Subset of train X shape: {sub_X.shape}")
    print(f"Subset of train y shape: {sub_y.shape}")
    # sub_X = sub_X / sub_X.std(dim=0, keepdim=True)  # Normalize the data


    theoretical_ntk_func = None
    match model_type:
        case "CNN":
            theoretical_ntk_func = get_cnn_theoretical_ntk_function(
                depth=3,
                k=3,
                sigma_w=1.0,
                sigma_b=0.1,
                implemented_phi="relu",
            )
        case "MLP":
            theoretical_ntk_func = get_mlp_theoretical_ntk_function(
                input_shape=sub_X.shape[1:],
                output_dim=1,
                depth=3,
                sigma_w=1.0,
                beta=0.1,
                implemented_sigma="relu",
            )
        case _:
            raise ValueError(f"Unknown model type: {model_type}")
    
    K = compute_gram_matrix(theoretical_ntk_func, sub_X)
    # print(f"Check if K is symmetric: {torch.allclose(K, K.T)}")

    eigvalues, eigvecs = torch.linalg.eigh(K)

    print("="*50)
    print(f"Eigenvalues of the NTK Gram matrix ({model_type}): {eigvalues}")
    print("condition number:", eigvalues[-1] / eigvalues[0].clamp_min(1e-12))
    print("effective rank:", (eigvalues.sum() ** 2) / (eigvalues.square().sum()))
    print("top energy share:", eigvalues[-1] / eigvalues.sum())

    top_vec = eigvecs[:, -1]

    ones = torch.ones_like(top_vec)
    cos = torch.abs(torch.dot(top_vec, ones) / (top_vec.norm() * ones.norm()))
    print("alignment with constant vector:", cos.item())
    
    # Plot the eigenvalue distribution
    plt.figure(figsize=(10, 6))
    plt.hist(eigvalues.numpy(), bins=50, density=True, alpha=0.7, color='blue')
    plt.title(f"Eigenvalue Distribution of NTK Gram matrix ({model_type})")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Density")
    plt.show()

    Kc = center_kernel(K)
    Kc = 0.5 * (Kc + Kc.T)

    eigvals_c, eigvecs_c = torch.linalg.eigh(Kc)
    print(eigvals_c)
    print("="*50)
    print("largest centered:", eigvals_c[-1])
    print("centered condition number:", eigvals_c[-1] / eigvals_c[0].clamp_min(1e-12))
    print("effective rank centered:", eigvals_c.sum().square() / eigvals_c.square().sum())
    print("Eigenvalues of centered NTK Gram matrix:", eigvals_c)

    plt.figure(figsize=(10, 6))
    plt.hist(eigvals_c.numpy(), bins=50, density=True, alpha=0.7, color='green')
    plt.title(f"Eigenvalue Distribution of Centered NTK Gram matrix ({model_type})")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Density")
    plt.show()

    stats_raw = within_between_kernel_stats(K, sub_y, normalize=True, center=False)
    stats_centered = within_between_kernel_stats(K, sub_y, normalize=True, center=True)

    print("Raw:", stats_raw)
    print("Centered:", stats_centered)

    # plot_sorted_kernel(K, sub_y, title=f"NTK Gram matrix sorted by class ({model_type})", center=False, normalize=True)
    plot_sorted_kernel_by_class(K, sub_y, title=f"NTK Gram matrix sorted by class ({model_type})", center=False, normalize=True)
    plot_sorted_kernel_by_class(Kc, sub_y, title=f"Centered NTK Gram matrix sorted by class ({model_type})", center=False, normalize=True)






if __name__ == "__main__":
    widths = [16, 32, 64, 128, 256, 512]
    subset_size = 50
    per_class_samples = 20
    seed = 45

    explore_spectral_properties(
        model_type="CNN",
        subset_size=subset_size,
        subset_samples_per_class=per_class_samples,
        seed=seed,
    )
    explore_spectral_properties(
        model_type="MLP",
        subset_size=subset_size,
        subset_samples_per_class=per_class_samples,
        seed=seed,
    )