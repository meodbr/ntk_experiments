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
from ntk_experiments.usage.dataset import (
    get_raw_mnist_data,
    plot_mnist_sample,
    get_lowres_mnist_data,
    plot_lowres_mnist_sample,
)
from ntk_experiments.utils.norms import frobenius_relative_change
from ntk_experiments.utils.gram import compute_gram_matrix

sns.set_theme()


def translate_zero_pad(x, dx=0, dy=0):
    """
    x: (H, W) tensor
    dx > 0: shift right
    dy > 0: shift down
    """
    print(f"translate_zero_pad: input shape: {x.shape}, dx={dx}, dy={dy}")
    if len(x.shape) == 3:
        x = x.squeeze(0)  # remove channel dim if present
    H, W = x.shape
    out = torch.zeros_like(x)

    src_x0 = max(0, -dx)
    src_x1 = min(W, W - dx) if dx >= 0 else W
    dst_x0 = max(0, dx)
    dst_x1 = min(W, W + dx) if dx <= 0 else W

    src_y0 = max(0, -dy)
    src_y1 = min(H, H - dy) if dy >= 0 else H
    dst_y0 = max(0, dy)
    dst_y1 = min(H, H + dy) if dy <= 0 else H

    out[dst_y0:dst_y1, dst_x0:dst_x1] = x[src_y0:src_y1, src_x0:src_x1]
    return out


def hflip(x):
    return torch.flip(x, dims=[1])


def vflip(x):
    return torch.flip(x, dims=[0])


def build_invariance_panel(X_test, y_test, base_idx=0, other_indices=None):
    """
    Returns:
        images: list of (28,28) tensors
        labels: list of strings
    """
    x = X_test[base_idx]

    images = []
    labels = []

    # original + translations
    translations = [
        (0, 0),
        (4, 4),
        (-4, 4),
        (-4, -4),
        (4, -4),
    ]

    for dx, dy in translations:
        images.append(translate_zero_pad(x, dx=dx, dy=dy))
        labels.append(f"T({dx},{dy})")

    # flips
    # images.append(hflip(x))
    # labels.append("Hflip")

    # images.append(vflip(x))
    # labels.append("Vflip")

    # unrelated images
    if other_indices is None:
        other_indices = [1, 2, 3, 4, 5]

    for idx in other_indices:
        images.append(X_test[idx])
        labels.append(f"other:{int(y_test[idx])}")

    return images, labels


def compute_kernel_matrix(images, ntk_kernel, flatten=True, device=None):
    n = len(images)
    K = torch.zeros((n, n), dtype=torch.float32)

    processed = []
    for x in images:
        if flatten:
            x_in = x.reshape(1, -1)
        else:
            x_in = x.unsqueeze(
                0
            )  # e.g. (1, 28, 28) or adapt if CNN expects channel dim

        if device is not None:
            x_in = x_in.to(device)

        processed.append(x_in)

    for i in range(n):
        for j in range(i, n):
            val = ntk_kernel(processed[i], processed[j])
            if isinstance(val, torch.Tensor):
                val = val.squeeze().detach().cpu().item()
            K[i, j] = val
            K[j, i] = val

    return K


def normalize_kernel_matrix(K, eps=1e-12):
    d = torch.sqrt(torch.clamp(torch.diag(K), min=eps))
    C = K / (d[:, None] * d[None, :])
    return C


def add_image_ticks(ax, images, zoom=0.8):
    n = len(images)

    # remove default labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # x-axis image ticks
    for i, img in enumerate(images):
        print(f"add_image_ticks: y-axis image {i} shape: {img.shape}")
        arr = img.detach().cpu().numpy().squeeze()
        print(f"add_image_ticks: y-axis image {i} shape: {arr.shape}")
        oi = OffsetImage(arr, zoom=zoom, cmap="gray")
        ab = AnnotationBbox(
            oi, (i, -1.1), frameon=False, box_alignment=(0.5, 1.0), xycoords="data"
        )
        ax.add_artist(ab)

    # y-axis image ticks
    for i, img in enumerate(images):
        print(f"add_image_ticks: y-axis image {i} shape: {img.shape}")
        arr = img.detach().cpu().numpy().squeeze()
        print(f"add_image_ticks: y-axis image {i} shape: {arr.shape}")
        oi = OffsetImage(arr, zoom=zoom, cmap="gray")
        ab = AnnotationBbox(
            oi, (-0.6, i), frameon=False, box_alignment=(1.0, 0.5), xycoords="data"
        )
        ax.add_artist(ab)

    ax.set_xlim(-1.2, n - 0.5)
    ax.set_ylim(n - 0.5, -1.2)


def plot_ntk_invariance_matrix(
    X_test,
    y_test,
    ntk_kernel,
    base_idx=0,
    other_indices=None,
    flatten=True,
    device=None,
    figsize=(8, 8),
    model_type="MLP",
):
    images, labels = build_invariance_panel(
        X_test, y_test, base_idx=base_idx, other_indices=other_indices
    )

    K = compute_kernel_matrix(images, ntk_kernel, flatten=flatten, device=device)
    C = normalize_kernel_matrix(K)
    min_C = torch.min(C)
    C = (C - torch.min(C)) / (torch.max(C) - torch.min(C))

    # fig, ax = plt.subplots(figsize=figsize)
    # im = ax.imshow(C.numpy(), vmin=0, vmax=1)
    # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    with plt.rc_context({
        "axes.grid": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.6,
        "xtick.bottom": False,
        "ytick.left": False,
    }):
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            C.numpy(),
            vmin=0,
            vmax=1,
            cmap="gray_r",
            interpolation="nearest"
        )

        cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        cbar.outline.set_linewidth(0.5)

    add_image_ticks(ax, images, zoom=0.7)

    # optional separators between groups
    n_trans = 5
    n_flips = 0
    sep1 = n_trans - 0.5
    sep2 = n_trans + n_flips - 0.5

    # ax.axhline(sep1, linewidth=1)
    # ax.axvline(sep1, linewidth=1)
    # ax.axhline(sep2, linewidth=1)
    # ax.axvline(sep2, linewidth=1)
    for sep in [sep1, sep2]:
        ax.axhline(sep, color="black", linewidth=0.6)
        ax.axvline(sep, color="black", linewidth=0.6)

    ax.set_title(f"Normalized NTK correlation ({model_type})")
    plt.tight_layout()
    plt.show()


    return K, C, images, labels


if __name__ == "__main__":

    seed = 47
    subset_size = 50

    dataset = get_raw_mnist_data(seed=seed)
    X_train, X_test, y_train, y_test = dataset

    # sample_random_indices = torch.randperm(X_train.size(0))[:3]
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    # sub_X_train = torch.randperm(X_train.size(0))[:subset_size]
    sub_X_train = X_train[:subset_size]
    sub_y_train = y_train[:subset_size]
    # Pad every image in X_test to 32x32
    print(f"sub_X_train shape before padding: {sub_X_train.shape}")
    sub_X_train = torch.nn.functional.pad(sub_X_train, (4, 4, 4, 4), mode='constant', value=0)
    print(f"sub_X_train shape after padding: {sub_X_train.shape}")


    kernel_func = get_mlp_theoretical_ntk_function(
        input_shape=sub_X_train.shape[1:],
        output_dim=1,
        depth=3,
        beta=0.1,
        sigma_w=1.0,
        implemented_sigma="relu",
    )

    K, C, images, labels = plot_ntk_invariance_matrix(
        sub_X_train,
        sub_y_train,
        ntk_kernel=kernel_func,
        flatten=False,  # if your kernel expects image shape
        model_type="MLP",
    )

    cnn_kernel_func = get_cnn_theoretical_ntk_function(
        depth=3,
        k=3,
        sigma_w=1.0,
        sigma_b=0.1,
        implemented_phi="relu",
    )

    K, C, images, labels = plot_ntk_invariance_matrix(
        sub_X_train,
        sub_y_train,
        ntk_kernel=cnn_kernel_func,
        flatten=False,  # if your kernel expects image shape
        model_type="CNN",
    )
