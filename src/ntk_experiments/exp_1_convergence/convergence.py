"""
Goal of the module : show the convergence of the empirical NTK to the theoretical NTK as the width of the network increases.
"""

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
)

from ntk_experiments.usage.dataset import get_raw_mnist_data, plot_mnist_sample, get_raw_lowres_mnist_data, plot_lowres_mnist_sample

sns.set_theme()


def add_image_xticks(ax, images, positions, zoom=0.65, y_offset=-0.14):
    """
    Add image tick labels at arbitrary x positions.
    """

    ax.set_xticks(positions)
    ax.set_xticklabels([""] * len(positions))
    ax.tick_params(axis="x", length=0)

    for img, x in zip(images, positions):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()

            if img.ndim == 1:
                img = img.reshape(28, 28)

            if img.ndim == 3:
                img = img.squeeze(0)

            img = img.numpy()

        imagebox = OffsetImage(img, cmap="gray", zoom=zoom)

        ab = AnnotationBbox(
            imagebox,
            (x, 0),
            xybox=(x, y_offset),
            xycoords=("data", "axes fraction"),
            boxcoords=("data", "axes fraction"),
            frameon=False,
            pad=0,
        )

        ax.add_artist(ab)

    plt.subplots_adjust(bottom=0.28)


def linear_interpolation_from_list(x_list, total_points=100):
    """
    Given a list of points x_list, return a list of points that are linearly interpolated between the points in x_list.
    The number of points returned is total_points.
    """
    num_points = total_points // (len(x_list) - 1)
    x_list = [x.detach().cpu().numpy() for x in x_list]
    # Ensure middle value is in the middle of the interpolation
    x_interp = []
    for i in range(len(x_list) - 1):
        x_start = x_list[i]
        x_end = x_list[i + 1]
        for j in range(num_points):
            alpha = j / num_points
            x_interp.append((1 - alpha) * x_start + alpha * x_end)
    x_interp.append(x_list[-1])  # Add the last point
    x_interp = np.array(x_interp)


    return torch.arange(len(x_interp)), torch.tensor(x_interp)


def plot_theoretical_ntk_along_path(x_list, total_points=100):
    """
    Given a list of points x_list, plot the theoretical NTK along the path defined by x_list.
    The number of points returned is total_points.
    """
    x_ref = x_list[len(x_list) // 2]  # Use the middle point as the reference point
    print("x_ref shape:", x_ref.shape)
    x_axis, x_interp = linear_interpolation_from_list(x_list, total_points=total_points)
    theo_ntk_func = get_mlp_theoretical_ntk_function(
        input_shape=x_interp[0].shape,
        output_dim=1,
        depth=3,
        sigma_w=1.0,
        beta=1.0,
        implemented_sigma="relu",
    )
    ntk_values = []
    for i in range(len(x_interp)):
        ntk_values.append(theo_ntk_func(x_interp[i], x_ref).item())
    ntk_values = np.array(ntk_values)

    fig, ax = plt.subplots()
    ax.plot(x_axis, ntk_values)
    ax.set_title(
        "Theoretical NTK from random noise to an input sample",
    )
    ax.set_xlabel("Interpolated Points")
    ax.set_ylabel("Theoretical NTK Value")
    plt.show()

def compare_theoretical_and_practical_ntk_along_path(x_list, widths, total_points=100):
    """
    Given a list of points x_list, plot the theoretical and practical NTK along the path defined by x_list.
    The number of points returned is total_points.
    """
    x_ref = x_list[len(x_list) // 2]  # Use the middle point as the reference point
    print("x_ref shape:", x_ref.shape)
    x_axis, x_interp = linear_interpolation_from_list(x_list, total_points=total_points)
    theo_ntk_func = get_mlp_theoretical_ntk_function(
        input_shape=x_interp[0].shape,
        output_dim=1,
        depth=3,
        sigma_w=1.0,
        beta=1.0,
        implemented_sigma="relu",
    )
    ntk_values_theo = []
    for i in range(len(x_interp)):
        ntk_values_theo.append(theo_ntk_func(x_interp[i], x_ref).item())
    ntk_values_theo = np.array(ntk_values_theo)

    fig, ax = plt.subplots()
    ax.plot(x_axis, ntk_values_theo, label="Theoretical NTK", color="blue")

    for width in widths:
        practical_ntk_func = get_mlp_practical_ntk_function(
            input_shape=x_interp[0].shape,
            output_dim=1,
            depth=3,
            width=width,
            sigma_w=1.0,
            beta=1.0,
            implemented_sigma="relu",
        )
        ntk_values_practical = []
        for i in range(len(x_interp)):
            ntk_values_practical.append(practical_ntk_func(x_interp[i], x_ref).item())
        ntk_values_practical = np.array(ntk_values_practical)
        ax.plot(
            x_axis,
            ntk_values_practical,
            label=f"Practical NTK (width={width})",
            linestyle="--",
        )

    ax.set_title(
        "Theoretical and Practical NTK from random noise to an input sample",
    )
    ax.set_xlabel("Interpolated Points")
    ax.set_ylabel("NTK Value")
    ax.legend(loc="upper right")
    plt.show()


# def compare_cnn_theoretical_and_practical_ntk_along_path(x_list, widths, total_points=100):
#     """
#     Given a list of points x_list, plot the theoretical and practical NTK along the path defined by x_list.
#     The number of points returned is total_points.
#     """
#     x_ref = x_list[len(x_list) // 2]  # Use the middle point as the reference point
#     print("x_ref shape:", x_ref.shape)
#     x_axis, x_interp = linear_interpolation_from_list(x_list, total_points=total_points)
#     theo_ntk_func = get_cnn_theoretical_ntk_function(
#         input_shape=x_interp[0].shape,
#         output_dim=1,
#         depth=3,
#         k=5,
#         sigma_w=1.0,
#         beta=1.0,
#         implemented_sigma="relu",
#     )
#     ntk_values_theo = []
#     for i in range(len(x_interp)):
#         ntk_values_theo.append(theo_ntk_func(x_interp[i], x_ref).item())
#     ntk_values_theo = np.array(ntk_values_theo)

#     fig, ax = plt.subplots()
#     ax.plot(x_axis, ntk_values_theo, label="Theoretical NTK", color="blue")

#     for width in widths:
#         practical_ntk_func = get_cnn_practical_ntk_function(
#             input_shape=x_interp[0].shape,
#             output_dim=1,
#             depth=3,
#             k=5,
#             width=width,
#             sigma_w=1.0,
#             beta=1.0,
#         )
#         ntk_values_practical = []
#         for i in range(len(x_interp)):
#             ntk_values_practical.append(practical_ntk_func(x_interp[i], x_ref).item())
#         ntk_values_practical = np.array(ntk_values_practical)
#         ax.plot(
#             x_axis,
#             ntk_values_practical,
#             label=f"Practical NTK (width={width})",
#             linestyle="--",
#         )

#     ax.set_title(
#         "Theoretical and Practical NTK from random noise to an input sample",
#     )
#     ax.set_xlabel("Interpolated Points")
#     ax.set_ylabel("NTK Value")
#     ax.legend(loc="upper right")
#     plt.show()

def normalize_for_display(x):
    x = x.detach().cpu()

    if x.ndim == 1:
        x = x.reshape(28, 28)

    if x.ndim == 3:
        x = x.squeeze(0)

    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def compare_cnn_theoretical_and_practical_at_point(
    x_selected,
    widths,
    total_points=100,
    kernel_size=5,
):
    """
    Plot theoretical and practical NTK along the path:
    random noise -> selected MNIST sample -> random noise.
    """

    x_ref = x_selected

    noise_left = torch.randn_like(x_ref)
    noise_right = torch.randn_like(x_ref)

    x_list = [noise_left, x_ref, noise_right]

    print("x_ref shape:", x_ref.shape)

    x_axis, x_interp = linear_interpolation_from_list(
        x_list,
        total_points=total_points,
    )

    theo_ntk_func = get_cnn_theoretical_ntk_function(
        depth=3,
        k=kernel_size,
        sigma_w=1.0,
        sigma_b=1.0,
        implemented_phi="relu",
    )

    ntk_values_theo = []

    for i in tqdm(range(len(x_interp)), desc="Computing theoretical NTK"):
        ntk_values_theo.append(theo_ntk_func(x_interp[i], x_ref).item())

    ntk_values_theo = np.array(ntk_values_theo)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        x_axis,
        ntk_values_theo,
        label="Theoretical NTK",
        color="blue",
    )

    for width in widths:
        practical_ntk_func = get_cnn_practical_ntk_function(
            input_shape=x_interp[0].shape,
            output_dim=1,
            depth=3,
            k=kernel_size,
            width=width,
            sigma_w=1.0,
            beta=1.0,
        )

        ntk_values_practical = []

        for i in tqdm(range(len(x_interp)), desc=f"Computing practical NTK (width={width})"):
            ntk_values_practical.append(
                practical_ntk_func(x_interp[i], x_ref).item()
            )

        ntk_values_practical = np.array(ntk_values_practical)

        ax.plot(
            x_axis,
            ntk_values_practical,
            label=f"Practical NTK (width={width})",
            linestyle="--",
        )

    n_ticks = 7  # for example: noise -> ... -> MNIST -> ... -> noise

    tick_indices = np.linspace(
        0,
        len(x_interp) - 1,
        n_ticks,
        dtype=int,
    )

    tick_positions = x_axis[tick_indices]

    tick_images = [
        normalize_for_display(x_interp[i])
        for i in tick_indices
    ]

    add_image_xticks(
        ax,
        images=tick_images,
        positions=tick_positions,
        zoom=0.8,
        y_offset=-0.14,
    )

    ax.set_title(
        "Theoretical and Practical CNN NTK from random noise to an input sample",
    )
    ax.set_xlabel("")
    ax.set_ylabel("NTK Value against the reference sample")
    # ax.legend(loc="upper right")
    ax.legend()

    plt.show()

def compare_mlp_theoretical_and_practical_at_point(
    x_selected,
    widths,
    total_points=100,
):
    """
    Plot theoretical and practical NTK along the path:
    random noise -> selected MNIST sample -> random noise.
    """

    x_ref = x_selected

    noise_left = torch.randn_like(x_ref)
    noise_right = torch.randn_like(x_ref)

    x_list = [noise_left, x_ref, noise_right]

    print("x_ref shape:", x_ref.shape)

    x_axis, x_interp = linear_interpolation_from_list(
        x_list,
        total_points=total_points,
    )

    theo_ntk_func = get_mlp_theoretical_ntk_function(
        input_shape=x_interp[0].shape,
        output_dim=1,
        depth=3,
        sigma_w=1.0,
        beta=1.0,
        implemented_sigma="relu",
    )

    ntk_values_theo = []

    for i in tqdm(range(len(x_interp)), desc="Computing theoretical NTK"):
        ntk_values_theo.append(theo_ntk_func(x_interp[i], x_ref).item())

    ntk_values_theo = np.array(ntk_values_theo)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        x_axis,
        ntk_values_theo,
        label="Theoretical NTK",
        color="blue",
    )

    for width in widths:
        practical_ntk_func = get_mlp_practical_ntk_function(
            input_shape=x_interp[0].shape,
            output_dim=1,
            depth=3,
            width=width,
            sigma_w=1.0,
            beta=1.0,
            implemented_sigma="relu",
        )

        ntk_values_practical = []

        for i in tqdm(range(len(x_interp)), desc=f"Computing practical NTK (width={width})"):
            ntk_values_practical.append(
                practical_ntk_func(x_interp[i], x_ref).item()
            )

        ntk_values_practical = np.array(ntk_values_practical)

        ax.plot(
            x_axis,
            ntk_values_practical,
            label=f"Practical NTK (width={width})",
            linestyle="--",
        )
    
    n_ticks = 7  # for example: noise -> ... -> MNIST -> ... -> noise

    tick_indices = np.linspace(
        0,
        len(x_interp) - 1,
        n_ticks,
        dtype=int,
    )

    tick_positions = x_axis[tick_indices]

    tick_images = [
        normalize_for_display(x_interp[i])
        for i in tick_indices
    ]

    add_image_xticks(
        ax,
        images=tick_images,
        positions=tick_positions,
        zoom=0.8,
        y_offset=-0.14,
    )
    
    ax.set_title(
        "Theoretical and Practical MLP NTK from random noise to an input sample",
    )

    ax.set_xlabel("")
    ax.set_ylabel("NTK Value against the reference sample")
    # ax.legend(loc="upper right")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    # Plot 3 mnist samples
    seed = 47
    torch.manual_seed(seed)  # Set seed for reproducibility
    X_train, X_test, y_train, y_test = get_raw_mnist_data(seed=seed)#config.SEED)
    for i in range(3):
        # plot_mnist_sample(X_train, y_train, index=i)
        pass

    # Plot one 6, one 7, and one 8 sample
    indices = [i for i, label in enumerate(y_train) if label in [6, 7, 8]]
    selected_indices = indices[:3]  # Select the first three indices
    for i in selected_indices:
        # plot_mnist_sample(X_train, y_train, index=i)
        pass

    # Just get a 7 image
    seven_indices = [i for i, label in enumerate(y_train) if label == 7]
    selected_x = X_train[seven_indices[0]]
    
    selected_x_list = [
        torch.randn_like(selected_x),
        selected_x,
        torch.randn_like(selected_x),
    ]
    widths = [1000, 2000, 5000, 10000]
    widths_cnn = [1000, 2000, 5000]

    # print("Selected indices for plotting:", selected_indices)
    # print("Selected labels for plotting:", y_train[selected_indices].tolist())
    # plot_theoretical_ntk_along_path(
    #     selected_x_list,
    #     total_points=100,
    # )
    # compare_theoretical_and_practical_ntk_along_path(
    #     selected_x_list,
    #     widths=widths,
    #     total_points=40,
    # )

    # compare_cnn_theoretical_and_practical_at_point(
    #     selected_x,
    #     widths=widths_cnn,
    #     total_points=34,
    # )
    # compare_mlp_theoretical_and_practical_at_point(
    #     selected_x,
    #     widths=widths,
    #     total_points=34,
    # )

    # Same for lowres MNIST
    X_train_lowres, X_test_lowres, y_train_lowres, y_test_lowres = get_raw_lowres_mnist_data(seed=seed)
    seven_indices_lowres = [i for i, label in enumerate(y_train_lowres) if label == 7]
    selected_x_lowres = torch.tensor(X_train_lowres[seven_indices_lowres[0]]).view(1, 8, 8).float()  # Reshape to (1, 8, 8) and convert to float
    print("Selected lowres x shape:", selected_x_lowres.shape)
    compare_cnn_theoretical_and_practical_at_point(
        selected_x_lowres,
        widths=widths_cnn,
        total_points=36,
        kernel_size=3,
    )
    compare_mlp_theoretical_and_practical_at_point(
        selected_x_lowres,
        widths=widths,
        total_points=36,
    )
