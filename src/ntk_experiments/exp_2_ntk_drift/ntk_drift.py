"""
Goal of the module : show how the finite-width NTK changes during training, and how the bigger the width, the more stable it is. This is a demonstration of the NTK drift phenomenon.
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
    get_practical_model_ntk_function,
)

from ntk_experiments.CNN.practical.cnn_model import NTKCNN
from ntk_experiments.MLP.practical.ntkmlp_model import NTKMLP

from ntk_experiments.usage.train import train_step
from ntk_experiments.usage.dataset import get_raw_mnist_data, plot_mnist_sample, get_lowres_mnist_data, plot_lowres_mnist_sample
from ntk_experiments.utils.norms import frobenius_relative_change
from ntk_experiments.utils.gram import compute_gram_matrix

sns.set_theme()


def plot_cnn_ntk_drift(
    seed: int = 42,
    widths: list = None,
    training_steps: int = 1000,
    evaluation_interval: int = 10,
):

    dataset = get_lowres_mnist_data(seed=seed)
    train_X, test_X, train_y, test_y = dataset
    subset_of_train_X = train_X[:10]
    print(f"Subset of train X shape: {subset_of_train_X.shape}")
    print(f"Subset of train y shape: {train_y[:10].shape}")
    print(f"Subset of train y: {train_y[:10]}")

    # Batched X, y for training with dataloaders
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(f"Train loader length: {len(train_loader)}")
    print(f"Train loader batch size: {train_loader.batch_size}")
    print(f"Train loader dataset length: {len(train_loader.dataset)}")

    per_width_results = {}

    for width in widths:
        gram_matrices = []
        relative_frobenius_changes = []
        steps = []
        losses = []
        model = NTKCNN(
            input_dim=1,
            output_dim=10,
            depth=3,
            kernel_size=3,
            width=width,
            beta=1.0,
            sigma_w=1.0,
            sigma_b=1.0,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        step = 0
        while step < training_steps:
            for X_batch, y_batch in tqdm(train_loader, total=training_steps, desc=f"Training width {width}"):
                step += 1
                if step % evaluation_interval == 0:
                    with torch.no_grad():
                        practical_ntk_func = get_practical_model_ntk_function(model)
                        gram_matrix = compute_gram_matrix(practical_ntk_func, subset_of_train_X)
                        gram_matrices.append(gram_matrix)
                        relative_change = frobenius_relative_change(gram_matrices[0], gram_matrix)
                        relative_frobenius_changes.append(relative_change)
                        steps.append(step)

                
                train_step(model, X_batch, y_batch, opt=opt, loss_fn=loss_fn)

                if step >= training_steps:
                    print(f"Reached maximum training steps: {step}/{training_steps}. Stopping training for width {width}.")
                    break

        print(f"Finished training for width {width}. Total steps: {step+1}")
        print(f"Collected {len(gram_matrices)} Gram matrices for width {width}.")
        print(f"Example Gram matrix at last evaluation step:\n{gram_matrices[-1]}")
        print(f"Relative Frobenius changes for width {width}: {relative_frobenius_changes}")

        per_width_results[width] = {
            "gram_matrices": gram_matrices,
            "relative_frobenius_changes": relative_frobenius_changes,
            "steps": steps,
        }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for width, results in per_width_results.items():
        ax.plot(results["steps"], results["relative_frobenius_changes"], label=f"Width {width}")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Relative Frobenius Change")
    ax.set_title("NTK Drift Analysis")
    ax.set_yscale("log")
    ax.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    for width, results in per_width_results.items():
        ax.plot(results["steps"], results["losses"], label=f"Width {width}")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Over Time")
    ax.set_yscale("log")
    ax.legend()


def plot_mlp_ntk_drift(
    seed: int = 42,
    widths: list = None,
    training_steps: int = 1000,
    evaluation_interval: int = 10,
):
    # Similar implementation as plot_cnn_ntk_drift but for MLPs
    dataset = get_lowres_mnist_data(seed=seed)
    train_X, test_X, train_y, test_y = dataset
    train_X = train_X.view(train_X.size(0), -1)  # Flatten for MLP
    test_X = test_X.view(test_X.size(0), -1)  # Flatten for MLP
    subset_of_train_X = train_X[:10]
    print(f"Subset of train X shape: {subset_of_train_X.shape}")
    print(f"Subset of train y shape: {train_y[:10].shape}")
    print(f"Subset of train y: {train_y[:10]}")

    per_width_results = {}

    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(f"Train loader length: {len(train_loader)}")
    print(f"Train loader batch size: {train_loader.batch_size}")
    print(f"Train loader dataset length: {len(train_loader.dataset)}")

    for width in widths:
        gram_matrices = []
        relative_frobenius_changes = []
        steps = []
        losses = []
        model = NTKMLP(
            input_dim=64,
            output_dim=10,
            width=width,
            depth=3,
            beta=1.0,
            sigma_w=1.0,
            sigma_b=1.0,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        step = 0
        while step < training_steps:
            for X_batch, y_batch in tqdm(train_loader, total=training_steps, desc=f"Training width {width}"):
                step += 1
                if step % evaluation_interval == 0:
                    with torch.no_grad():
                        practical_ntk_func = get_practical_model_ntk_function(model, flatten_input=True)
                        gram_matrix = compute_gram_matrix(practical_ntk_func, subset_of_train_X)
                        gram_matrices.append(gram_matrix)
                        relative_change = frobenius_relative_change(gram_matrices[0], gram_matrix)
                        relative_frobenius_changes.append(relative_change)
                        steps.append(step)

                
                train_step(model, X_batch, y_batch, opt=opt, loss_fn=loss_fn)

                if step >= training_steps:
                    print(f"Reached maximum training steps: {step}/{training_steps}. Stopping training for width {width}.")
                    break

        print(f"Finished training for width {width}. Total steps: {step+1}")
        print(f"Collected {len(gram_matrices)} Gram matrices for width {width}.")
        print(f"Example Gram matrix at last evaluation step:\n{gram_matrices[-1]}")
        print(f"Relative Frobenius changes for width {width}: {relative_frobenius_changes}")

        per_width_results[width] = {
            "gram_matrices": gram_matrices,
            "relative_frobenius_changes": relative_frobenius_changes,
            "losses": losses,
            "steps": steps,
        }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for width, results in per_width_results.items():
        ax.plot(results["steps"], results["relative_frobenius_changes"], label=f"Width {width}")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Relative Frobenius Change")
    ax.set_title("NTK Drift Analysis")
    ax.set_yscale("log")
    ax.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    for width, results in per_width_results.items():
        ax.plot(results["steps"], results["losses"], label=f"Width {width}")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Over Time")
    ax.set_yscale("log")
    ax.legend()
    plt.show()


def plot_ntk_drift_for_widths(
    seed: int = 42,
    widths: list = None,
    sample_per_width: int = 10,
    subset_size: int = 10,
    model_type: str = "CNN",  # or "MLP"
):
    """
    Plot the NTK drift for different widths of a model.
    """
    dataset = get_lowres_mnist_data(seed=seed)
    train_X, test_X, train_y, test_y = dataset
    subset_of_train_X = train_X[:subset_size]

    match model_type:
        case "CNN":
            theoretical_ntk_func = get_cnn_theoretical_ntk_function(
                depth=3,
                k=3,
                sigma_w=1.0,
                sigma_b=1.,
                implemented_phi="relu",
            )
        case "MLP":
            theoretical_ntk_func = get_mlp_theoretical_ntk_function(
                input_shape=subset_of_train_X.shape[1:],
                output_dim=1,
                depth=3,
                beta=1.0,
                sigma_w=1.0,
                implemented_sigma="relu",
            )
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")

    gram_matrix_theoretical = compute_gram_matrix(theoretical_ntk_func, subset_of_train_X)


    if widths is None:
        widths = [16, 32, 64, 128, 256, 512]

    width_results = []
    width_mean_relative_changes = []
    width_std_relative_changes = []

    for width in widths:
        relative_changes = []

        for sample_idx in tqdm(range(sample_per_width), desc=f"Width {width}"):
            match model_type:
                case "CNN":
                    model = NTKCNN(
                        input_dim=1,
                        output_dim=10,
                        depth=3,
                        kernel_size=3,
                        width=width,
                        beta=1.,
                        sigma_w=1.0,
                        sigma_b=1.,
                    )
                case "MLP":
                    model = NTKMLP(
                        input_dim=64,
                        output_dim=10,
                        width=width,
                        depth=3,
                        beta=1.0,
                        sigma_w=1.0,
                        sigma_b=1.,
                    )
                case _:
                    raise ValueError(f"Unsupported model type: {model_type}")

            practical_ntk_func = get_practical_model_ntk_function(model)
            gram_matrix = compute_gram_matrix(practical_ntk_func, subset_of_train_X)
            print("Gram matrix theoretical:", gram_matrix_theoretical)
            print("Gram matrix practical:", gram_matrix)
            relative_change = frobenius_relative_change(gram_matrix_theoretical, gram_matrix)
            relative_changes.append(relative_change)

        width_results.append(relative_changes)
        width_mean_relative_changes.append(np.mean(relative_changes))
        width_std_relative_changes.append(np.std(relative_changes))
        print(f"Width {width}: Mean Relative Change = {width_mean_relative_changes[-1]}, Std Dev = {width_std_relative_changes[-1]}")
    

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.errorbar(
    #     widths,
    #     width_mean_relative_changes,
    #     yerr=width_std_relative_changes,
    #     fmt='-o',
    #     capsize=5,
    #     label='Mean Relative Change ± Std Dev on 10 samples',
    # )
    # ax.set_xlabel("Width of the Model")
    # ax.set_ylabel("Relative Frobenius Change")
    # ax.set_title("NTK Drift Analysis Across Widths")
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.legend()

    fig, ax = plt.subplots(figsize=(8, 5))

    mean = np.array(width_mean_relative_changes)
    std = np.array(width_std_relative_changes)

    lower = np.maximum(mean - std, 1e-12)
    upper = mean + std

    ax.plot(widths, mean, "-o", linewidth=2, markersize=5, label="Mean")

    ax.fill_between(
        widths,
        lower,
        upper,
        color="lightskyblue",
        alpha=0.25,
        label="±1 std",
    )

    ax.set_xlabel("Width")
    ax.set_ylabel("Relative Frobenius change")
    ax.set_title(f"NTK Drift Across Widths ({model_type})")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    plt.show()
 


if __name__ == "__main__":
    # Plot 3 mnist samples
    seed = 47
    subset_size = 50
    torch.manual_seed(seed)  # Set seed for reproducibility
    X_train, X_test, y_train, y_test = get_raw_mnist_data(seed=seed)#config.SEED)

    widths = [10, 50, 100, 200, 500]
    trial_training_steps = 1000
    evaluation_interval = 10

    # widths_no_training = [8, 16, 32, 64, 128, 256, 512]
    # widths_no_training = np.arrange(10, 500, 20).tolist()
    # arange logarithmically spaced widths for better visualization
    # widths_no_training = np.logspace(1, 2.7, num=20, dtype=int).tolist()  # Widths from 10 to ~500
    widths_no_training = np.logspace(1, 1.9, num=20, dtype=int).tolist()  # Widths from 10 to ~500
    widths_no_training[0] = 300

    # plot_mlp_ntk_drift(
    #     seed=seed,
    #     widths=widths,
    #     training_steps=trial_training_steps,
    #     evaluation_interval=evaluation_interval,
    # )
    # plot_cnn_ntk_drift(
    #     seed=seed,
    #     widths=widths,
    #     training_steps=trial_training_steps,
    #     evaluation_interval=evaluation_interval,
    # )

    # plot_ntk_drift_for_widths(
    #     seed=seed,
    #     widths=widths_no_training,
    #     sample_per_width=10,
    #     subset_size=10,
    #     model_type="MLP"
    # )
    plot_ntk_drift_for_widths(
        seed=seed,
        widths=widths_no_training,
        sample_per_width=10,
        subset_size=10,
        model_type="CNN"
    )

