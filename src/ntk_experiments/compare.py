import numpy as np
import pandas as pd

import torch
from torch import nn

import seaborn as sns
import matplotlib.pyplot as plt

from .config import config
from .random_walk import random_walk_unit_sphere, unit_sphere
from .theoretical_ntk import infinite_width_ntk, relu, relu_prime
from .empirical_ntk import empirical_ntk, reshape_to_2D_jacobian
from .ntkmlp_model import NTKMLP
from .train import train_model
from .dataset import get_dataset

sns.set_theme()

def compare_empirical_theoretical_ntk_on_sample(width):
    model = NTKMLP(input_dim=config.INPUT_DIM, width=width, depth=4, beta=config.BETA)
    # model = MLP_classic(width)
    dataset = get_dataset(config.DATASET)
    X_train, X_test, y_train, y_test = dataset

    x = X_train[0:1] # Shape: (1, 64)
    x_prime = X_train[1:2] # Shape: (1, 64)

    empirical = empirical_ntk(model, x, x_prime).item()
    theoretical = infinite_width_ntk(
        x=x.view(-1),
        xp=x_prime.view(-1),
        depth=4,
        sigma=relu,
        sigma_prime=relu_prime,
        sigma_w=np.sqrt(2.0),
        beta=config.BETA,
        n_gh=40,
    )[0]

    print("Empirical NTK:", empirical)
    print("Theoretical NTK:", theoretical)

def compare_empirical_theoretical_ntk_on_circle(widths, steps=25):
    models = [NTKMLP(input_dim=config.INPUT_DIM, width=width, depth=4, beta=config.BETA) for width in widths]
    # traj = random_walk_unit_sphere(dim=config.INPUT_DIM, steps=100, step_size=0.05, seed=42)
    gamma, traj = unit_sphere(dim=config.INPUT_DIM, steps=steps, seed=42)
    traj = torch.tensor(traj, dtype=torch.float32)# .unsqueeze(1)  # Shape: (steps, 1, dim)
    print("Trajectory shape:", traj.shape)

    empirical_ntks_per_model = []
    theoretical_ntks = []

    x = torch.zeros((config.INPUT_DIM, ), dtype=torch.float32)
    x[0] = 1.0  # Compare every point on the trajectory to this fixed reference point
    print(x)

    for i in range(len(traj)):
        x_prime = traj[i]
        print(f"x_prime shape at step {i}:", x_prime.shape)
        print(f"x shape: {x.shape}")

        theoretical = infinite_width_ntk(
            x=x,
            xp=x_prime,
            depth=4,
            # sigma=relu,
            # sigma_prime=relu_prime,
            implemented_sigma="relu",
            sigma_w=1.0,
            beta=config.BETA,
            n_gh=40,
        )[0]

        theoretical_ntks.append(theoretical)

    for model in models:
        empirical_ntks = []
        for i in range(len(traj)):
            x_prime = traj[i]
            empirical = empirical_ntk(model, torch.tensor(x).unsqueeze(0), torch.tensor(x_prime).unsqueeze(0)).item()
            empirical_ntks.append(empirical)
        empirical_ntks_per_model.append(empirical_ntks)

    plt.figure(figsize=(10, 5))
    for empirical_ntks, width in zip(empirical_ntks_per_model, widths):
        plt.plot(gamma, empirical_ntks, label=f"Finite-width NTK width={width}", linestyle='dashed')
    plt.plot(gamma, theoretical_ntks, label='Theoretical infinite-width NTK', color="red")
    plt.title('Empirical finite-width vs Theoretical infinite-width NTK along unit circle')
    plt.xlabel('Gamma (angle along circle)')
    plt.ylabel('NTK Value')
    plt.legend()
    plt.show()

def compare_empirical_theoretical_during_training(width):
    model = NTKMLP(input_dim=config.INPUT_DIM, width=width, depth=4, beta=config.BETA)
    dataset = get_dataset(config.DATASET)
    gamma, traj = unit_sphere(dim=config.INPUT_DIM, steps=100, seed=config.SEED)

    epoch_step = 40
    epochs = np.arange(0, 6) * epoch_step
    x = torch.zeros((config.INPUT_DIM, ), dtype=torch.float32)
    x[0] = 1.0  # Compare every point on the trajectory to this fixed reference point

    theoretical_ntks = []
    empirical_ntks_per_num_epoch = []

    for i in range(len(traj)):
        x_prime = traj[i]
        theoretical = infinite_width_ntk(
            x=x,
            xp=x_prime,
            depth=4,
            # sigma=relu,
            # sigma_prime=relu_prime,
            implemented_sigma="relu",
            sigma_w=1.0,
            beta=config.BETA,
            n_gh=40,
        )[0]

        theoretical_ntks.append(theoretical)
    
    for num_epoch in epochs:
        empirical_ntks = []
        for i in range(len(traj)):
            x_prime = traj[i]
            empirical = empirical_ntk(model, torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.tensor(x_prime, dtype=torch.float32).unsqueeze(0)).item()
            empirical_ntks.append(empirical)
        empirical_ntks_per_num_epoch.append(empirical_ntks)
        print(f"Training for {epoch_step} epochs...")
        acc = train_model(model, dataset=config.DATASET, epochs=num_epoch)
        print(f"Epochs: {num_epoch}, Accuracy: {acc:.4f}")

    
    plt.figure(figsize=(10, 5))
    for empirical_ntks, num_epoch in zip(empirical_ntks_per_num_epoch, epochs):
        plt.plot(gamma, empirical_ntks, label=f"Finite-width NTK after {num_epoch} epochs", linestyle='dashed')
    plt.plot(gamma, theoretical_ntks, label='Theoretical infinite-width NTK', color="red")
    plt.title(f'Theoretical constant NTK vs Empirical NTK of width {width} during training')
    plt.xlabel('Gamma (angle along circle)')
    plt.ylabel('NTK Value')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    widths = [100, 500, 1000, 5000]
    # acc = train_model(width)
    print(f"Comparing empirical and theoretical NTK for widths {widths}...")

    # compare_empirical_theoretical_ntk_on_sample(width)
    # compare_empirical_theoretical_ntk_on_circle(widths, steps=100)
    compare_empirical_theoretical_during_training(width=1000)