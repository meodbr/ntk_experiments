import math

import numpy as np
import pandas as pd

import torch
from torch import nn

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from .config import config
from .utils.random_walk import random_walk_unit_sphere, unit_sphere
from .MLP.theory.theoretical_ntk import infinite_width_ntk, relu, relu_prime
from .usage.empirical_ntk import empirical_ntk
from .MLP.practical.ntkmlp_model import NTKMLP

sns.set_theme()

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

    for i in tqdm(range(len(traj)), desc="Computing theoretical NTKs"):
        x_prime = traj[i]
        # print(f"x_prime shape at step {i}:", x_prime.shape)
        # print(f"x shape: {x.shape}")
        print(f"x, x_prime : {x}, {x_prime}")

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

    for model, width in zip(models, widths):
        empirical_ntks = []
        for i in tqdm(range(len(traj)), desc=f"Computing empirical NTKs for width={width}"):
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

if __name__ == "__main__":
    widths = [100, 500, 1000, 2000]
    print(f"Comparing empirical and theoretical NTK for widths {widths}...")

    compare_empirical_theoretical_ntk_on_circle(widths, steps=100)