"""
Goal of the module : show the convergence of the empirical NTK to the theoretical NTK as the width of the network increases.
"""

import torch
import numpy as np

from ntk_experiments.config import config
from ntk_experiments.usage.common_ntk import get_mlp_theoretical_ntk_function, get_mlp_practical_ntk_function

from ntk_experiments.usage.dataset import get_raw_mnist_data, plot_mnist_sample

def linear_interpolation(x_list, total_points=100):
    """
    Given a list of points x_list, return a list of points that are linearly interpolated between the points in x_list.
    The number of points returned is total_points.
    """
    num_points = total_points // (len(x_list) - 1)
    x_list = [x.detach().cpu().numpy() for x in x_list]
    x_list = np.array(x_list)
    x_interp = []
    for i in range(len(x_list) - 1):
        x_start = x_list[i]
        x_end = x_list[i + 1]
        for j in range(num_points):
            alpha = j / (num_points - 1)
            x_interp.append((1 - alpha) * x_start + alpha * x_end)
    return torch.tensor(x_interp, dtype=torch.float32)

if __name__ == "__main__":
    # Plot 3 mnist samples
    X_train, X_test, y_train, y_test = get_raw_mnist_data(seed=config.SEED)
    for i in range(3):
        plot_mnist_sample(X_train, y_train, index=i)
    
    # Plot one 6, one 7, and one 8 sample
    indices = [i for i, label in enumerate(y_train) if label in [6, 7, 8]]
    selected_indices = indices[:3]  # Select the first three indices
    for i in selected_indices:
        plot_mnist_sample(X_train, y_train, index=i)