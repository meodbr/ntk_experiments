import math

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.func import functional_call, jacrev

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

from .random_walk import random_walk_unit_sphere, unit_sphere
from .theoretical_ntk import infinite_width_ntk, relu, relu_prime
from .mlp_ntk import NTKMLP

sns.set_theme()

INPUT_DIM = 2
OUTPUT_DIM = 1
BETA = 0.1

# Data (MNIST-like digits)
digits = load_digits()
X = digits.data
y = digits.target

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Simple MLP
class MLP_classic(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, width),
            nn.ReLU(),
            nn.Linear(width, OUTPUT_DIM)
        )

    def forward(self, x):
        return self.net(x)# / math.sqrt(self.width)  # NTK scaling


class MLP_NTK(nn.Module):
    def __init__(self, width):
        super().__init__()

        self.fc1 = nn.Linear(INPUT_DIM, width, bias=True)
        self.fc2 = nn.Linear(width, OUTPUT_DIM, bias=True)

        self.width = width

        self.reset_parameters()

    def reset_parameters(self):
        # N(0,1) weights (NOT Kaiming)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0)

        nn.init.normal_(self.fc1.bias, mean=0.0, std=BETA)
        nn.init.normal_(self.fc2.bias, mean=0.0, std=BETA)

    def forward(self, x):
        # explicit NTK scaling
        x = self.fc1(x) / math.sqrt(self.width)
        x = torch.relu(x)
        x = self.fc2(x) / math.sqrt(self.width)

        return x



def reshape_to_2D_jacobian(jacobian_dict, output_size):
    # Reshape the Jacobian from a dict of parameter tensors to a single 2D matrix
    # Each row corresponds to the gradient of one output dimension w.r.t. all parameters
    jacobian_rows = []
    for key, jac in jacobian_dict.items():
        # jac shape: (output_size, param_shape...)
        jac_reshaped = jac.reshape(output_size, -1)  # Shape: (output_size, num_params)
        jacobian_rows.append(jac_reshaped)

    full_jacobian = torch.cat(jacobian_rows, dim=1)  # Shape: (output_size, total_num_params)
    return full_jacobian
    
def empirical_ntk(model, x, x_prime):
    # The neural tangent kernel between two inputs x and x':
    # K(x, x') = J(x) @ J(x').T where J is the Jacobian of the network output w.r.t. parameters
    # Compute Jacobian for x

    def model_output_x(params):
        return functional_call(model, params, x)
    
    def model_output_x_prime(params):
        return functional_call(model, params, x_prime)

    output_size = model(x).shape[-1] 
    
    params = dict(model.named_parameters())
    print("Number of parameters:", sum(p.numel() for p in params.values()))
    print("Parameter shapes:")
    for name, p in params.items():
        print(f"{name}: {p.shape}")

    J_x = jacrev(model_output_x)(params)
    J_x = reshape_to_2D_jacobian(J_x, output_size)
    print("J_x shape:", J_x.shape)
    J_x_prime = jacrev(model_output_x_prime)(params)
    J_x_prime = reshape_to_2D_jacobian(J_x_prime, output_size)
    print("J_x_prime shape:", J_x_prime.shape)

    ntk = J_x @ J_x_prime.T
    print("Empirical NTK shape:", ntk.shape)
    if ntk.numel() == 1:
        print("Empirical NTK value:", ntk.item())

    return ntk

# Training function
def train_model(width, epochs=50, lr=1e-3):
    model = MLP_NTK(width)
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


def compare_empirical_theoretical_ntk_on_sample(width):
    model = NTKMLP(input_dim=INPUT_DIM, width=width, depth=4, beta=BETA)
    # model = MLP_classic(width)
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
        beta=BETA,
        n_gh=40,
    )[0]

    print("Empirical NTK:", empirical)
    print("Theoretical NTK:", theoretical)

def compare_empirical_theoretical_ntk_on_random_walk(width, steps=100):
    model = NTKMLP(input_dim=INPUT_DIM, width=width, depth=4, beta=BETA)
    # traj = random_walk_unit_sphere(dim=INPUT_DIM, steps=100, step_size=0.05, seed=42)
    gamma, traj = unit_sphere(dim=INPUT_DIM, steps=steps, seed=42)
    traj = torch.tensor(traj, dtype=torch.float32)# .unsqueeze(1)  # Shape: (steps, 1, dim)
    print("Trajectory shape:", traj.shape)

    empirical_ntks = []
    theoretical_ntks = []

    x = torch.zeros((INPUT_DIM, ), dtype=torch.float32)
    x[0] = 1.0  # Compare every point on the trajectory to this fixed reference point
    print(x)

    for i in range(len(traj)):
        x_prime = traj[i]
        print(f"x_prime shape at step {i}:", x_prime.shape)
        print(f"x shape: {x.shape}")

        empirical = empirical_ntk(model, torch.tensor(x).unsqueeze(0), torch.tensor(x_prime).unsqueeze(0)).item()
        theoretical = infinite_width_ntk(
            x=x,
            xp=x_prime,
            depth=4,
            # sigma=relu,
            # sigma_prime=relu_prime,
            implemented_sigma="relu",
            sigma_w=1.0,
            beta=BETA,
            n_gh=40,
        )[0]

        empirical_ntks.append(empirical)
        theoretical_ntks.append(theoretical)

    plt.figure(figsize=(10, 5))
    plt.plot(gamma, theoretical_ntks, label='Theoretical NTK')
    plt.plot(gamma, empirical_ntks, label='Empirical NTK', linestyle='dashed')
    plt.title('Empirical vs Theoretical NTK along unit circle')
    plt.xlabel('Gamma (angle along circle)')
    plt.ylabel('NTK Value')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    width = 1000
    # acc = train_model(width)
    # print(f"Test Accuracy for width {width}: {acc:.4f}")
    print(f"Comparing empirical and theoretical NTK for width {width}...")

    # compare_empirical_theoretical_ntk_on_sample(width)
    compare_empirical_theoretical_ntk_on_random_walk(width, steps=100)