import torch
import torch.nn as nn


from ntk_experiments.usage.empirical_ntk import empirical_ntk
from ntk_experiments.CNN.practical.cnn_model import NTKCNN
from ntk_experiments.CNN.theory.ntk import compute_cnn_ntk


def example_cnn_ntk():
    # Create two random inputs
    x = torch.randn(1, 3, 32, 32)  # Single image input
    x_prime = torch.randn(1, 3, 32, 32)

    # Create a CNN model
    model = NTKCNN(input_dim=3, output_dim=1, width=64, depth=5, beta=1.0)

    # Compute the empirical NTK between the two inputs
    ntk_value = empirical_ntk(model, x, x_prime)
    print("Empirical NTK value:", ntk_value)


def compare_cnn_ntk():
    H = 8
    W = H
    Cin = 3
    Cout = 1
    k = 3
    L = 3
    width = 1000
    sigma_w = 1.0
    sigma_b = 1.0
    beta = 1.0

    x = torch.randn(1, Cin, H, W)  # Single image input
    x_prime = torch.randn(1, Cin, H, W)
    model = NTKCNN(
        input_dim=Cin,
        output_dim=Cout,
        width=width,
        depth=L,
        kernel_size=k,
        beta=beta,
        sigma_w=sigma_w,
        sigma_b=sigma_b,
    )
    empirical_value = empirical_ntk(model, x, x_prime)
    print("Empirical NTK value:", empirical_value)

    theoretical_value = compute_cnn_ntk(
        x=x.squeeze(0).numpy(),  # Remove batch dimension for theory function
        xbar=x_prime.squeeze(0).numpy(),
        depth=L,
        k=k,
        sigma_w=sigma_w,
        sigma_b=sigma_b,
        implemented_phi="relu",
    )
    print("Theoretical NTK value:", theoretical_value)


if __name__ == "__main__":
    compare_cnn_ntk()
