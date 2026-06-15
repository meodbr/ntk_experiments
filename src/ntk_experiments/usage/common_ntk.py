from functools import partial

import torch

from ntk_experiments.CNN.theory import ntk
from ntk_experiments.CNN.theory.optimized_ntk import cumulative_compute_cnn_ntk
from ntk_experiments.CNN.practical.cnn_model import NTKCNN
from ntk_experiments.MLP.theory.theoretical_ntk import infinite_width_ntk as mlp_infinite_width_ntk
from ntk_experiments.MLP.practical.ntkmlp_model import NTKMLP

from ntk_experiments.usage.empirical_ntk import empirical_ntk

def get_cnn_theoretical_ntk_function(
    depth: int,
    k: int,
    sigma_w: float = 1.0,
    sigma_b: float = 1.0,
    implemented_phi: str = "",
):
    return partial(
        cumulative_compute_cnn_ntk,
        depth=depth,
        k=k,
        sigma_w=sigma_w,
        sigma_b=sigma_b,
        implemented_phi=implemented_phi,
    )

def mlp_theoretical_ntk_wrapper(
    x,
    xp,
    input_shape: int,
    output_dim: int,
    depth: int,
    sigma_w: float = 1.0,
    beta: float = 1.0,
    implemented_sigma: str = "",
):
    if input_shape is not None and len(input_shape) > 1:
        x = x.view(x.size(0), -1)
        xp = xp.view(xp.size(0), -1)

    Theta, _, _ = mlp_infinite_width_ntk(
        x.numpy(),
        xp.numpy(),
        depth=depth,
        sigma_w=sigma_w,
        beta=beta,
        implemented_sigma=implemented_sigma,
    )
    if isinstance(Theta, torch.Tensor):
        return Theta
    else:
        return torch.tensor(Theta)



def get_mlp_theoretical_ntk_function(
    input_shape: int,
    output_dim: int,
    depth: int,
    sigma_w: float = 1.0,
    beta: float = 1.0,
    implemented_sigma: str = "",
):
    # flattening the input shape for MLP
    if len(input_shape) > 1:
        input_dim = 1
        for dim in input_shape:
            input_dim *= dim

    return partial(
        mlp_theoretical_ntk_wrapper,
        input_shape=input_shape,
        output_dim=output_dim,
        depth=depth,
        sigma_w=sigma_w,
        beta=beta,
        implemented_sigma=implemented_sigma,
    )

def get_cnn_practical_ntk_function(
    input_shape: int,
    output_dim: int,
    width: int,
    depth: int,
    k: int,
    sigma_w: float = 1.0,
    beta: float = 1.0,
    sigma_b: float = 1.0,
    implemented_phi: str = "",
):
    assert len(input_shape) == 3, "Input shape must be a tuple of (C, H, W)"
    channel_input_dim = input_shape[0]  # Assuming input_shape is (C, H, W)
    model = NTKCNN(
        input_dim=channel_input_dim,
        output_dim=output_dim,
        width=width,
        depth=depth,
        kernel_size=k,
        beta=beta,
        sigma_w=sigma_w,
        sigma_b=sigma_b,
    )
    # print(f"Created NTKCNN model with input_dim={channel_input_dim}, output_dim={output_dim}, width={width}, depth={depth}, kernel_size={k}, beta={beta}, sigma_w={sigma_w}, sigma_b={sigma_b}")
    # print(f"Model architecture: {model}")
    return partial(empirical_ntk, model=model, output_size=output_dim)

def mlp_practical_ntk_wrapper(
    x,
    xp,
    input_shape: int,
    output_dim: int,
    model: NTKMLP,
):
    if input_shape is not None and len(input_shape) > 1:
        x = x.view(x.size(0), -1)
        xp = xp.view(xp.size(0), -1)
    
    res = empirical_ntk(x, xp, model=model, output_size=output_dim)
    print(f"Empirical NTK: {res}")
    return res

def practical_flatten_wrapper(
    x,
    xp,
    model,
    output_dim: int = 1,
):
    x_flat = x.view(x.size(0), -1)
    xp_flat = xp.view(xp.size(0), -1)
    return empirical_ntk(x_flat, xp_flat, model=model, output_size=output_dim)

def get_practical_model_ntk_function(
    model,
    output_dim: int = 1,
    flatten_input: bool = False,
):
    if flatten_input:
        return partial(practical_flatten_wrapper, model=model, output_dim=output_dim)
    return partial(empirical_ntk, model=model, output_size=output_dim)


def get_mlp_practical_ntk_function( 
    input_shape: int,
    output_dim: int,
    width: int,
    depth: int,
    beta: float = 1.0,
    sigma_w: float = 1.0,
    sigma_b: float = 1.0,
    implemented_sigma: str = "",
):
    # flattening the input shape for MLP
    if len(input_shape) > 1:
        input_dim = 1
        for dim in input_shape:
            input_dim *= dim

    model = NTKMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        width=width,
        depth=depth,
        beta=beta,
        sigma_w=sigma_w,
        sigma_b=sigma_b,
    )
    return partial(mlp_practical_ntk_wrapper, input_shape=input_shape, output_dim=output_dim, model=model)


if __name__ == "__main__":
    # Example usage
    input_shape = (1, 28, 28)  # For MNIST
    output_dim = 1
    width = 1000
    depth = 3
    sigma_w = 1.0
    sigma_b = 1.0
    beta = 1.0
    implemented_sigma = "relu"

    practical_ntk_func = get_mlp_practical_ntk_function(
        input_shape=input_shape,
        output_dim=output_dim,
        width=width,
        depth=depth,
        sigma_w=sigma_w,
        beta=beta,
        implemented_sigma=implemented_sigma,
    )

    # Generate some random data for testing
    x = torch.randn(1, *input_shape)
    xp = torch.randn(1, *input_shape)

    ntk_value = practical_ntk_func(x, xp).item()
    print(f"Empirical NTK value: {ntk_value}")

    theoretical_ntk_func = get_mlp_theoretical_ntk_function(
        input_shape=input_shape,
        output_dim=output_dim,
        depth=depth,
        sigma_w=sigma_w,
        beta=beta,
        implemented_sigma=implemented_sigma,
    )

    theo_ntk_value = theoretical_ntk_func(x, xp).item()
    print(f"Theoretical NTK value: {theo_ntk_value}")

    practical_ntk_func_cnn = get_cnn_practical_ntk_function(
        input_shape=input_shape,
        output_dim=output_dim,
        width=width,
        depth=depth,
        k=3,
        sigma_w=sigma_w,
        beta=beta,
        sigma_b=sigma_b,
    )

    cnn_ntk_value = practical_ntk_func_cnn(x, xp).item()
    print(f"Empirical CNN NTK value: {cnn_ntk_value}")

    theoretical_ntk_func_cnn = get_cnn_theoretical_ntk_function(
        depth=depth,
        k=3,
        sigma_w=sigma_w,
        sigma_b=sigma_b,
        implemented_phi=implemented_sigma,
    )
    theo_cnn_ntk_value = theoretical_ntk_func_cnn(x, xp).item()
    print(f"Theoretical CNN NTK value: {theo_cnn_ntk_value}")
