from functools import partial

import torch

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
    depth: int,
    sigma_w: float = 1.0,
    beta: float = 1.0,
    implemented_sigma: str = "",
):
    return torch.tensor(
        mlp_infinite_width_ntk(
            x.numpy(),
            xp.numpy(),
            depth=depth,
            sigma_w=sigma_w,
            beta=beta,
            implemented_sigma=implemented_sigma,
        ),
        dtype=torch.float32,
    )

def get_mlp_theoretical_ntk_function(
    depth: int,
    sigma_w: float = 1.0,
    beta: float = 1.0,
    implemented_sigma: str = "",
):
    return partial(
        mlp_theoretical_ntk_wrapper,
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
        implemented_phi=implemented_phi,
    )
    return partial(empirical_ntk, model=model)

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
    
    return empirical_ntk(model, x, xp)


def get_mlp_practical_ntk_function( 
    input_shape: int,
    output_dim: int,
    width: int,
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

    model = NTKMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        width=width,
        depth=depth,
        beta=beta,
        sigma_w=sigma_w,
        implemented_sigma=implemented_sigma,
    )
    return partial(mlp_practical_ntk_wrapper, input_shape=input_shape, output_dim=output_dim, model=model)