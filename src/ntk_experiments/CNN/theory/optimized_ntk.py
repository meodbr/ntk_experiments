# ntk.py
import math

import torch
import torch.nn.functional as F

import numpy as np

from ntk_experiments.CNN.theory.four_d_cumsum import patches_sum_4d_cumsum_method

from .optimized_cov import (
    naive_compute_covariance_layers,
    cumulative_compute_covariance_layers,
    bivariate_gaussian_expectation,
    relu,
)


def relu_prime(x):
    """
    Derivative of ReLU.

    The value at zero is arbitrary for the Gaussian expectation.
    We choose 0.
    """
    return (x > 0).astype(float) if isinstance(x, np.ndarray) else float(x > 0)


def relu_prime_gaussian_expectation(lambda_cov):
    """
    Computes E[relu'(u) relu'(v)] where (u, v) ~ N(0, lambda_cov).

    Since relu' is 1 for positive inputs and 0 for negative inputs,
    this is simply the probability that u > 0 and v > 0 under the Gaussian distribution.

    This can be computed using the CDF of a bivariate Gaussian.
    """
    q_xx = lambda_cov[0, 0]
    q_xbarxbar = lambda_cov[1, 1]
    q_xxbar = lambda_cov[0, 1]

    rho = q_xxbar / np.sqrt(q_xx * q_xbarxbar)

    # Probability that both u > 0 and v > 0
    return 0.25 + np.arcsin(rho) / (2 * np.pi)



def relu_gaussian_expectation_full(
    Sigma_xx,
    Sigma_xxbar,
    Sigma_xbarxbar,
    eps=1e-12,
):
    """
    Vectorized computation of E[ReLU(u) ReLU(v)]
    where (u,v) ~ N(0, [[q1, c], [c, q2]]).

    Shapes:
        Sigma_xx:       (H, W, H, W)
        Sigma_xxbar:    (H, W, H, W)
        Sigma_xbarxbar: (H, W, H, W)

    Returns:
        (H, W, H, W)
    """

    H, W, _, _ = Sigma_xxbar.shape

    device = Sigma_xxbar.device
    dtype = Sigma_xxbar.dtype

    # Diagonal variances q1[a1,a2] = Sigma_xx[a1,a2,a1,a2]
    idx_h = torch.arange(H, device=device)
    idx_w = torch.arange(W, device=device)

    q1 = Sigma_xx[idx_h[:, None], idx_w[None, :], idx_h[:, None], idx_w[None, :]]
    q2 = Sigma_xbarxbar[idx_h[:, None], idx_w[None, :], idx_h[:, None], idx_w[None, :]]

    # Reshape for broadcasting:
    # q1 indexed by (a1,a2), q2 indexed by (b1,b2)
    q1 = q1[:, :, None, None]
    q2 = q2[None, None, :, :]

    c = Sigma_xxbar

    denom = torch.sqrt(torch.clamp(q1 * q2, min=eps))
    cos_theta = torch.clamp(c / denom, -1.0 + eps, 1.0 - eps)

    theta = torch.acos(cos_theta)

    out = denom / (2.0 * math.pi) * (
        torch.sin(theta) + (math.pi - theta) * torch.cos(theta)
    )

    return out


def relu_prime_gaussian_expectation_full(
    Sigma_xx,
    Sigma_xxbar,
    Sigma_xbarxbar,
    eps=1e-12,
):
    """
    Vectorized computation of E[ReLU'(u) ReLU'(v)]
    where (u,v) ~ N(0, [[q1, c], [c, q2]]).

    Shapes:
        Sigma_xx:       (H, W, H, W)
        Sigma_xxbar:    (H, W, H, W)
        Sigma_xbarxbar: (H, W, H, W)
    Returns:
        (H, W, H, W)
    """
    
    H, W, _, _ = Sigma_xxbar.shape

    device = Sigma_xxbar.device
    dtype = Sigma_xxbar.dtype

    # Diagonal variances q1[a1,a2] = Sigma_xx[a1,a2,a1,a2]
    idx_h = torch.arange(H, device=device)
    idx_w = torch.arange(W, device=device)

    q1 = Sigma_xx[idx_h[:, None], idx_w[None, :], idx_h[:, None], idx_w[None, :]]
    q2 = Sigma_xbarxbar[idx_h[:, None], idx_w[None, :], idx_h[:, None], idx_w[None, :]]

    # Reshape for broadcasting:
    # q1 indexed by (a1,a2), q2 indexed by (b1,b2)
    q1 = q1[:, :, None, None]
    q2 = q2[None, None, :, :]

    c = Sigma_xxbar

    denom = torch.sqrt(torch.clamp(q1 * q2, min=eps))
    cos_theta = torch.clamp(c / denom, -1.0 + eps, 1.0 - eps)

    theta = torch.acos(cos_theta)

    out = (math.pi - theta) / (2.0 * math.pi)

    return out


def full_gauss_exp(
    Sigma_xx,
    Sigma_xxbar,
    Sigma_xbarxbar,
    phi=None,
    n_gh=30,
    implemented_phi="relu",
):
    assert implemented_phi == "relu", "Only ReLU is implemented for full_gauss_exp."
    return relu_gaussian_expectation_full(
        Sigma_xx,
        Sigma_xxbar,
        Sigma_xbarxbar,
    )

def full_gauss_exp_prime(
    Sigma_xx,
    Sigma_xxbar,
    Sigma_xbarxbar,
    phi_prime=None,
    n_gh=30,
    implemented_phi="relu",
):
    assert implemented_phi == "relu", "Only ReLU is implemented for full_gauss_exp_prime."
    return relu_prime_gaussian_expectation_full(
        Sigma_xx,
        Sigma_xxbar,
        Sigma_xbarxbar,
    )



def naive_dot_sigma_next(
    Sigma_xx,
    Sigma_xxbar,
    Sigma_xbarxbar,
    gamma,
    gamma_bar,
    phi_prime=relu_prime,
    n_gh=30,
    implemented_phi=None,
):
    """
    Computes

        dotSigma^{(l)}(x, xbar; gamma, gamma_bar)
        =
        E[phi'(u) phi'(v)]

    where

        (u, v) ~ N(0, lambda^{(l)})

    and lambda^{(l)} is built from the covariance matrices.
    """
    g1, g2 = gamma
    gb1, gb2 = gamma_bar

    q_xx = Sigma_xx[g1, g2, g1, g2]
    q_xbarxbar = Sigma_xbarxbar[gb1, gb2, gb1, gb2]
    q_xxbar = Sigma_xxbar[g1, g2, gb1, gb2]

    lambda_cov = np.array(
        [
            [q_xx, q_xxbar],
            [q_xxbar, q_xbarxbar],
        ]
    )
    match implemented_phi:
        case "relu":
            return relu_prime_gaussian_expectation(lambda_cov)
        case _:
            print("Warning: using numerical Gaussian expectation for phi_prime.")
            return bivariate_gaussian_expectation(
                phi=phi_prime,
                cov=lambda_cov,
                n_gh=n_gh,
            )


def full_old_contribution(
    Sigma_xx,
    Sigma_xxbar,
    Sigma_xbarxbar,
    Theta_prev,
    k,
    phi_prime=relu_prime,
    n_gh=30,
    implemented_phi=None,
):
    Gauss_exp_prime = full_gauss_exp_prime(
        Sigma_xx,
        Sigma_xxbar,
        Sigma_xbarxbar,
        phi_prime=phi_prime,
        n_gh=n_gh,
        implemented_phi=implemented_phi,
    )

    # elementwise multiplication of Gauss_exp_prime and Theta_prev
    old_contribution = Gauss_exp_prime * Theta_prev

    return old_contribution


def cumulative_ntk_next_layer(
    Sigma_next,
    Sigma_xx,
    Sigma_xxbar,
    Sigma_xbarxbar,
    Theta_prev,
    k,
    phi_prime=relu_prime,
    n_gh=30,
    implemented_phi=None,
):
    """
    Computes the NTK recursion

        Theta^{(l+1)}(x, xbar; alpha, alpha_bar)
        =
        Sigma^{(l+1)}(x, xbar; alpha, alpha_bar)
        +
        sigma_w^2 / |P|
        sum_{beta in P}
        dotSigma^{(l)}(x, xbar; alpha+beta, alpha_bar+beta)
        Theta^{(l)}(x, xbar; alpha+beta, alpha_bar+beta)

    Inputs:
        Sigma_next:
            Sigma^{(l+1)}(x, xbar)
            shape (H_next, W_next, H_next, W_next)

        Sigma_xx:
            Sigma^{(l)}(x, x)
            shape (H_l, W_l, H_l, W_l)

        Sigma_xxbar:
            Sigma^{(l)}(x, xbar)
            shape (H_l, W_l, H_l, W_l)

        Sigma_xbarxbar:
            Sigma^{(l)}(xbar, xbar)
            shape (H_l, W_l, H_l, W_l)

        Theta_prev:
            Theta^{(l)}(x, xbar)
            shape (H_l, W_l, H_l, W_l)

    Returns:
        Theta_next:
            Theta^{(l+1)}(x, xbar)
            shape (H_next, W_next, H_next, W_next)
    """
    H_next, W_next, _, _ = Sigma_next.shape

    Gauss_exp_prime = full_gauss_exp_prime(
        Sigma_xx,
        Sigma_xxbar,
        Sigma_xbarxbar,
        phi_prime=phi_prime,
        n_gh=n_gh,
        implemented_phi=implemented_phi,
    )

    old_contribution = Gauss_exp_prime * Theta_prev

    # Compute the patches sums of old_contribution using the cumsum method
    old_contribution_patches_sum = patches_sum_4d_cumsum_method(
        old_contribution, k1=k, k2=k
    )

    Theta_next = Sigma_next + (1.0 / (k * k)) * old_contribution_patches_sum
    return Theta_next


def naive_ntk_next_layer(
    Sigma_next,
    Sigma_xx,
    Sigma_xxbar,
    Sigma_xbarxbar,
    Theta_prev,
    k,
    phi_prime=relu_prime,
    sigma_w=1.0,
    n_gh=30,
    implemented_phi=None,
):
    """
    Computes the NTK recursion

        Theta^{(l+1)}(x, xbar; alpha, alpha_bar)
        =
        Sigma^{(l+1)}(x, xbar; alpha, alpha_bar)
        +
        sigma_w^2 / |P|
        sum_{beta in P}
        dotSigma^{(l)}(x, xbar; alpha+beta, alpha_bar+beta)
        Theta^{(l)}(x, xbar; alpha+beta, alpha_bar+beta)

    Inputs:
        Sigma_next:
            Sigma^{(l+1)}(x, xbar)
            shape (H_next, W_next, H_next, W_next)

        Sigma_xx:
            Sigma^{(l)}(x, x)
            shape (H_l, W_l, H_l, W_l)

        Sigma_xxbar:
            Sigma^{(l)}(x, xbar)
            shape (H_l, W_l, H_l, W_l)

        Sigma_xbarxbar:
            Sigma^{(l)}(xbar, xbar)
            shape (H_l, W_l, H_l, W_l)

        Theta_prev:
            Theta^{(l)}(x, xbar)
            shape (H_l, W_l, H_l, W_l)

    Returns:
        Theta_next:
            Theta^{(l+1)}(x, xbar)
            shape (H_next, W_next, H_next, W_next)
    """
    H_next, W_next, _, _ = Sigma_next.shape

    Theta_next = np.zeros_like(Sigma_next)

    for a1 in range(H_next):
        for a2 in range(W_next):
            for b1 in range(H_next):
                for b2 in range(W_next):

                    old_contribution = 0.0

                    for p1 in range(k):
                        for p2 in range(k):
                            gamma = (a1 + p1, a2 + p2)
                            gamma_bar = (b1 + p1, b2 + p2)

                            dot = naive_dot_sigma_next(
                                Sigma_xx=Sigma_xx,
                                Sigma_xxbar=Sigma_xxbar,
                                Sigma_xbarxbar=Sigma_xbarxbar,
                                gamma=gamma,
                                gamma_bar=gamma_bar,
                                phi_prime=phi_prime,
                                n_gh=n_gh,
                                implemented_phi=implemented_phi,
                            )

                            old_contribution += (
                                dot
                                * Theta_prev[
                                    gamma[0], gamma[1], gamma_bar[0], gamma_bar[1]
                                ]
                            )

                    Theta_next[a1, a2, b1, b2] = (
                        Sigma_next[a1, a2, b1, b2]
                        + sigma_w**2 / (k * k) * old_contribution
                    )

    return Theta_next


def cumulative_compute_ntk_layers(
    x,
    xbar,
    depth,
    k,
    phi=relu,
    phi_prime=relu_prime,
    sigma_w=1.0,
    sigma_b=1.0,
    n_gh=30,
    implemented_phi=None,
):
    """
    Computes the CNN NTK Theta^{(l)} up to layer `depth`.

    This function first computes the covariance recursion using covariance.py,
    then computes the NTK recursion.

    Returns:
        Thetas_xx:
            list of Theta^{(l)}(x, x)

        Thetas_xxbar:
            list of Theta^{(l)}(x, xbar)

        Thetas_xbarxbar:
            list of Theta^{(l)}(xbar, xbar)

        Sigmas_xx, Sigmas_xxbar, Sigmas_xbarxbar:
            covariance lists returned from compute_covariance_layers
    """
    (
        Sigmas_xx,
        Sigmas_xxbar,
        Sigmas_xbarxbar,
    ) = cumulative_compute_covariance_layers(
        x=x,
        xbar=xbar,
        depth=depth,
        k=k,
        phi=phi,
        sigma_w=sigma_w,
        sigma_b=sigma_b,
        n_gh=n_gh,
        implemented_phi=implemented_phi,
    )

    Thetas_xx = []
    Thetas_xxbar = []
    Thetas_xbarxbar = []

    # Initialization:
    # Theta^{(1)} = Sigma^{(1)}
    Theta_xx = Sigmas_xx[0].detach().clone()
    Theta_xxbar = Sigmas_xxbar[0].detach().clone()
    Theta_xbarxbar = Sigmas_xbarxbar[0].detach().clone()

    Thetas_xx.append(Theta_xx)
    Thetas_xxbar.append(Theta_xxbar)
    Thetas_xbarxbar.append(Theta_xbarxbar)

    # Recursion: layers 2, ..., depth
    for layer_idx in range(1, depth):
        Sigma_xx_prev = Sigmas_xx[layer_idx - 1]
        Sigma_xxbar_prev = Sigmas_xxbar[layer_idx - 1]
        Sigma_xbarxbar_prev = Sigmas_xbarxbar[layer_idx - 1]

        Sigma_xx_next = Sigmas_xx[layer_idx]
        Sigma_xxbar_next = Sigmas_xxbar[layer_idx]
        Sigma_xbarxbar_next = Sigmas_xbarxbar[layer_idx]

        Theta_xx_next = cumulative_ntk_next_layer(
            Sigma_next=Sigma_xx_next,
            Sigma_xx=Sigma_xx_prev,
            Sigma_xxbar=Sigma_xx_prev,
            Sigma_xbarxbar=Sigma_xx_prev,
            Theta_prev=Theta_xx,
            k=k,
            phi_prime=phi_prime,
            n_gh=n_gh,
            implemented_phi=implemented_phi,
        )

        Theta_xxbar_next = cumulative_ntk_next_layer(
            Sigma_next=Sigma_xxbar_next,
            Sigma_xx=Sigma_xx_prev,
            Sigma_xxbar=Sigma_xxbar_prev,
            Sigma_xbarxbar=Sigma_xbarxbar_prev,
            Theta_prev=Theta_xxbar,
            k=k,
            phi_prime=phi_prime,
            n_gh=n_gh,
            implemented_phi=implemented_phi,
        )

        Theta_xbarxbar_next = cumulative_ntk_next_layer(
            Sigma_next=Sigma_xbarxbar_next,
            Sigma_xx=Sigma_xbarxbar_prev,
            Sigma_xxbar=Sigma_xbarxbar_prev,
            Sigma_xbarxbar=Sigma_xbarxbar_prev,
            Theta_prev=Theta_xbarxbar,
            k=k,
            phi_prime=phi_prime,
            n_gh=n_gh,
            implemented_phi=implemented_phi,
        )

        Theta_xx = Theta_xx_next
        Theta_xxbar = Theta_xxbar_next
        Theta_xbarxbar = Theta_xbarxbar_next

        Thetas_xx.append(Theta_xx)
        Thetas_xxbar.append(Theta_xxbar)
        Thetas_xbarxbar.append(Theta_xbarxbar)

    return (
        Thetas_xx,
        Thetas_xxbar,
        Thetas_xbarxbar,
        Sigmas_xx,
        Sigmas_xxbar,
        Sigmas_xbarxbar,
    )


def naive_compute_ntk_layers(
    x,
    xbar,
    depth,
    k,
    phi=relu,
    phi_prime=relu_prime,
    sigma_w=1.0,
    sigma_b=1.0,
    n_gh=30,
    implemented_phi=None,
):
    """
    Computes the CNN NTK Theta^{(l)} up to layer `depth`.

    This function first computes the covariance recursion using covariance.py,
    then computes the NTK recursion.

    Returns:
        Thetas_xx:
            list of Theta^{(l)}(x, x)

        Thetas_xxbar:
            list of Theta^{(l)}(x, xbar)

        Thetas_xbarxbar:
            list of Theta^{(l)}(xbar, xbar)

        Sigmas_xx, Sigmas_xxbar, Sigmas_xbarxbar:
            covariance lists returned from compute_covariance_layers
    """
    (
        Sigmas_xx,
        Sigmas_xxbar,
        Sigmas_xbarxbar,
    ) = naive_compute_covariance_layers(
        x=x,
        xbar=xbar,
        depth=depth,
        k=k,
        phi=phi,
        sigma_w=sigma_w,
        sigma_b=sigma_b,
        n_gh=n_gh,
        implemented_phi=implemented_phi,
    )

    Thetas_xx = []
    Thetas_xxbar = []
    Thetas_xbarxbar = []

    # Initialization:
    # Theta^{(1)} = Sigma^{(1)}
    Theta_xx = Sigmas_xx[0].copy()
    Theta_xxbar = Sigmas_xxbar[0].copy()
    Theta_xbarxbar = Sigmas_xbarxbar[0].copy()

    Thetas_xx.append(Theta_xx)
    Thetas_xxbar.append(Theta_xxbar)
    Thetas_xbarxbar.append(Theta_xbarxbar)

    # Recursion: layers 2, ..., depth
    for layer_idx in range(1, depth):
        Sigma_xx_prev = Sigmas_xx[layer_idx - 1]
        Sigma_xxbar_prev = Sigmas_xxbar[layer_idx - 1]
        Sigma_xbarxbar_prev = Sigmas_xbarxbar[layer_idx - 1]

        Sigma_xx_next = Sigmas_xx[layer_idx]
        Sigma_xxbar_next = Sigmas_xxbar[layer_idx]
        Sigma_xbarxbar_next = Sigmas_xbarxbar[layer_idx]

        Theta_xx_next = naive_ntk_next_layer(
            Sigma_next=Sigma_xx_next,
            Sigma_xx=Sigma_xx_prev,
            Sigma_xxbar=Sigma_xx_prev,
            Sigma_xbarxbar=Sigma_xx_prev,
            Theta_prev=Theta_xx,
            k=k,
            phi_prime=phi_prime,
            sigma_w=sigma_w,
            n_gh=n_gh,
            implemented_phi=implemented_phi,
        )

        Theta_xxbar_next = naive_ntk_next_layer(
            Sigma_next=Sigma_xxbar_next,
            Sigma_xx=Sigma_xx_prev,
            Sigma_xxbar=Sigma_xxbar_prev,
            Sigma_xbarxbar=Sigma_xbarxbar_prev,
            Theta_prev=Theta_xxbar,
            k=k,
            phi_prime=phi_prime,
            sigma_w=sigma_w,
            n_gh=n_gh,
            implemented_phi=implemented_phi,
        )

        Theta_xbarxbar_next = naive_ntk_next_layer(
            Sigma_next=Sigma_xbarxbar_next,
            Sigma_xx=Sigma_xbarxbar_prev,
            Sigma_xxbar=Sigma_xbarxbar_prev,
            Sigma_xbarxbar=Sigma_xbarxbar_prev,
            Theta_prev=Theta_xbarxbar,
            k=k,
            phi_prime=phi_prime,
            sigma_w=sigma_w,
            n_gh=n_gh,
            implemented_phi=implemented_phi,
        )

        Theta_xx = Theta_xx_next
        Theta_xxbar = Theta_xxbar_next
        Theta_xbarxbar = Theta_xbarxbar_next

        Thetas_xx.append(Theta_xx)
        Thetas_xxbar.append(Theta_xxbar)
        Thetas_xbarxbar.append(Theta_xbarxbar)

    return (
        Thetas_xx,
        Thetas_xxbar,
        Thetas_xbarxbar,
        Sigmas_xx,
        Sigmas_xxbar,
        Sigmas_xbarxbar,
    )


def cumulative_final_readout_ntk(Theta_L):
    """
    Computes the NTK after global average pooling:

        Theta(x, xbar)
        =
        1 / |Omega_L|^2
        sum_{alpha, alpha_bar}
        Theta^{(L)}(x, xbar; alpha, alpha_bar)

    Since Theta_L has shape (H_L, W_L, H_L, W_L),
    this is simply the mean of all entries.
    """
    print("Theta_L shape:", Theta_L.shape)
    return Theta_L.mean()


def naive_final_readout_ntk(Theta_L):
    """
    Computes the NTK after global average pooling:

        Theta(x, xbar)
        =
        1 / |Omega_L|^2
        sum_{alpha, alpha_bar}
        Theta^{(L)}(x, xbar; alpha, alpha_bar)

    Since Theta_L has shape (H_L, W_L, H_L, W_L),
    this is simply the mean of all entries.
    """
    print("Theta_L shape:", Theta_L.shape)
    # Use np whenever naive
    return np.mean(Theta_L)


def cumulative_compute_cnn_ntk(
    x,
    xbar,
    depth,
    k,
    phi=relu,
    phi_prime=relu_prime,
    sigma_w=1.0,
    sigma_b=1.0,
    n_gh=30,
    implemented_phi=None,
):
    """
    Convenience function to compute the final CNN NTK value between two inputs.

    This is just a wrapper around compute_ntk_layers and final_readout_ntk.
    """
    (
        Thetas_xx,
        Thetas_xxbar,
        Thetas_xbarxbar,
        Sigmas_xx,
        Sigmas_xxbar,
        Sigmas_xbarxbar,
    ) = cumulative_compute_ntk_layers(
        x=x,
        xbar=xbar,
        depth=depth,
        k=k,
        phi=phi,
        phi_prime=phi_prime,
        sigma_w=sigma_w,
        sigma_b=sigma_b,
        n_gh=n_gh,
        implemented_phi=implemented_phi,
    )

    Theta_L = Thetas_xxbar[-1]
    final_ntk = cumulative_final_readout_ntk(Theta_L)

    return final_ntk


def naive_compute_cnn_ntk(
    x,
    xbar,
    depth,
    k,
    phi=None,
    phi_prime=None,
    sigma_w=1.0,
    sigma_b=1.0,
    n_gh=30,
    implemented_phi=None,
):
    """
    Convenience function to compute the final CNN NTK value between two inputs.

    This is just a wrapper around compute_ntk_layers and final_readout_ntk.
    """
    (
        Thetas_xx,
        Thetas_xxbar,
        Thetas_xbarxbar,
        Sigmas_xx,
        Sigmas_xxbar,
        Sigmas_xbarxbar,
    ) = naive_compute_ntk_layers(
        x=x,
        xbar=xbar,
        depth=depth,
        k=k,
        phi=phi,
        phi_prime=phi_prime,
        sigma_w=sigma_w,
        sigma_b=sigma_b,
        n_gh=n_gh,
        implemented_phi=implemented_phi,
    )

    Theta_L = Thetas_xxbar[-1]
    final_ntk = naive_final_readout_ntk(Theta_L)

    return final_ntk


if __name__ == "__main__":
    # Small example
    C0 = 3
    H = 28
    W = H
    depth = 3
    k = 3
    sigma_w = 1.0
    sigma_b = 1.0

    x = np.random.randn(C0, H, W)
    xbar = np.random.randn(C0, H, W)

    x = torch.tensor(x, dtype=torch.float32)
    xbar = torch.tensor(xbar, dtype=torch.float32)



    (
        Thetas_xx,
        Thetas_xxbar,
        Thetas_xbarxbar,
        Sigmas_xx,
        Sigmas_xxbar,
        Sigmas_xbarxbar,
    ) = cumulative_compute_ntk_layers(
        x=x,
        xbar=xbar,
        depth=depth,
        k=k,
        phi=relu,
        phi_prime=relu_prime,
        sigma_w=sigma_w,
        sigma_b=sigma_b,
        n_gh=30,
        implemented_phi="relu",
    )
    Theta_L = Thetas_xxbar[-1]
    final_ntk = cumulative_final_readout_ntk(Theta_L)

    print("Final NTK:", final_ntk)


    (
        n_Thetas_xx,
        n_Thetas_xxbar,
        n_Thetas_xbarxbar,
        n_Sigmas_xx,
        n_Sigmas_xxbar,
        n_Sigmas_xbarxbar,
    ) = naive_compute_ntk_layers(
        x=x.numpy(),
        xbar=xbar.numpy(),
        depth=depth,
        k=k,
        phi=relu,
        phi_prime=relu_prime,
        sigma_w=sigma_w,
        sigma_b=sigma_b,
        n_gh=30,
        implemented_phi="relu",
    )

    n_Theta_L = n_Thetas_xxbar[-1]
    n_final_ntk = naive_final_readout_ntk(n_Theta_L)

    print("Final NTK (naive):", n_final_ntk)

    print("Difference between cumulative and naive NTK:", abs(final_ntk - n_final_ntk))
    print("difference on last layer:", torch.linalg.norm(Theta_L - torch.tensor(n_Theta_L)))
