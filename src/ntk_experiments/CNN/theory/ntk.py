# ntk.py

import numpy as np

from .covariance import (
    compute_covariance_layers,
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


def dot_sigma_next(
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

    lambda_cov = np.array([
        [q_xx, q_xxbar],
        [q_xxbar, q_xbarxbar],
    ])
    match implemented_phi:
        case "relu":
            return relu_prime_gaussian_expectation(lambda_cov)
        case _:
            return bivariate_gaussian_expectation(
                phi=phi_prime,
                cov=lambda_cov,
                n_gh=n_gh,
            )


def ntk_next_layer(
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

                            dot = dot_sigma_next(
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
                                    gamma[0], gamma[1],
                                    gamma_bar[0], gamma_bar[1]
                                ]
                            )

                    Theta_next[a1, a2, b1, b2] = (
                        Sigma_next[a1, a2, b1, b2]
                        + sigma_w**2 / (k * k) * old_contribution
                    )

    return Theta_next


def compute_ntk_layers(
    x,
    xbar,
    depth,
    k,
    phi=relu,
    phi_prime=relu_prime,
    sigma_w=1.0,
    sigma_b=0.0,
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
    ) = compute_covariance_layers(
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

        Theta_xx_next = ntk_next_layer(
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

        Theta_xxbar_next = ntk_next_layer(
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

        Theta_xbarxbar_next = ntk_next_layer(
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


def final_readout_ntk(Theta_L):
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
    return np.mean(Theta_L)


if __name__ == "__main__":
    # Small example
    C0, H, W = 3, 16, 16
    x = np.random.randn(C0, H, W)
    xbar = np.random.randn(C0, H, W)

    depth = 3
    k = 3
    sigma_w = 1.0
    sigma_b = 0.1

    (
        Thetas_xx,
        Thetas_xxbar,
        Thetas_xbarxbar,
        Sigmas_xx,
        Sigmas_xxbar,
        Sigmas_xbarxbar,
    ) = compute_ntk_layers(
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
    final_ntk = final_readout_ntk(Theta_L)

    print("Final NTK:", final_ntk)