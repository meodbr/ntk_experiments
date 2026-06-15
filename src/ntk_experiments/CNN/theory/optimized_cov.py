"""
Reimplementation of the covariance computation for CNN-GP, using a more optimized approach.
using torch.Tensors and cumulative sums to compute the patch sums more efficiently.
"""
import math

import torch
import numpy as np
from numpy.polynomial.hermite import hermgauss

from ntk_experiments.CNN.theory.four_d_cumsum import patches_sum_4d_cumsum_method


def relu(x):
    return np.maximum(x, 0.0)


def bivariate_gaussian_expectation(phi, cov, n_gh=30, jitter=1e-10):
    """
    Computes E[phi(u) phi(v)] where (u,v) ~ N(0, cov)
    using 2D Gauss-Hermite quadrature.

    cov: shape (2, 2)
    """
    print("Warning: using numerical Gaussian expectation for phi.")
    cov = np.asarray(cov, dtype=float)

    # Numerical stabilization
    cov = cov + jitter * np.eye(2)

    # Cholesky: z ~ N(0, I), then [u,v] = L z
    L = np.linalg.cholesky(cov)

    nodes, weights = hermgauss(n_gh)

    expectation = 0.0

    for i, xi in enumerate(nodes):
        for j, xj in enumerate(nodes):
            z = np.sqrt(2.0) * np.array([xi, xj])
            u, v = L @ z
            expectation += weights[i] * weights[j] * phi(u) * phi(v)

    # Hermite rule has normalization 1 / pi in dimension 2
    return expectation / np.pi


def relu_gaussian_expectation(cov):
    """
    Computes E[relu(u) relu(v)] where (u,v) ~ N(0, cov)
    using the closed-form formula for ReLU.

    cov: shape (2, 2)
    """
    q_xx = cov[0, 0]
    q_xbarxbar = cov[1, 1]
    q_xxbar = cov[0, 1]

    if q_xx <= 0 or q_xbarxbar <= 0:
        return 0.0

    rho = q_xxbar / np.sqrt(q_xx * q_xbarxbar)
    sigma_x = np.sqrt(q_xx)
    sigma_xbar = np.sqrt(q_xbarxbar)

    res = (sigma_x * sigma_xbar / (2 * np.pi)) * (
        np.sqrt(1 - rho * rho) + rho * (np.arcsin(rho) + np.pi / 2)
    )
    return float(res)


def cumulative_covariance_initialization(x, xbar, k, sigma_w=1.0, sigma_b=1.0):
    """
    Computes Sigma^{(1)} for two images x and xbar.

    x, xbar: arrays of shape (C0, H, W)
    k: kernel size, with P = {0,...,k-1}^2

    Returns:
        Sigma: shape (H1, W1, H1, W1)
        where H1 = H - k + 1, W1 = W - k + 1
    """
    if x.ndim == 4:
        x = x.squeeze(0)
    if xbar.ndim == 4:
        xbar = xbar.squeeze(0)
    # print(f"x shape: {x.shape}, xbar shape: {xbar.shape}")
    C0, H, W = x.shape
    C0_bar, H_bar, W_bar = xbar.shape

    assert xbar.shape == x.shape
    assert C0 == C0_bar

    H1 = H - k + 1
    W1 = W - k + 1

    Sigma = sigma_b**2 + (sigma_w**2 / (C0 * k * k)) * patches_sum_4d_cumsum_method(
        torch.einsum("cij,ckl->ijkl", x, xbar),
        k1=k,
        k2=k,
    )

    assert Sigma.shape == (H1, W1, H1, W1)

    return Sigma


def naive_covariance_initialization(x, xbar, k, sigma_w=1.0, sigma_b=1.0):
    """
    Computes Sigma^{(1)} for two images x and xbar.

    x, xbar: arrays of shape (C0, H, W)
    k: kernel size, with P = {0,...,k-1}^2

    Returns:
        Sigma: shape (H1, W1, H1, W1)
        where H1 = H - k + 1, W1 = W - k + 1
    """
    C0, H, W = x.shape
    C0_bar, H_bar, W_bar = xbar.shape

    assert xbar.shape == x.shape
    assert C0 == C0_bar

    H1 = H - k + 1
    W1 = W - k + 1

    Sigma = np.zeros((H1, W1, H1, W1))

    for a1 in range(H1):
        for a2 in range(W1):
            for b1 in range(H1):
                for b2 in range(W1):
                    s = 0.0

                    for c in range(C0):
                        for p1 in range(k):
                            for p2 in range(k):
                                s += x[c, a1 + p1, a2 + p2] * xbar[c, b1 + p1, b2 + p2]

                    Sigma[a1, a2, b1, b2] = sigma_b**2 + sigma_w**2 / (C0 * k * k) * s

    return Sigma



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


def full_gauss_exp(
    Sigma_xx,
    Sigma_xxbar,
    Sigma_xbarxbar,
    phi=None,
    n_gh=30,
    implemented_phi=None,
):
    assert implemented_phi == "relu", "Only ReLU is implemented for full_gauss_exp."
    return relu_gaussian_expectation_full(
        Sigma_xx,
        Sigma_xxbar,
        Sigma_xbarxbar,
    )

    # return full_gauss_exp_gh(
    #     Sigma_xx,
    #     Sigma_xxbar,
    #     Sigma_xbarxbar,
    #     phi=phi,
    #     n_gh=n_gh,
    # )


def old_full_gauss_exp(
    Sigma_xx,
    Sigma_xxbar,
    Sigma_xbarxbar,
    phi=relu,
    n_gh=30,
    implemented_phi=None,
):
    """
    Computes E[phi(u) phi(v)] where (u,v) ~ N(0, cov)
    for all pairs of positions in Sigma_xx, Sigma_xxbar, Sigma_xbarxbar.

    Returns:
        Gauss_exp: shape (H_l, W_l, H_l, W_l)
    """
    H_l, W_l, _, _ = Sigma_xxbar.shape

    Gauss_exp = torch.zeros((H_l, W_l, H_l, W_l))

    for a1 in range(H_l):
        for a2 in range(W_l):
            for b1 in range(H_l):
                for b2 in range(W_l):

                    lambda_cov = np.array(
                        [
                            [
                                Sigma_xx[a1, a2, a1, a2],
                                Sigma_xxbar[a1, a2, b1, b2]],
                            [
                                Sigma_xxbar[a1, a2, b1, b2],
                                Sigma_xbarxbar[b1, b2, b1, b2],
                            ],
                        ]
                    )

                    match implemented_phi:
                        case "relu":
                            Gauss_exp[a1, a2, b1, b2] = relu_gaussian_expectation(
                                lambda_cov
                            )
                        case _:
                            Gauss_exp[a1, a2, b1, b2] = bivariate_gaussian_expectation(
                                phi=phi,
                                cov=lambda_cov,
                                n_gh=n_gh,
                            )

    return Gauss_exp


def cumulative_covariance_next_layer(
    Sigma_xx,
    Sigma_xxbar,
    Sigma_xbarxbar,
    k,
    phi=relu,
    sigma_w=1.0,
    sigma_b=1.0,
    n_gh=30,
    implemented_phi=None,
):
    """
    Computes Sigma^{(l+1)}(x, xbar) from Sigma^{(l)}.

    Inputs:
        Sigma_xx:
            Sigma^{(l)}(x, x; alpha, alpha_bar)
            shape (H_l, W_l, H_l, W_l)

        Sigma_xxbar:
            Sigma^{(l)}(x, xbar; alpha, alpha_bar)
            shape (H_l, W_l, H_l, W_l)

        Sigma_xbarxbar:
            Sigma^{(l)}(xbar, xbar; alpha, alpha_bar)
            shape (H_l, W_l, H_l, W_l)
    Returns:
        Sigma_next:
            Sigma^{(l+1)}(x, xbar)
            shape (H_{l+1}, W_{l+1}, H_{l+1}, W_{l+1})
    """
    Gauss_exp = full_gauss_exp(
        Sigma_xx,
        Sigma_xxbar,
        Sigma_xbarxbar,
        phi=phi,
        n_gh=n_gh,
        implemented_phi=implemented_phi,
    )
    Patches_sum = patches_sum_4d_cumsum_method(Gauss_exp, k1=k, k2=k)
    return sigma_b**2 + (sigma_w**2 / (k * k)) * Patches_sum


def naive_covariance_next_layer(
    Sigma_xx,
    Sigma_xxbar,
    Sigma_xbarxbar,
    k,
    phi=relu,
    sigma_w=1.0,
    sigma_b=1.0,
    n_gh=30,
    implemented_phi=None,
):
    """
    Computes Sigma^{(l+1)}(x, xbar) from Sigma^{(l)}.

    Inputs:
        Sigma_xx:
            Sigma^{(l)}(x, x; alpha, alpha_bar)
            shape (H_l, W_l, H_l, W_l)

        Sigma_xxbar:
            Sigma^{(l)}(x, xbar; alpha, alpha_bar)
            shape (H_l, W_l, H_l, W_l)

        Sigma_xbarxbar:
            Sigma^{(l)}(xbar, xbar; alpha, alpha_bar)
            shape (H_l, W_l, H_l, W_l)

    Returns:
        Sigma_next:
            Sigma^{(l+1)}(x, xbar)
            shape (H_{l+1}, W_{l+1}, H_{l+1}, W_{l+1})
    """
    H_l, W_l, _, _ = Sigma_xxbar.shape

    H_next = H_l - k + 1
    W_next = W_l - k + 1

    Sigma_next = np.zeros((H_next, W_next, H_next, W_next))

    for a1 in range(H_next):
        for a2 in range(W_next):
            for b1 in range(H_next):
                for b2 in range(W_next):

                    s = 0.0

                    for p1 in range(k):
                        for p2 in range(k):
                            gamma = (a1 + p1, a2 + p2)
                            gamma_bar = (b1 + p1, b2 + p2)

                            q_xx = Sigma_xx[gamma[0], gamma[1], gamma[0], gamma[1]]

                            q_xbarxbar = Sigma_xbarxbar[
                                gamma_bar[0], gamma_bar[1], gamma_bar[0], gamma_bar[1]
                            ]

                            q_xxbar = Sigma_xxbar[
                                gamma[0], gamma[1], gamma_bar[0], gamma_bar[1]
                            ]

                            lambda_cov = np.array(
                                [
                                    [q_xx, q_xxbar],
                                    [q_xxbar, q_xbarxbar],
                                ]
                            )

                            match implemented_phi:
                                case "relu":
                                    s += relu_gaussian_expectation(lambda_cov)
                                case _:
                                    s += bivariate_gaussian_expectation(
                                        phi=phi,
                                        cov=lambda_cov,
                                        n_gh=n_gh,
                                    )

                    Sigma_next[a1, a2, b1, b2] = sigma_b**2 + sigma_w**2 / (k * k) * s

    return Sigma_next


def cumulative_compute_covariance_layers(
    x,
    xbar,
    depth,
    k,
    phi=relu,
    sigma_w=1.0,
    sigma_b=0.0,
    n_gh=30,
    implemented_phi=None,
):
    """
    Computes the CNN-GP covariance Sigma^{(l)} up to layer `depth`.

    Returns:
        Sigmas_xx:
            list containing Sigma^{(l)}(x, x)
        Sigmas_xxbar:
            list containing Sigma^{(l)}(x, xbar)
        Sigmas_xbarxbar:
            list containing Sigma^{(l)}(xbar, xbar)

    The first element of each list corresponds to l = 1.
    """
    Sigmas_xx = []
    Sigmas_xxbar = []
    Sigmas_xbarxbar = []

    # Initialization: layer 1
    Sigma_xx = cumulative_covariance_initialization(
        x, x, k=k, sigma_w=sigma_w, sigma_b=sigma_b
    )
    Sigma_xxbar = cumulative_covariance_initialization(
        x, xbar, k=k, sigma_w=sigma_w, sigma_b=sigma_b
    )
    Sigma_xbarxbar = cumulative_covariance_initialization(
        xbar, xbar, k=k, sigma_w=sigma_w, sigma_b=sigma_b
    )

    Sigmas_xx.append(Sigma_xx)
    Sigmas_xxbar.append(Sigma_xxbar)
    Sigmas_xbarxbar.append(Sigma_xbarxbar)

    # Recursion: layers 2, ..., depth
    for _ in range(1, depth):
        Sigma_xx_next = cumulative_covariance_next_layer(
            Sigma_xx,
            Sigma_xx,
            Sigma_xx,
            k=k,
            phi=phi,
            sigma_w=sigma_w,
            sigma_b=sigma_b,
            n_gh=n_gh,
            implemented_phi=implemented_phi,
        )

        Sigma_xxbar_next = cumulative_covariance_next_layer(
            Sigma_xx,
            Sigma_xxbar,
            Sigma_xbarxbar,
            k=k,
            phi=phi,
            sigma_w=sigma_w,
            sigma_b=sigma_b,
            n_gh=n_gh,
            implemented_phi=implemented_phi,
        )

        Sigma_xbarxbar_next = cumulative_covariance_next_layer(
            Sigma_xbarxbar,
            Sigma_xbarxbar,
            Sigma_xbarxbar,
            k=k,
            phi=phi,
            sigma_w=sigma_w,
            sigma_b=sigma_b,
            n_gh=n_gh,
            implemented_phi=implemented_phi,
        )

        Sigma_xx = Sigma_xx_next
        Sigma_xxbar = Sigma_xxbar_next
        Sigma_xbarxbar = Sigma_xbarxbar_next

        Sigmas_xx.append(Sigma_xx)
        Sigmas_xxbar.append(Sigma_xxbar)
        Sigmas_xbarxbar.append(Sigma_xbarxbar)

    return (
        Sigmas_xx,
        Sigmas_xxbar,
        Sigmas_xbarxbar,
    )


def naive_compute_covariance_layers(
    x,
    xbar,
    depth,
    k,
    phi=relu,
    sigma_w=1.0,
    sigma_b=0.0,
    n_gh=30,
    implemented_phi=None,
):
    """
    Computes the CNN-GP covariance Sigma^{(l)} up to layer `depth`.

    Returns:
        Sigmas_xx:
            list containing Sigma^{(l)}(x, x)
        Sigmas_xxbar:
            list containing Sigma^{(l)}(x, xbar)
        Sigmas_xbarxbar:
            list containing Sigma^{(l)}(xbar, xbar)

    The first element of each list corresponds to l = 1.
    """
    Sigmas_xx = []
    Sigmas_xxbar = []
    Sigmas_xbarxbar = []

    # Initialization: layer 1
    Sigma_xx = naive_covariance_initialization(
        x, x, k=k, sigma_w=sigma_w, sigma_b=sigma_b
    )
    Sigma_xxbar = naive_covariance_initialization(
        x, xbar, k=k, sigma_w=sigma_w, sigma_b=sigma_b
    )
    Sigma_xbarxbar = naive_covariance_initialization(
        xbar, xbar, k=k, sigma_w=sigma_w, sigma_b=sigma_b
    )

    Sigmas_xx.append(Sigma_xx)
    Sigmas_xxbar.append(Sigma_xxbar)
    Sigmas_xbarxbar.append(Sigma_xbarxbar)

    # Recursion: layers 2, ..., depth
    for _ in range(1, depth):
        Sigma_xx_next = naive_covariance_next_layer(
            Sigma_xx,
            Sigma_xx,
            Sigma_xx,
            k=k,
            phi=phi,
            sigma_w=sigma_w,
            sigma_b=sigma_b,
            n_gh=n_gh,
            implemented_phi=implemented_phi,
        )

        Sigma_xxbar_next = naive_covariance_next_layer(
            Sigma_xx,
            Sigma_xxbar,
            Sigma_xbarxbar,
            k=k,
            phi=phi,
            sigma_w=sigma_w,
            sigma_b=sigma_b,
            n_gh=n_gh,
            implemented_phi=implemented_phi,
        )

        Sigma_xbarxbar_next = naive_covariance_next_layer(
            Sigma_xbarxbar,
            Sigma_xbarxbar,
            Sigma_xbarxbar,
            k=k,
            phi=phi,
            sigma_w=sigma_w,
            sigma_b=sigma_b,
            n_gh=n_gh,
            implemented_phi=implemented_phi,
        )

        Sigma_xx = Sigma_xx_next
        Sigma_xxbar = Sigma_xxbar_next
        Sigma_xbarxbar = Sigma_xbarxbar_next

        Sigmas_xx.append(Sigma_xx)
        Sigmas_xxbar.append(Sigma_xxbar)
        Sigmas_xbarxbar.append(Sigma_xbarxbar)

    return Sigmas_xx, Sigmas_xxbar, Sigmas_xbarxbar


def cumulative_final_readout_covariance(Sigma_L):
    """
    Computes the covariance after global average pooling:

        K(x, xbar) =
        1 / |Omega_L|^2 sum_{alpha, alpha_bar}
        Sigma^{(L)}(x, xbar; alpha, alpha_bar)

    Sigma_L: shape (H_L, W_L, H_L, W_L)
    """
    return torch.mean(Sigma_L)


def naive_final_readout_covariance(Sigma_L):
    """
    Computes the covariance after global average pooling:

        K(x, xbar) =
        1 / |Omega_L|^2 sum_{alpha, alpha_bar}
        Sigma^{(L)}(x, xbar; alpha, alpha_bar)

    Sigma_L: shape (H_L, W_L, H_L, W_L)
    """
    return np.mean(Sigma_L)


def cumulative_full_covariance(x, xbar, depth, k, phi=relu, sigma_w=1.0, sigma_b=0.0, n_gh=30, implemented_phi=None):
    """
    Computes the full covariance K(x, xbar) for two images x and xbar.

    Returns:
        K: scalar
    """
    Sigmas_xx, Sigmas_xxbar, Sigmas_xbarxbar = cumulative_compute_covariance_layers(
        x,
        xbar,
        depth=depth,
        k=k,
        phi=phi,
        sigma_w=sigma_w,
        sigma_b=sigma_b,
        n_gh=n_gh,
        implemented_phi=implemented_phi,
    )

    Sigma_L = Sigmas_xxbar[-1]

    return cumulative_final_readout_covariance(Sigma_L)


def naive_full_covariance(x, xbar, depth, k, phi=relu, sigma_w=1.0, sigma_b=0.0, n_gh=30, implemented_phi=None):
    """
    Computes the full covariance K(x, xbar) for two images x and xbar.

    Returns:
        K: scalar
    """
    Sigmas_xx, Sigmas_xxbar, Sigmas_xbarxbar = naive_compute_covariance_layers(
        x,
        xbar,
        depth=depth,
        k=k,
        phi=phi,
        sigma_w=sigma_w,
        sigma_b=sigma_b,
        n_gh=n_gh,
        implemented_phi=implemented_phi,
    )

    Sigma_L = Sigmas_xxbar[-1]

    return naive_final_readout_covariance(Sigma_L)


if __name__ == "__main__":
    # Example usage
    H = 28
    W = H
    depth = 3
    k = 5
    c0 = 3

    x = torch.tensor(np.random.randn(c0, H, W), dtype=torch.float32)
    xbar = torch.tensor(np.random.randn(c0, H, W), dtype=torch.float32)

    Sigmas_xx, Sigmas_xxbar, Sigmas_xbarxbar = cumulative_compute_covariance_layers(
        x,
        xbar,
        depth=depth,
        k=k,
        sigma_w=1.0,
        sigma_b=1.0,
        n_gh=30,
        implemented_phi="relu",
    )
    print(f"Euclidian simple distance between x and xbar: {torch.linalg.norm(x - xbar)}")
    print(f"Distance between x and xbar at input layer: {torch.sqrt(Sigmas_xx[0].mean() + Sigmas_xbarxbar[0].mean() - 2*Sigmas_xxbar[0].mean())}")
    print("Covariance at final layer (x,x):", cumulative_final_readout_covariance(Sigmas_xx[-1]))
    print("Covariance at final layer (x,xbar):", cumulative_final_readout_covariance(Sigmas_xxbar[-1]))
    print("Covariance at final layer (xbar,xbar):", cumulative_final_readout_covariance(Sigmas_xbarxbar[-1]))
    print("=========================================")
    n_Sigmas_xx, n_Sigmas_xxbar, n_Sigmas_xbarxbar = naive_compute_covariance_layers(
        x,
        xbar,
        depth=depth,
        k=k,
        sigma_w=1.0,
        sigma_b=1.0,
        n_gh=30,
        implemented_phi="relu",
    )


    print("Covariance (naive) at final layer (x,x):", naive_final_readout_covariance(n_Sigmas_xx[-1]))
    print("Covariance (naive) at final layer (x,xbar):", naive_final_readout_covariance(n_Sigmas_xxbar[-1]))
    print("Covariance (naive) at final layer (xbar,xbar):", naive_final_readout_covariance(n_Sigmas_xbarxbar[-1]))