import numpy as np
from numpy.polynomial.hermite import hermgauss


def relu(x):
    return np.maximum(x, 0.0)


def bivariate_gaussian_expectation(phi, cov, n_gh=30, jitter=1e-10):
    """
    Computes E[phi(u) phi(v)] where (u,v) ~ N(0, cov)
    using 2D Gauss-Hermite quadrature.

    cov: shape (2, 2)
    """
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

    return (
        np.sqrt(q_xx * q_xbarxbar) * (np.arcsin(rho) + np.pi / 2) + q_xxbar / 2
    )


def covariance_initialization(x, xbar, k, sigma_w=1.0, sigma_b=1.0):
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
                                s += (
                                    x[c, a1 + p1, a2 + p2]
                                    * xbar[c, b1 + p1, b2 + p2]
                                )

                    Sigma[a1, a2, b1, b2] = (
                        sigma_b**2
                        + sigma_w**2 / (C0 * k * k) * s
                    )

    return Sigma


def covariance_next_layer(
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

                            q_xx = Sigma_xx[
                                gamma[0], gamma[1],
                                gamma[0], gamma[1]
                            ]

                            q_xbarxbar = Sigma_xbarxbar[
                                gamma_bar[0], gamma_bar[1],
                                gamma_bar[0], gamma_bar[1]
                            ]

                            q_xxbar = Sigma_xxbar[
                                gamma[0], gamma[1],
                                gamma_bar[0], gamma_bar[1]
                            ]

                            lambda_cov = np.array([
                                [q_xx, q_xxbar],
                                [q_xxbar, q_xbarxbar],
                            ])

                            match implemented_phi:
                                case "relu":
                                    s += relu_gaussian_expectation(lambda_cov)
                                case _:
                                    s += bivariate_gaussian_expectation(
                                        phi=phi,
                                        cov=lambda_cov,
                                        n_gh=n_gh,
                                    )

                    Sigma_next[a1, a2, b1, b2] = (
                        sigma_b**2
                        + sigma_w**2 / (k * k) * s
                    )

    return Sigma_next


def compute_covariance_layers(
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
    Sigma_xx = covariance_initialization(
        x, x, k=k, sigma_w=sigma_w, sigma_b=sigma_b
    )
    Sigma_xxbar = covariance_initialization(
        x, xbar, k=k, sigma_w=sigma_w, sigma_b=sigma_b
    )
    Sigma_xbarxbar = covariance_initialization(
        xbar, xbar, k=k, sigma_w=sigma_w, sigma_b=sigma_b
    )

    Sigmas_xx.append(Sigma_xx)
    Sigmas_xxbar.append(Sigma_xxbar)
    Sigmas_xbarxbar.append(Sigma_xbarxbar)

    # Recursion: layers 2, ..., depth
    for _ in range(1, depth):
        Sigma_xx_next = covariance_next_layer(
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

        Sigma_xxbar_next = covariance_next_layer(
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

        Sigma_xbarxbar_next = covariance_next_layer(
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


def final_readout_covariance(Sigma_L):
    """
    Computes the covariance after global average pooling:

        K(x, xbar) =
        1 / |Omega_L|^2 sum_{alpha, alpha_bar}
        Sigma^{(L)}(x, xbar; alpha, alpha_bar)

    Sigma_L: shape (H_L, W_L, H_L, W_L)
    """
    return np.mean(Sigma_L)

if __name__ == "__main__":
    # Example usage
    H = 8
    W = H

    x = np.random.randn(3, H, W)  # Random image 1
    xbar = np.random.randn(3, H, W)  # Random image 2

    depth = 3
    k = 3

    Sigmas_xx, Sigmas_xxbar, Sigmas_xbarxbar = compute_covariance_layers(
        x, xbar, depth=depth, k=k, sigma_w=1.0, sigma_b=1.0, n_gh=30, implemented_phi="relu"
    )

    print(f"Euclidian simple distance between x and xbar: {np.linalg.norm(x - xbar)}")
    print(f"Distance between x and xbar at input layer: {np.sqrt(Sigmas_xx[0].mean() + Sigmas_xbarxbar[0].mean() - 2*Sigmas_xxbar[0].mean())}")
    print("Covariance at final layer (x,x):", final_readout_covariance(Sigmas_xx[-1]))
    print("Covariance at final layer (x,xbar):", final_readout_covariance(Sigmas_xxbar[-1]))
    print("Covariance at final layer (xbar,xbar):", final_readout_covariance(Sigmas_xbarxbar[-1]))