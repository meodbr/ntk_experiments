"""
This script implements the theoretical infinite-width Neural
Tangent Kernel (NTK) recursion for a fully-connected multilayer
perceptron (MLP) of arbitrary depth L.

The implementation follows:

    Jacot, Gabriel, Hongler (2018)
    "Neural Tangent Kernel: Convergence and Generalization
     in Neural Networks"

Paper:
https://arxiv.org/abs/1806.07572


# COVARIANCE RECURSION (Σ)

Base layer:

    Σ^(0)(x,x') = (1/d) x^T x' + β²

Recursive covariance:

    Σ^(ℓ+1)(x,x')
        = σ_w² E[σ(u)σ(v)] + β²

Derivative covariance:

    dΣ^(ℓ+1)(x,x')
        = σ_w² E[σ'(u)σ'(v)]

where:

    (u,v) ~ N(0, Λ^(ℓ))

with covariance matrix:

    Λ^(ℓ) = [ Σ^(ℓ)(x,x)    Σ^(ℓ)(x,x')  ]
             [ Σ^(ℓ)(x',x)   Σ^(ℓ)(x',x') ]

# NTK RECURSION (Θ)

The infinite-width NTK recursion is:

    Θ^(1)(x,x') = Σ^(1)(x,x')

and recursively:

    Θ^(ℓ+1)(x,x')
        = Σ^(ℓ+1)(x,x')
          + dΣ^(ℓ+1)(x,x') Θ^(ℓ)(x,x')

"""

import math

import numpy as np
from numpy.polynomial.hermite import hermgauss


# GAUSSIAN EXPECTATION HELPERS

def gaussian_expectation(cov, f, n_gh=40):
    """
    Compute a Gaussian expectation:

        E[f(u,v)]

    where:

        (u,v) ~ N(0, cov)

    using Gauss-Hermite quadrature.

    Parameters
    ----------
    cov : ndarray shape (2,2)
        Covariance matrix.

    f : callable
        Function f(u,v).

    n_gh : int
        Number of Gauss-Hermite quadrature points.

    Returns
    -------
    float
        Approximation of the Gaussian expectation.
    """

    # Gauss-Hermite nodes and weights
    xs, ws = hermgauss(n_gh)

    # Convert to standard Gaussian coordinates
    xs = xs * np.sqrt(2.0)
    ws = ws / np.sqrt(np.pi)

    # Cholesky decomposition
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(2))

    expectation = 0.0

    for i in range(n_gh):
        for j in range(n_gh):

            z = np.array([xs[i], xs[j]])

            u, v = L @ z

            expectation += ws[i] * ws[j] * f(u, v)

    return expectation

def gaussian_expectation_relu_prime(cov):
    """
    Computes E(u, v)~N(0, cov)[ReLU'(u)ReLU'(v)]
    """
    rho = cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1])
    rho = np.clip(rho, -0.9999999, 0.9999999)  # Avoid numerical issues with arcsin
    expectation = .25 + (1/(2*math.pi)) * math.asin(rho)
    return expectation

def gaussian_expectation_relu(cov):
    """
    Computes E(u, v)~N(0, cov)[ReLU(u)ReLU(v)]
    """
    rho = cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1])
    rho = np.clip(rho, -0.9999999, 0.9999999)  # Avoid numerical issues with arcos and sqrt
    
    expectation = (
        (math.sqrt(cov[0, 0]*cov[1, 1]) / (2*math.pi))
        * (math.sqrt(1-rho*rho) + (math.pi-math.acos(rho)) * rho)
    )
    return expectation



# ===============================================================
# INFINITE-WIDTH NTK RECURSION
# ===============================================================


def infinite_width_ntk(
    x,
    xp,
    depth,
    sigma=None,
    sigma_prime=None,
    implemented_sigma=None,
    sigma_w=1.0,
    beta=1.0,
    n_gh=40,
):
    """
    Compute the theoretical infinite-width NTK recursively.

    Parameters
    ----------
    x : ndarray
        First input vector.

    xp : ndarray
        Second input vector.

    depth : int
        Number of hidden layers L.

    sigma : callable
        Activation function σ.

    sigma_prime : callable
        Derivative σ'.

    sigma_w : float
        Weight variance scaling.

    beta : float
        Bias scaling coefficient.

    n_gh : int
        Number of Gauss-Hermite quadrature points.

    Returns
    -------
    theta : float
        Infinite-width NTK Θ^(L)(x,x').

    sigma_cov : float
        Final covariance Σ^(L)(x,x').

    sigma_matrix : ndarray
        Final 2×2 covariance matrix.
    """

    if implemented_sigma is None:
        assert sigma is not None and sigma_prime is not None

    # Input dimension

    d = x.shape[0]
    print(f"Input dimension d for theoretical NTK: {d}")

    # Base covariance Σ^(0)
    # Σ^(0)(x,x') = (1/d) x^T x' + β²

    Sigma_xx = np.dot(x, x) / d + beta**2
    Sigma_xxp = np.dot(x, xp) / d + beta**2
    Sigma_xpxp = np.dot(xp, xp) / d + beta**2

    Sigma = np.array([
        [Sigma_xx, Sigma_xxp],
        [Sigma_xxp, Sigma_xpxp],
    ])

    # -----------------------------------------------------------
    # Initial NTK
    # -----------------------------------------------------------

    Theta = Sigma_xxp

    # -----------------------------------------------------------
    # Recursive propagation through layers
    # -----------------------------------------------------------

    for ell in range(depth):

        Sigma_next = np.zeros((2, 2))
        DotSigma_next = np.zeros((2, 2))

        # -------------------------------------------------------
        # Compute Σ^(ℓ+1)
        # Compute dΣ^(ℓ+1)
        # -------------------------------------------------------

        for i in range(2):
            for j in range(2):

                cov = np.array([
                    [Sigma[i, i], Sigma[i, j]],
                    [Sigma[j, i], Sigma[j, j]],
                ])

                # ------------------------------------------------
                # Σ recursion
                #
                # Σ^(ℓ+1)
                #   = σ_w² E[σ(u)σ(v)] + β²
                # ------------------------------------------------

                g_expectation = None
                match implemented_sigma:
                    case "relu":
                        g_expectation = gaussian_expectation_relu(cov)
                    case _:
                        g_expectation = gaussian_expectation(
                            cov,
                            lambda u, v: sigma(u) * sigma(v),
                            n_gh=n_gh,
                        )

                Sigma_next[i, j] = (
                    sigma_w**2
                    * g_expectation
                    + beta**2
                )

                # ------------------------------------------------
                # dΣ recursion
                #
                # dΣ^(ℓ+1)
                #   = σ_w² E[σ'(u)σ'(v)]
                # ------------------------------------------------

                g_expectation_prime = None
                match implemented_sigma:
                    case "relu":
                        g_expectation_prime = gaussian_expectation_relu_prime(cov)
                    case _:
                        g_expectation_prime = gaussian_expectation(
                            cov,
                            lambda u, v:
                            sigma_prime(u) * sigma_prime(v),
                            n_gh=n_gh,
                        )

                DotSigma_next[i, j] = (
                    sigma_w**2
                    * g_expectation_prime
                )

        # -------------------------------------------------------
        # NTK recursion
        #
        # Θ^(ℓ+1)
        #   = Σ^(ℓ+1)
        #     + dΣ^(ℓ+1) Θ^(ℓ)
        # -------------------------------------------------------

        Theta = (
            Sigma_next[0, 1]
            + DotSigma_next[0, 1] * Theta
        )

        # Update covariance
        Sigma = Sigma_next

    return Theta, Sigma[0, 1], Sigma


# ===============================================================
# ACTIVATION FUNCTIONS
# ===============================================================


def relu(z):
    """ReLU activation."""

    return np.maximum(0.0, z)



def relu_prime(z):
    """Derivative of ReLU."""

    return 1.0 * (z > 0)



def tanh(z):
    """tanh activation."""

    return np.tanh(z)



def tanh_prime(z):
    """Derivative of tanh."""

    return 1.0 - np.tanh(z)**2


# ===============================================================
# EXAMPLE USAGE
# ===============================================================

if __name__ == "__main__":

    np.random.seed(0)

    # -----------------------------------------------------------
    # Example inputs
    # -----------------------------------------------------------

    d = 32

    x = np.random.randn(d)
    xp = np.random.randn(d)

    # -----------------------------------------------------------
    # Network hyperparameters
    # -----------------------------------------------------------

    depth = 5

    sigma_w = np.sqrt(2.0)
    beta = 1.

    # -----------------------------------------------------------
    # Compute infinite-width NTK
    # -----------------------------------------------------------

    Theta, Sigma_cov, Sigma_matrix = infinite_width_ntk(
        x=x,
        xp=xp,
        depth=depth,
        sigma=relu,
        sigma_prime=relu_prime,
        sigma_w=sigma_w,
        beta=beta,
        n_gh=40,
    )

    # -----------------------------------------------------------
    # Results
    # -----------------------------------------------------------

    print("=" * 60)
    print("INFINITE-WIDTH NTK")
    print("=" * 60)

    print(f"Depth L                : {depth}")
    print(f"Weight scale σ_w       : {sigma_w:.4f}")
    print(f"Bias scale β           : {beta:.4f}")

    print()

    print(f"Σ^(L)(x,x')            : {Sigma_cov:.8f}")
    print(f"Θ^(L)(x,x')            : {Theta:.8f}")

    print()

    print("Final covariance matrix Σ^(L):")
    print(Sigma_matrix)

    print()
    print("Done.")
