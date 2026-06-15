import torch

def compute_gram_matrix(kernel_function, X):
    """
    Compute the Gram matrix for a given kernel function and input data X.
    """
    n_samples = X.shape[0]
    gram_matrix = torch.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):
            gram_matrix[i, j] = kernel_function(X[i:i+1], X[j:j+1])
            gram_matrix[j, i] = gram_matrix[i, j]  # Symmetric

    return gram_matrix
