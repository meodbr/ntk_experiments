import torch

def predict_finite(x, model):
    model.eval()
    with torch.no_grad():
        output = model(x)
    return output

def predict_infinite(x, gram_matrix, kernel_fn, X_train, y_train):
    # Compute the kernel vector between x and the training data
    k_x = kernel_fn(x, X_train)  # Shape: (n_train,)
    k_x = torch.tensor(k_x, dtype=torch.float32)
    print("Kernel vector:", k_x)
    print("Kernel vector shape:", k_x.shape)

    # Compute the kernel matrix for the training data
    if gram_matrix is None:
        gram_matrix = kernel_fn(X_train, X_train)  # Shape: (n_train, n_train)
        gram_matrix = torch.tensor(gram_matrix, dtype=torch.float32)
        print("Gram matrix shape:", gram_matrix.shape)

    stable_gram = gram_matrix + 1e-6 * torch.eye(gram_matrix.shape[0])  # Add small regularization for numerical stability
    k_inverse = torch.linalg.inv(stable_gram)  # Shape: (n_train, n_train)

    # Solve for the coefficients using the kernel ridge regressio
    y_pred = k_x @ k_inverse @ y_train  # Shape: (1,)

    # Compute variance of the prediction
    k_xx = kernel_fn(x, x)  # Shape: (1,)
    k_xx = torch.tensor(k_xx, dtype=torch.float32)

    y_var = k_xx - k_x @ k_inverse @ k_x.T  # Shape: (1,)
    # keep only diagonal of y_var if it's a matrix
    y_var = torch.diag(y_var) if y_var.dim() == 2 else y_var
    print("Predicted mean:", y_pred)
    print("Predicted variance:", y_var.shape)

    return y_pred, y_var, gram_matrix
