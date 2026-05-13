
import torch
import torch.nn as nn
import math
from torch.func import jacrev
from torch.func import functional_call

def reshape_to_2D_jacobian(jacobian_dict, output_size):
    # Reshape the Jacobian from a dict of parameter tensors to a single 2D matrix
    # Each row corresponds to the gradient of one output dimension w.r.t. all parameters
    jacobian_rows = []
    for key, jac in jacobian_dict.items():
        # jac shape: (output_size, param_shape...)
        jac_reshaped = jac.reshape(output_size, -1)  # Shape: (output_size, num_params)
        jacobian_rows.append(jac_reshaped)

    full_jacobian = torch.cat(jacobian_rows, dim=1)  # Shape: (output_size, total_num_params)
    return full_jacobian

def empirical_ntk(model, x, x_prime):
    # The neural tangent kernel between two inputs x and x':
    # K(x, x') = J(x) @ J(x').T where J is the Jacobian of the network output w.r.t. parameters
    # Compute Jacobian for x

    def model_output_x(params):
        return functional_call(model, params, x)
    
    def model_output_x_prime(params):
        return functional_call(model, params, x_prime)

    output_size = model(x).shape[-1] 
    
    params = dict(model.named_parameters())
    print("Number of parameters:", sum(p.numel() for p in params.values()))
    print("Parameter shapes:")
    for name, p in params.items():
        print(f"{name}: {p.shape}")

    J_x = jacrev(model_output_x)(params)
    J_x = reshape_to_2D_jacobian(J_x, output_size)
    print("J_x shape:", J_x.shape)
    J_x_prime = jacrev(model_output_x_prime)(params)
    J_x_prime = reshape_to_2D_jacobian(J_x_prime, output_size)
    print("J_x_prime shape:", J_x_prime.shape)

    ntk = J_x @ J_x_prime.T
    print("Empirical NTK shape:", ntk.shape)
    if ntk.numel() == 1:
        print("Empirical NTK value:", ntk.item())

    return ntk

def simulate_batched_empirical_ntk(model, x_batch, x_prime_batch):
    # Compute the empirical NTK for batches of inputs
    batch_size = x_batch.shape[0]
    ntk_matrix = torch.zeros((batch_size, batch_size))

    for i in range(batch_size):
        for j in range(batch_size):
            ntk_matrix[i, j] = empirical_ntk(model, x_batch[i:i+1], x_prime_batch[j:j+1])

    return ntk_matrix