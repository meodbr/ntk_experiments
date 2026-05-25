import torch

def frobenius_norm(operator: torch.Tensor):
    flattened = operator.view(-1)
    return (flattened @ flattened.T).item()

def frobenius_relative_change(initial: torch.Tensor, current: torch.Tensor):
    initial_norm = frobenius_norm(initial)
    net_change_norm = frobenius_norm(current - initial)
    return net_change_norm / initial_norm