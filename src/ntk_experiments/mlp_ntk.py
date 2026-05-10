import torch
import torch.nn as nn
import math


# ============================================================
# NTK-STYLE LINEAR LAYER
# ============================================================

class NTKLinear(nn.Module):
    """
    Linear layer in NTK parameterization:

        y = (1/sqrt(in_features)) * W x + b
    """

    def __init__(self, in_features, out_features, beta, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.beta = beta

        # Parameters: W ~ N(0,1)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features)
        )

        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None

    def forward(self, x):
        # NTK scaling
        x = (self.weight @ x.T).T / math.sqrt(self.in_features)

        if self.bias is not None:
            x = x + self.beta*self.bias

        return x


# ============================================================
# NTK MLP
# ============================================================

class NTKMLP(nn.Module):
    """
    Fully-connected NTK-style MLP:

    - fixed depth and width
    - NTK scaling in forward pass
    - Gaussian initialization N(0,1)
    """

    def __init__(self, input_dim, width, depth, beta, activation=nn.ReLU()):
        super().__init__()

        self.depth = depth
        self.activation = activation
        self.beta = beta

        layers = []

        # Input layer
        layers.append(NTKLinear(input_dim, width, beta))

        # Hidden layers
        for _ in range(depth-1):
            layers.append(NTKLinear(width, width, beta))

        self.layers = nn.ModuleList(layers)

        # Output layer (scalar output NTK common choice)
        self.out_layer = NTKLinear(width, 1, beta)

        self.reset_parameters()

    # --------------------------------------------------------
    # Custom initialization (CRITICAL for NTK match)
    # --------------------------------------------------------

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=1.0)
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=0.0, std=1.0)

        nn.init.normal_(self.out_layer.weight, mean=0.0, std=1.0)
        if self.out_layer.bias is not None:
            nn.init.normal_(self.out_layer.bias, mean=0.0, std=1.0)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    def forward(self, x):
        """
        x shape: (batch, input_dim)
        """

        # NTK convention: work with row vectors
        h = x

        for layer in self.layers:
            h = layer(h)
            h = self.activation(h)

        out = self.out_layer(h)

        return out.squeeze(-1)