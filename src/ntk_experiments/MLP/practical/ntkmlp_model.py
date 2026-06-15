import torch
import torch.nn as nn
import math

from ntk_experiments.config import config

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
        # print(f"NTKLinear: input shape: {x.shape}")
        x = (self.weight @ x.T).T / math.sqrt(self.in_features)

        # print(f"NTKLinear: after weight shape: {x.shape}")

        if self.bias is not None:
            x = x + self.beta*self.bias
        
        # print(f"NTKLinear: after bias shape: {x.shape}")

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

    def __init__(self, input_dim, output_dim, width, depth, beta, sigma_w=1.0, sigma_b=1.0, activation=nn.ReLU()):
        super().__init__()

        self.depth = depth
        self.activation = activation
        self.beta = beta
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b

        layers = []

        # Input layer
        layers.append(NTKLinear(input_dim, width, beta))

        # Hidden layers
        for _ in range(depth-1):
            layers.append(NTKLinear(width, width, beta))

        self.layers = nn.ModuleList(layers)

        # Output layer (scalar output NTK common choice)
        self.out_layer = NTKLinear(width, output_dim, beta)

        self.reset_parameters()

    # Custom initialization (CRITICAL for NTK match)

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=self.sigma_w)
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=0.0, std=self.sigma_b)

        nn.init.normal_(self.out_layer.weight, mean=0.0, std=self.sigma_w)
        if self.out_layer.bias is not None:
            nn.init.normal_(self.out_layer.bias, mean=0.0, std=self.sigma_b)

    # Forward pass

    def forward(self, x):
        """
        x shape: (batch, input_dim)
        """

        # NTK convention: work with row vectors
        h = x
        # print(f"Input shape: {h.shape}")

        for layer in self.layers:
            h = layer(h)
            h = self.activation(h)

        out = self.out_layer(h)
        # print(f"Output shape: {out.shape}")

        return out.squeeze(-1)

# Simple MLP
class MLP_classic(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.net = nn.Sequential(
            nn.Linear(config.INPUT_DIM, width),
            nn.ReLU(),
            nn.Linear(width, config.OUTPUT_DIM)
        )

    def forward(self, x):
        return self.net(x)# / math.sqrt(self.width)  # NTK scaling


class Old_NTKMLP(nn.Module):
    def __init__(self, width):
        super().__init__()

        self.fc1 = nn.Linear(config.INPUT_DIM, width, bias=True)
        self.fc2 = nn.Linear(width, config.OUTPUT_DIM, bias=True)

        self.width = width

        self.reset_parameters()

    def reset_parameters(self):
        # N(0,1) weights (NOT Kaiming)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0)

        nn.init.normal_(self.fc1.bias, mean=0.0, std=config.BETA)
        nn.init.normal_(self.fc2.bias, mean=0.0, std=config.BETA)

    def forward(self, x):
        # explicit NTK scaling
        x = self.fc1(x) / math.sqrt(self.width)
        x = torch.relu(x)
        x = self.fc2(x) / math.sqrt(self.width)

        return x