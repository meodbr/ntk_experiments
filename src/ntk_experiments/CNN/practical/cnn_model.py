import torch
import torch.nn as nn
import math

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

class NTKConv2d(nn.Module):
    """
    2D convolutional layer in NTK parameterization:

        y = (1/sqrt(in_channels * kernel_size^2)) * W * x + b
    """

    def __init__(self, in_channels, out_channels, kernel_size, beta, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.beta = beta

        # Parameters: W ~ N(0,1)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        # NTK scaling
        weight_scaled = self.weight / math.sqrt(self.in_channels * self.kernel_size**2)
        x = nn.functional.conv2d(x, weight_scaled)

        if self.bias is not None:
            x = x + self.beta*self.bias.view(1, -1, 1, 1)

        return x


class NTKCNN(nn.Module):
    """
    Fully-connected NTK-style CNN:

    - fixed depth and width
    - NTK scaling in forward pass
    - Gaussian initialization N(0,1)
    """

    def __init__(self, input_dim, output_dim, width, depth, beta, kernel_size=5, sigma_w=1.0, sigma_b=1.0, activation=nn.ReLU()):
        super().__init__()

        self.depth = depth
        self.activation = activation
        self.beta = beta
        self.kernel_size = kernel_size
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b

        layers = []

        # Input layer
        layers.append(NTKConv2d(input_dim, width, kernel_size, beta))

        # Hidden layers
        for _ in range(depth-2):
            layers.append(NTKConv2d(width, width, kernel_size, beta))

        self.layers = nn.ModuleList(layers)

        # Output layer (conv into output_dim channels before Global average pooling)
        self.out_layer = NTKConv2d(width, output_dim, kernel_size, beta)

        # Custom initialization (CRITICAL for NTK match)
        self.reset_parameters()

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

        for i, layer in enumerate(self.layers):
            h = layer(h)
            # print("Model hidden shape for layer ", i, ":", h.shape)
            h = self.activation(h)

        out = self.out_layer(h)
        # print("Model output shape:", out.shape)
        out = out.mean(dim=[2,3])  # Global average pooling

        return out.squeeze(-1)
    

if __name__ == "__main__":
    # Test the model
    model = NTKCNN(input_dim=3, output_dim=10, width=64, depth=5, beta=1.0)
    x = torch.randn(8, 3, 32, 32)  # Batch of 8 images
    out = model(x)
    print(out.shape)  # Should be (8, 10)