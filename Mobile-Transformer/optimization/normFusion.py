import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormGELU(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNormGELU, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        # LayerNorm
        x = (x - mean) / (std + self.eps)

        # Scaling and shifting
        x = x * self.weight + self.bias

        # GELU activation
        return self.gelu(x)
