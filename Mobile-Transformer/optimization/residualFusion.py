import torch
import torch.nn as nn

class LayerNormResidual(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNormResidual, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, residual):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        # LayerNorm
        norm_x = (x - mean) / (std + self.eps)

        # Scaling and shifting
        norm_x = norm_x * self.weight + self.bias

        # Residual connection
        return norm_x + residual
