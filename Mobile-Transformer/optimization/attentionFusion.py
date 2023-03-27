import torch
import torch.nn as nn

class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(EfficientMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_linear = nn.Linear(d_model, d_model * 3)
        self.out_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Compute Q, K, and V with a single linear layer
        qkv = self.qkv_linear(x)
        qkv = qkv.view(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        q, k, v = torch.split(qkv, self.head_dim, dim=-1)

        # Transpose Q, K, V for multi-head attention
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Weighted sum of V
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final linear layer
        out = self.out_linear(attn_output)
        return out
