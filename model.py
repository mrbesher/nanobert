import torch
from torch import nn


class Config:
  embed_dim: int = 32
  n_heads: int = 4
  bias: bool = False

class SelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.attn_mat = nn.Linear(config.embed_dim, 3*config.embed_dim, bias=config.bias)
    pass

  def forward(self, x):
    batch_size, seq_len, embed_dim = x.size()

    qkv = self.attn_mat(x) # batch_size, seq_len, embed_dim * 3
    q, k, v = qkv.split(embed_dim, dim=-1)
    q = q.view(batch_size, seq_len, self.config.n_heads, embed_dim// self.config.n_heads) # (batch_size, seq_len, n_heads, embed_dim // n_heads)
    k = k.view(batch_size, seq_len, self.config.n_heads, embed_dim // self.config.n_heads)
    v = v.view(batch_size, seq_len, self.config.n_heads, embed_dim // self.config.n_heads)
    
    q = q.transpose(-2, -3) # (batch_size, n_heads, seq_len, embed_dim // n_heads)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)
    return q, k, v