from dataclasses import dataclass
import math
import torch
from torch import nn

import torch.nn.functional as F


@dataclass
class Config:
    embed_dim: int = 32
    n_heads: int = 4
    bias: bool = False
    dropout_prob: float = 0.2


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=config.bias)
        self.fc2 = nn.Linear(4 * config.embed_dim, config.embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.attn_mat = nn.Linear(
            config.embed_dim, 3 * config.embed_dim, bias=config.bias
        )
        self.proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout_prob)
        self.proj_dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        head_embed_dim = embed_dim // self.config.n_heads

        qkv = self.attn_mat(x)  # batch_size, seq_len, embed_dim * 3
        q, k, v = qkv.split(embed_dim, dim=-1)
        q = q.view(
            batch_size, seq_len, self.config.n_heads, head_embed_dim
        )  # (batch_size, seq_len, n_heads, embed_dim // n_heads)
        k = k.view(batch_size, seq_len, self.config.n_heads, head_embed_dim)
        v = v.view(batch_size, seq_len, self.config.n_heads, head_embed_dim)

        q = q.transpose(-2, -3)  # (batch_size, n_heads, seq_len, head_embed_dim)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        qk_dot = (
            q @ k.transpose(-1, -2) / math.sqrt(head_embed_dim)
        )  # (batch_size, n_heads, seq_len, seq_len)
        attn_scores = F.softmax(qk_dot, dim=-1)
        attn_scores = self.attn_dropout(attn_scores)
        embeddings = attn_scores @ v  # (batch_size, n_heads, seq_len, head_embed_dim)
        embeddings = embeddings.transpose(
            -2, -3
        ).contiguous()  # (batch_size, seq_len, n_heads, head_embed_dim)
        embeddings = embeddings.view(batch_size, seq_len, embed_dim)

        embeddings = self.proj(embeddings)
        embeddings = self.proj_dropout(embeddings)

        return embeddings


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self_attn = SelfAttention(config)
        self.layer_norm = nn.LayerNorm()
        self.mlp = MLP(config)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        embeddings = self.self_attn(x)
        x = embeddings + x
        x = self.layer_norm(x)
        x = self.mlp(x)
        # TODO: Add addition and layernorm here

        return x


x = torch.randn((4, 15, 32))
config = Config()
self_attn = SelfAttention(config)
self_attn.train()
print(self_attn(x).size())
