from dataclasses import dataclass
import math
import torch
from torch import nn

import torch.nn.functional as F


@dataclass
class Config:
    n_layers: int = 3
    embed_dim: int = 32
    n_heads: int = 4
    bias: bool = False
    dropout_prob: float = 0.2
    vocab_size: int = 64
    n_positions: int = 64
    

def get_relative_position(n_positions: int, n_heads: int) -> torch.Tensor:
    # Source: ofirpress/attention_with_linear_biases
    context_position = torch.arange(n_positions)[:, None]
    memory_position = torch.arange(n_positions)[None, :]

    relative_position = memory_position - context_position
    relative_position = torch.abs(relative_position).unsqueeze(0).expand(n_heads, -1, -1)

    return relative_position

def get_slopes(n_heads):
    # Source: https://nn.labml.ai/transformers/alibi/index.html
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)

    m = torch.pow(m_0, torch.arange(1, 1 + n))

    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)

        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))

        m = torch.cat([m, m_hat])

    return m



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
    def __init__(self, config, alibi_matrix):
        super().__init__()
        self.config = config

        self.attn_mat = nn.Linear(
            config.embed_dim, 3 * config.embed_dim, bias=config.bias
        )
        self.proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout_prob)
        self.proj_dropout = nn.Dropout(config.dropout_prob)

        self.alibi_matrix = alibi_matrix

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
        qk_dot = qk_dot + self.alibi_matrix[:, :seq_len, :seq_len]

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
    def __init__(self, config, alibi_matrix):
        super().__init__()
        self.config = config
        self.self_attn = SelfAttention(config, alibi_matrix)
        self.attn_layer_norm = nn.LayerNorm(config.embed_dim)
        self.mlp_layer_norm = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        # input size: (batch_size, seq_len, embed_dim)
        x = self.self_attn(x) + x
        x = self.attn_layer_norm(x)
        x = self.mlp(x) + x
        x = self.mlp_layer_norm(x)
        return x


class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        
        # ALIBI
        relative_pos = get_relative_position(config.n_positions, config.n_heads)
        slopes = get_slopes(config.n_heads)
        alibi_matrix = relative_pos * slopes[:, None, None]

        self.encoder_blocks = nn.ModuleList([EncoderBlock(config, alibi_matrix) for _ in range(config.n_layers)])
    
    def forward(self, x):
        # input size: (batch_size, seq_len)
        x = self.embedding(x) # (batch_size, seq_len, embed_dim)
        # TODO: Add positional embeddings

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        
        return x



x = torch.LongTensor([[1, 9, 15, 0, 6], [8, 1, 2, 21, 6]])
config = Config()
bert = BERT(config)
print(bert(x).size())