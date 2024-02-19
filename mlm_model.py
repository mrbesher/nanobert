from dataclasses import dataclass

import torch
from torch import nn

from model import BERT


@dataclass
class Config:
    n_layers: int = 3
    embed_dim: int = 32
    n_heads: int = 4
    bias: bool = False
    dropout_prob: float = 0.2
    vocab_size: int = 64
    n_positions: int = 64

class BERTwithMLMHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.bert = BERT(config)

        # Transformation
        self.linear_proj= nn.Linear(config.embed_dim, config.embed_dim)
        self.hidden_activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.embed_dim)

        # LM head
        self.mlm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=config.bias)

    def forward(self, x, attention_mask=None):
        # input size: (bs, seq_len)
        x = self.bert(x, attention_mask=attention_mask) # (bs, seq_len, embed_dim)
        x = self.linear_proj(x) # (bs, seq_len, embed_dim)
        x = self.hidden_activation(x)
        x = self.layer_norm(x)

        x = self.mlm_head(x)

        return x


if __name__ == '__main__':
    config = Config(embed_dim=8, n_heads=2)
    bert_mlm = BERTwithMLMHead(config)
    x = torch.LongTensor([[4, 6, 1, 2, 0, 0], [6, 8, 1, 2, 4, 7]])
    attention_mask = torch.LongTensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    output = bert_mlm(x, attention_mask=attention_mask)
    assert output.shape == torch.Size([x.shape[0], x.shape[1], config.vocab_size])