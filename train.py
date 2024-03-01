from dataclasses import dataclass
import math
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from tokenizer import Tokenizer, TokenizerConfig
from mlm_model import BERTwithMLMHead
from model import BERTConfig


@dataclass
class MLMTrainConfig:
    mlm_ratio: float = 0.2
    label_smoothing: float = 0.01
    lr: float = 1e-3


def mlm_collate_fn(batch: List[str]) -> Dict:
    """
    Creates the necessary tensors required for training a Masked Language Model (MLM) using. Tokenizes the given batch of strings, creates attention masks, and generates random masks for the MLM task.

    Args:
        - batch (List[str]): A list of strings representing texts to be used during training.

    Returns:
        - dict: A dictionary containing the following keys:
            + 'input_ids': Torch Tensor of shape (batch_size, seq_length) holding the token ids generated from the provided text batch.
            + 'attention_mask': Torch Tensor of shape (batch_size, seq_length) filled with either 0 or 1 indicating presence or absence of tokens respectively.
            + 'mlm_mask': Torch Tensor of shape (batch_size, seq_length) having values 0 or 1 depending if the token has been chosen for masking in the MLM task.
    """
    tokenized = tokenizer.batch_encode(
        batch, max_length=model.config.n_positions, return_type="pt", padding="longest"
    )

    # Create the attention mask
    attention_mask = (tokenized != 0).long()

    # Create the MLM mask
    input_lens = attention_mask.sum(dim=-1).tolist()

    mlm_mask = torch.zeros(tokenized.shape, dtype=torch.long)
    for idx, input_len in enumerate(input_lens):
        n_masked_tokens = math.ceil(train_config.mlm_ratio * input_len)
        masked_token_idx = torch.randperm(input_len)[:n_masked_tokens]
        mlm_mask[idx, masked_token_idx] = 1

    return {
        "input_ids": tokenized,
        "attention_mask": attention_mask,
        "mlm_mask": mlm_mask,
    }


def train(model, optimizer, criterion, train_dataloader, tokenizer):
    model.train()

    for batch in train_dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        mlm_mask = batch["mlm_mask"]

        input_ids = torch.where(mlm_mask == 0, input_ids, tokenizer.mask_token_id)

        optimizer.zero_grad()
        output = model(
            input_ids, attention_mask=attention_mask
        )  # output size: (bs, seq_len, vocab_size)

        num_mask_tokens = int(mlm_mask.sum().item())
        output = output.masked_select(mlm_mask.unsqueeze(-1) == 1).view(
            num_mask_tokens, -1
        )

        labels = torch.masked_select(input_ids, mlm_mask == 1)

        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()


def evaluate(model, criterion, metric_fn, test_dataloader, tokenizer):
    model.eval()

    losses = []
    scores = []

    for batch in test_dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        mlm_mask = batch["mlm_mask"]

        input_ids = torch.where(mlm_mask == 0, input_ids, tokenizer.mask_token_id)

        with torch.no_grad():
            output = model(
                input_ids, attention_mask=attention_mask
            )  # output size: (bs, seq_len, vocab_size)

        num_mask_tokens = int(mlm_mask.sum().item())
        output = output.masked_select(mlm_mask.unsqueeze(-1) == 1).view(
            num_mask_tokens, -1
        )

        labels = torch.masked_select(input_ids, mlm_mask == 1)

        loss = criterion(output, labels)
        score = metric_fn(output, labels)

        losses.append(loss.detach().item())
        scores.append(score.detach().item())

    return losses, scores
        

model_config = BERTConfig(embed_dim=8, n_heads=2)
tokenizer_config = TokenizerConfig()
train_config = MLMTrainConfig()

tokenizer = Tokenizer(tokenizer_config)
model = BERTwithMLMHead(model_config)

texts = ["مرحبا من أنت؟", "اهلا وسهلا"] * 10
dataloader = DataLoader(texts, batch_size=2, shuffle=False, collate_fn=mlm_collate_fn)

criterion = nn.CrossEntropyLoss(label_smoothing=train_config.label_smoothing)
adam_optim = torch.optim.Adam(model.parameters(), lr=train_config.lr)


for _ in range(1):
  train(model, adam_optim, criterion, dataloader, tokenizer)
  results = evaluate(model, criterion, lambda x, y: torch.tensor(0), dataloader, tokenizer)
  print(results)
