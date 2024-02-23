from dataclasses import dataclass
import math
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from tokenizer import Tokenizer, TokenizerConfig
from mlm_model import BERTwithMLMHead
from model import BERTConfig

@dataclass
class MLMTrainConfig:
  mlm_ratio: float = 0.2

tokenizer_config = TokenizerConfig()
tokenizer = Tokenizer(tokenizer_config)

model_config = BERTConfig(embed_dim=8, n_heads=2)
model = BERTwithMLMHead(model_config)

train_config = MLMTrainConfig()

def mlm_collate_fn(batch: List[str]) -> Dict:
  tokenized = tokenizer.batch_encode(batch, max_length=model.config.n_positions, return_type='pt')

  # Create the attention mask
  attention_mask = torch.zeros(tokenized.shape, dtype=torch.long)
  attention_mask = torch.where(tokenized == 0, attention_mask, 1)

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
    "mlm_mask": mlm_mask
  }
    
  

def train(model, tokenizer, criterion, train_dataloader):

  model.train()

  for batch in train_dataloader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    mlm_mask = batch["mlm_mask"]

    input_ids = torch.where(mlm_mask == 0, input_ids, tokenizer.mask_token_id)
    output = model(input_ids, attention_mask=attention_mask) # output size: (bs, seq_len, vocab_size)

    num_mask_tokens = int(mlm_mask.sum().item())
    output = output.masked_select(mlm_mask.unsqueeze(-1) == 1).view(num_mask_tokens, -1)

    labels = torch.masked_select(input_ids, mlm_mask == 1)
    
    loss = criterion(output, labels)

    loss.backward()


texts = ["مرحبا من أنت؟", "اهلا وسهلا"] * 10
dataloader = DataLoader(texts, batch_size=2, shuffle=False, collate_fn=mlm_collate_fn)
print(next(iter(dataloader)))


# for _ in range(n_epochs):
#   train(model, tokenizer, criterion, train_dataloader)
#   evaluate(model, tokenizer, criterion, test_dataloader)