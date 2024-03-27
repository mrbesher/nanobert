import math
from typing import Dict, List

import torch

from tokenizer import Tokenizer


def mlm_collate_fn(batch: List[str], tokenizer: Tokenizer, max_length: int, mlm_ratio: float) -> Dict:
    """
    Creates the necessary tensors required for training a Masked Language Model (MLM) using. Tokenizes the given batch of strings, creates attention masks, and generates random masks for the MLM task.

    Args:
        - batch (List[str]): A list of strings representing texts to be used during training.
        - tokenizer (Tokenizer): The tokenizer object to use.
        - max_length (int): The max length argument to be provided to the tokenizer.
        - mlm_ratio (float): The ratio of tokens to mask.

    Returns:
        - dict: A dictionary containing the following keys:
            + 'input_ids': Torch Tensor of shape (batch_size, seq_length) holding the token ids generated from the provided text batch.
            + 'attention_mask': Torch Tensor of shape (batch_size, seq_length) filled with either 0 or 1 indicating presence or absence of tokens respectively.
            + 'mlm_mask': Torch Tensor of shape (batch_size, seq_length) having values 0 or 1 depending if the token has been chosen for masking in the MLM task.
    """
    tokenized = tokenizer.batch_encode(
        batch, max_length=max_length, return_type="pt", padding="longest"
    )

    # Create the attention mask
    attention_mask = (tokenized != 0).long()

    # Create the MLM mask
    input_lens = attention_mask.sum(dim=-1).tolist()

    mlm_mask = torch.zeros(tokenized.shape, dtype=torch.long)
    for idx, input_len in enumerate(input_lens):
        n_masked_tokens = math.ceil(mlm_ratio * input_len)
        masked_token_idx = torch.randperm(input_len)[:n_masked_tokens]
        mlm_mask[idx, masked_token_idx] = 1

    return {
        "input_ids": tokenized,
        "attention_mask": attention_mask,
        "mlm_mask": mlm_mask,
    }