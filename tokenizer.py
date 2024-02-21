from dataclasses import dataclass
from typing import List

import torch


@dataclass
class TokenizerConfig:
    vocab_str: str = "ابتةثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئ0123456789.؟ "
    pad_token_id: int = 0
    unk_token_id: int = 1
    mask_token_id: int = 2
    unk_token: str = '$'
    num_special_token_ids: int = 3


class Tokenizer:
    def __init__(self, config: TokenizerConfig):
        self.vocab_str = config.vocab_str
        self.pad_token_id = config.pad_token_id
        self.unk_token_id = config.unk_token_id
        self.mask_token_id = config.mask_token_id
        self.unk_token = config.unk_token
        self.char2id = {
            char: idx + config.num_special_token_ids
            for idx, char in enumerate(self.vocab_str)
        }
        self.id2char = {
            id: char for char, id in self.char2id.items()
        }

    def token2id(self, char: str) -> int:
        return self.char2id.get(char, self.unk_token_id)

    def encode(
        self,
        text: str,
        max_length: int = None,
        padding: bool = False,
        return_type: str = None,
    ) -> List[int] | torch.LongTensor:
        """
        Args:
          padding (bool): Adding padding tokens to `max_length` if True and `max_length` is not None.
        """

        tokenized_text = [self.token2id(char) for char in text]

        if max_length is not None:
            tokenized_text = tokenized_text[:max_length]

            if padding:
                tokenized_text = self.pad(tokenized_text, max_length)

        if return_type == "pt":
            tokenized_text = torch.LongTensor(tokenized_text)

        return tokenized_text
    
    def decode(self, token_ids: List[int] | torch.LongTensor, skip_special_tokens: bool = True) -> str:
        """
        Args:
          token_ids (List[int] | torch.LongTensor): A list of token ids in the shape of (seq_len).
        """

        if isinstance(token_ids, torch.LongTensor):
          if len(token_ids.shape) != 1:
            raise Exception("`decode` only supports tensors of a single dimension.")
          
          token_ids = token_ids.tolist()

        # Remove pad tokens
        token_ids = [token_id for token_id in token_ids if token_id != self.pad_token_id]

        if skip_special_tokens:
            return ''.join([self.id2char[token_id] for token_id in token_ids if token_id in self.id2char])
        
        return ''.join([self.id2char.get(token_id, self.unk_token) for token_id in token_ids])

    def batch_encode(
        self, texts: List[str], max_length: int = None, return_type: str = None
    ) -> List[List[int]] | torch.LongTensor:
        tokenized_texts = [
            self.encode(text, max_length=None, padding=False) for text in texts
        ]

        if max_length is None:
            max_length = max(len(token_ids) for token_ids in tokenized_texts)

        tokenized_texts = [token_ids[:max_length] for token_ids in tokenized_texts]
        tokenized_texts = [
            self.pad(token_ids, max_length) for token_ids in tokenized_texts
        ]

        if return_type == "pt":
            tokenized_texts = torch.LongTensor(tokenized_texts)

        return tokenized_texts

    def pad(self, token_ids: List[int], max_length: int) -> List[int]:
        return token_ids + [self.pad_token_id] * (max_length - len(token_ids))


if __name__ == '__main__':
    config = TokenizerConfig()
    tokenizer = Tokenizer(config)

    tokenized = tokenizer.encode('ابجد')
    assert len(tokenized) == 4

    tokenized = tokenizer.encode('ابجد', max_length=2)
    assert len(tokenized) == 2

    tokenized = tokenizer.encode('ابجد', max_length=10, padding=True)
    assert len(tokenized) == 10

    tokenized = tokenizer.encode('ابجد', max_length=10, padding=True, return_type='pt')
    assert tokenized.shape == torch.Size([10])

    tokenized = tokenizer.batch_encode(['ابجد', 'هوز'], return_type='pt')
    assert tokenized.shape == torch.Size([2, 4])

    tokenized = tokenizer.batch_encode(['ابجد', 'هوز'], max_length=10, return_type='pt')
    assert tokenized.shape == torch.Size([2, 10])