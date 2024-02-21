import torch


def train(model, tokenizer, criterion, train_dataloader):
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