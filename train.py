import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import mlm_collate_fn
from mlm_model import BERTwithMLMHead
from model import BERTConfig
from tokenizer import Tokenizer, TokenizerConfig


@dataclass
class MLMTrainConfig:
    mlm_ratio: float = 0.2
    label_smoothing: float = 0.01
    lr: float = 1e-4
    epochs: int = 3
    batch_size: int = 1024
    test_ratio: float = 0.25


def accuracy_fn(y_pred, y):
    predicted_token_ids = y_pred.max(dim=-1).indices
    batch_scores = (y == predicted_token_ids).type(torch.FloatTensor)
    return batch_scores.mean()


def train(model, optimizer, criterion, train_dataloader, tokenizer, device="cpu"):
    model.train()

    for batch in tqdm(train_dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        mlm_mask = batch["mlm_mask"].to(device)

        masked_input_ids = torch.where(
            mlm_mask == 0, input_ids, tokenizer.mask_token_id
        )

        optimizer.zero_grad()
        output = model(
            masked_input_ids, attention_mask=attention_mask
        )  # output size: (bs, seq_len, vocab_size)

        num_mask_tokens = int(mlm_mask.sum().item())
        output = output.masked_select(mlm_mask.unsqueeze(-1) == 1).view(
            num_mask_tokens, -1
        )

        labels = torch.masked_select(input_ids, mlm_mask == 1)

        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()


def evaluate(model, criterion, metric_fn, test_dataloader, tokenizer, device="cpu"):
    model.eval()

    losses = []
    scores = []

    for batch in tqdm(test_dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        mlm_mask = batch["mlm_mask"].to(device)

        masked_input_ids = torch.where(
            mlm_mask == 0, input_ids, tokenizer.mask_token_id
        )

        with torch.no_grad():
            output = model(
                masked_input_ids, attention_mask=attention_mask
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


def load_config(config_path, config_class):
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    return config_class(**config_dict)


def save_model(model, save_path: Path, epoch):
    save_path.mkdir(parents=True, exist_ok=True)
    model_path = save_path / f"model_epoch_{epoch}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")


def save_plots(train_losses, train_scores, test_losses, test_scores, save_path, epoch):
    save_path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss - Epoch {epoch}")

    plt.subplot(1, 2, 2)
    plt.plot(train_scores, label="Train Score")
    plt.plot(test_scores, label="Test Score")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.title(f"Score - Epoch {epoch}")

    plt.savefig(save_path / f"plots_epoch_{epoch}.pdf")
    plt.show()
    print(f"Plots saved at {save_path / f'plots_epoch_{epoch}.pdf'}")


def train_and_evaluate(
    model,
    optimizer,
    criterion,
    train_dataloader,
    test_dataloader,
    tokenizer,
    device,
    train_config,
):
    train_losses = []
    train_scores = []
    test_losses = []
    test_scores = []

    for epoch in range(train_config.epochs):
        # Train
        train(model, optimizer, criterion, train_dataloader, tokenizer, device=device)

        # Evaluate
        batch_train_losses, batch_train_scores = evaluate(
            model, criterion, accuracy_fn, train_dataloader, tokenizer, device=device
        )
        batch_test_losses, batch_test_scores = evaluate(
            model, criterion, accuracy_fn, test_dataloader, tokenizer, device=device
        )

        train_loss = mean(batch_train_losses)
        train_score = mean(batch_train_scores)
        test_loss = mean(batch_test_losses)
        test_score = mean(batch_test_scores)

        train_losses.append(train_loss)
        train_scores.append(train_score)
        test_losses.append(test_loss)
        test_scores.append(test_score)

        print(
            f"Epoch {epoch+1}/{train_config.epochs} - Train Loss: {train_loss:.4f}, Train Score: {train_score:.4f}, Test Loss: {test_loss:.4f}, Test Score: {test_score:.4f}"
        )

        # Save model every N epochs
        if (epoch + 1) % args.save_every == 0:
            save_model(model, args.save_path, epoch + 1)
            save_plots(
                train_losses,
                train_scores,
                test_losses,
                test_scores,
                args.save_path,
                epoch + 1,
            )

    # Save model at the end
    save_model(model, args.save_path, train_config.epochs)
    save_plots(
        train_losses,
        train_scores,
        test_losses,
        test_scores,
        args.save_path,
        train_config.epochs,
    )


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="MLM Training")
    parser.add_argument(
        "--model_config_path",
        type=str,
        required=True,
        help="Path to the model configuration file",
    )
    parser.add_argument(
        "--tokenizer_config_path",
        type=str,
        required=True,
        help="Path to the tokenizer configuration file",
    )
    parser.add_argument(
        "--train_config_path",
        type=str,
        required=True,
        help="Path to the training configuration file",
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the jsonlines data file"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--test_ratio", type=float, default=0.25, help="Ratio of test data"
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default="./saved_models",
        help="Path to save the model and plots",
    )
    parser.add_argument(
        "--save_every", type=int, default=1, help="Save model and plots every N epochs"
    )
    args = parser.parse_args()

    # Load configurations
    model_config = load_config(args.model_config_path, BERTConfig)
    tokenizer_config = load_config(args.tokenizer_config_path, TokenizerConfig)
    train_config = load_config(args.train_config_path, MLMTrainConfig)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer(tokenizer_config)
    model = BERTwithMLMHead(model_config).to(device)

    # Data
    with open(args.data_path, "r", encoding="utf-8") as f:
        texts = [json.loads(line) for line in f]

    random.seed(42)
    random.shuffle(texts)

    test_size = int(len(texts) * train_config.test_ratio)
    train_data = texts[test_size:]
    test_data = texts[:test_size]

    print(f"Train size: {len(train_data)}")
    print(f"Test size: {len(test_data)}")

    train_dataloader = DataLoader(
        train_data,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: mlm_collate_fn(
            batch,
            tokenizer=tokenizer,
            max_length=model.config.n_positions,
            mlm_ratio=train_config.mlm_ratio,
        ),
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: mlm_collate_fn(
            batch,
            tokenizer=tokenizer,
            max_length=model.config.n_positions,
            mlm_ratio=train_config.mlm_ratio,
        ),
    )

    # Train loop
    criterion = nn.CrossEntropyLoss(label_smoothing=train_config.label_smoothing)
    adam_optim = torch.optim.Adam(model.parameters(), lr=train_config.lr)

    train_and_evaluate(
        model=model,
        optimizer=adam_optim,
        criterion=criterion,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        tokenizer=tokenizer,
        device=device,
        train_config=train_config,
    )
