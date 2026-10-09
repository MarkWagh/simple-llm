#!/usr/bin/env python3
"""
tiny_llm.py

A minimal, educational causal language model trained from scratch on a tiny
text corpus (~1000 words is enough to demonstrate the mechanics, but NOT enough
to produce a useful general-purpose LLM).

Important:
- ChatGPT itself is not reproducible from a single public "ChatGPT paper".
- This implements the core autoregressive Transformer language-modeling idea:
  tokenize text -> predict the next token -> causal self-attention -> minimize
  cross-entropy loss.
- It does NOT reproduce ChatGPT/InstructGPT training, RLHF, or the proprietary
  tokenizer/model/data.

Requirements:
    pip install torch

Usage:
    python tiny_llm.py --text my_text.txt
    python tiny_llm.py --text my_text.txt --steps 2000 --prompt "Once upon a time"

If no --text file is supplied, the built-in example corpus is used.
"""

import argparse
import math
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1. Very small word tokenizer
# -----------------------------

def tokenize(text):
    # Keeps words and punctuation as separate tokens.
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def detokenize(tokens):
    text = " ".join(tokens)
    # Basic punctuation cleanup.
    text = re.sub(r"\s+([,.!?;:%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text


# -----------------------------
# 2. Tiny causal Transformer
# -----------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, block_size):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Prevent tokens from attending to future tokens.
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout, block_size):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model, n_heads, dropout, block_size
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm Transformer block.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=64,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_model, n_heads, dropout, block_size
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: input and output token embeddings share weights.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError("Sequence is longer than block_size.")

        positions = torch.arange(T, device=idx.device)

        x = (
            self.token_embedding(idx)
            + self.position_embedding(positions)[None, :, :]
        )

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss


# -----------------------------
# 3. Training data
# -----------------------------

DEFAULT_TEXT = """
Language models learn statistical patterns in sequences of tokens.
A small language model can learn simple relationships between words when
it sees enough examples. The model reads a context and predicts the next
token. During training, the correct next token is known, so the prediction
can be compared with the target using cross entropy loss.

Transformers use attention to allow each position to examine earlier
positions in the sequence. Causal attention prevents a position from looking
at future tokens. This makes the architecture appropriate for autoregressive
language modeling.

This example is intentionally small. A tiny dataset and a tiny model are
useful for learning how training works, but they cannot produce the broad
knowledge or reasoning ability of a modern large language model. Real models
use enormous datasets, many layers, large hidden dimensions, sophisticated
tokenizers, distributed training, careful optimization, and extensive
evaluation.

The goal of this program is educational. It demonstrates tokenization,
embeddings, positional information, masked self attention, feed forward
layers, residual connections, normalization, next token prediction, and
sampling from the trained model.
"""


def build_dataset(text):
    tokens = tokenize(text)

    # Add special tokens.
    vocab = ["<PAD>", "<UNK>"] + sorted(set(tokens))
    stoi = {tok: i for i, tok in enumerate(vocab)}
    itos = {i: tok for tok, i in stoi.items()}

    ids = torch.tensor(
        [stoi.get(tok, stoi["<UNK>"]) for tok in tokens],
        dtype=torch.long,
    )

    return ids, stoi, itos


def make_batch(data, block_size, batch_size, device):
    if len(data) <= block_size:
        raise ValueError(
            "The text is too short. Provide more text or reduce block_size."
        )

    starts = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,),
    )

    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])

    return x.to(device), y.to(device)


# -----------------------------
# 4. Autoregressive generation
# -----------------------------

@torch.no_grad()
def generate(model, prompt, stoi, itos, device,
             max_new_tokens=50, temperature=0.8, top_k=20):

    tokens = tokenize(prompt)

    ids = [
        stoi.get(tok, stoi["<UNK>"])
        for tok in tokens
    ]

    if not ids:
        ids = [stoi["<UNK>"]]

    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = idx[:, -model.block_size:]

        logits, _ = model(context)
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        if top_k is not None:
            values, _ = torch.topk(
                logits,
                min(top_k, logits.size(-1))
            )
            cutoff = values[:, [-1]]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))

        probabilities = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probabilities, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)

    generated = [
        itos[int(i)]
        for i in idx[0].tolist()
    ]

    return detokenize(generated)


# -----------------------------
# 5. Main training program
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Path to a plain-text training file.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of optimizer steps.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="language models",
    )

    args = parser.parse_args()

    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if args.text is None:
        print("No --text supplied; using the built-in example corpus.")
        text = DEFAULT_TEXT
    else:
        text = Path(args.text).read_text(encoding="utf-8")

    ids, stoi, itos = build_dataset(text)

    print("Training tokens:", len(ids))
    print("Vocabulary size:", len(stoi))

    model = TinyGPT(
        vocab_size=len(stoi),
        block_size=args.block_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    model.train()

    for step in range(1, args.steps + 1):
        x, y = make_batch(
            ids,
            args.block_size,
            args.batch_size,
            device,
        )

        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping improves stability for this tiny example.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        if step == 1 or step % 100 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f}")

    model.eval()

    print("\n--- Generated text ---")
    print(
        generate(
            model,
            args.prompt,
            stoi,
            itos,
            device,
            max_new_tokens=60,
        )
    )


if __name__ == "__main__":
    main()
