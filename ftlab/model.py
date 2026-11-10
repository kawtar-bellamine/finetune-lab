"""A small GPT and a LoRA adapter written from scratch."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 97
    block_size: int = 112
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.n_head = c.n_head
        self.ln1, self.ln2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.proj = nn.Linear(c.d_model, c.d_model)
        self.fc = nn.Linear(c.d_model, 4 * c.d_model)
        self.fc2 = nn.Linear(4 * c.d_model, c.d_model)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))
        return x + self.drop(self.fc2(F.gelu(self.fc(self.ln2(x)))))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.config = c
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.pos = nn.Embedding(c.block_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight                       # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

    @torch.no_grad()
    def generate(self, prompts: list[list[int]], max_new: int = 32, stop: int | None = None) -> list[list[int]]:
        """Greedy decoding, batched by prompt length (no padding, so positions stay exact)."""
        self.eval()
        out: list[list[int] | None] = [None] * len(prompts)
        groups: dict[int, list[int]] = {}
        for i, p in enumerate(prompts):
            groups.setdefault(len(p), []).append(i)
        for _, ids in groups.items():
            x = torch.tensor([prompts[i] for i in ids])
            done = torch.zeros(len(ids), dtype=torch.bool)
            gen = []
            for _ in range(max_new):
                nxt = self(x[:, -self.config.block_size:])[:, -1].argmax(-1)
                gen.append(nxt)
                x = torch.cat([x, nxt[:, None]], 1)
                if stop is not None:
                    done |= nxt == stop
                    if done.all():
                        break
            g = torch.stack(gen, 1).tolist()
            for j, i in enumerate(ids):
                seq = g[j]
                out[i] = seq[: seq.index(stop) + 1] if stop in seq else seq
        return out  # type: ignore[return-value]


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x, with W frozen, A random and B zero so training starts from the base model."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def merged(self) -> nn.Linear:
        """Fold the adapter into a plain Linear: same outputs, zero inference overhead."""
        m = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            m.weight.copy_(self.base.weight + self.scale * self.B @ self.A)
            if self.base.bias is not None:
                m.bias.copy_(self.base.bias)
        return m


TARGETS = ("qkv", "proj", "fc", "fc2")


def apply_lora(model: GPT, r: int, alpha: float | None = None, targets=TARGETS) -> GPT:
    for p in model.parameters():
        p.requires_grad_(False)
    for b in model.blocks:
        for name in targets:
            setattr(b, name, LoRALinear(getattr(b, name), r, alpha or 2 * r))
    return model


def merge_lora(model: GPT) -> GPT:
    for b in model.blocks:
        for name in TARGETS:
            m = getattr(b, name)
            if isinstance(m, LoRALinear):
                setattr(b, name, m.merged())
    return model


def n_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
