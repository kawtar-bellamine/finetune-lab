"""Pretraining, supervised fine-tuning (full or LoRA), DPO and evaluation."""
from __future__ import annotations

import copy
import math
import time

import torch
import torch.nn.functional as F

from .data import EOS, CharTokenizer, Example, grade
from .model import GPT, n_trainable

TOK = CharTokenizer()


def _cosine(step: int, total: int, warmup: int = 50) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def pretrain(model: GPT, text: str, steps: int, batch: int = 48, lr: float = 3e-3, seed: int = 0,
             log_every: int = 100) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    data = torch.tensor(TOK.encode(text))
    T = model.config.block_size
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    log = []
    model.train()
    for step in range(steps):
        ix = torch.randint(len(data) - T - 1, (batch,), generator=g)
        x = torch.stack([data[i:i + T] for i in ix]); y = torch.stack([data[i + 1:i + T + 1] for i in ix])
        for pg in opt.param_groups:
            pg["lr"] = lr * _cosine(step, steps)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % log_every == 0 or step == steps - 1:
            log.append({"step": step, "loss": loss.item()})
    return log


def text_bpc(model: GPT, text: str) -> float:
    """Bits per character on held-out plain text: the forgetting metric."""
    model.eval()
    data = torch.tensor(TOK.encode(text))
    T = model.config.block_size
    n = (len(data) - 1) // T
    x = data[: n * T].view(n, T); y = data[1: n * T + 1].view(n, T)
    with torch.no_grad():
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
    return loss.item() / math.log(2)


def _pack(pairs: list[tuple[str, str]]):
    """Right-padded batch; the mask selects the answer tokens only."""
    seqs = [TOK.encode(p + a) for p, a in pairs]
    L = max(map(len, seqs))
    x = torch.zeros(len(seqs), L - 1, dtype=torch.long); y = torch.zeros_like(x)
    mask = torch.zeros_like(x, dtype=torch.bool)
    for i, ((p, _), s) in enumerate(zip(pairs, seqs)):
        x[i, :len(s) - 1] = torch.tensor(s[:-1]); y[i, :len(s) - 1] = torch.tensor(s[1:])
        mask[i, len(p) - 1: len(s) - 1] = True
    return x, y, mask


def answer_logp(model: GPT, pairs: list[tuple[str, str]]) -> torch.Tensor:
    x, y, mask = _pack(pairs)
    logp = torch.log_softmax(model(x), -1).gather(-1, y[..., None])[..., 0]
    return (logp * mask).sum(1)


def sft(model: GPT, examples: list[Example], steps: int, batch: int = 32, lr: float = 1e-3, seed: int = 0,
        answers: list[str] | None = None) -> dict:
    """Supervised fine-tuning on the answer tokens. Works for full fine-tuning and LoRA alike."""
    g = torch.Generator().manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    answers = answers or [e.answer for e in examples]
    t0, log = time.time(), []
    model.train()
    for step in range(steps):
        ix = torch.randint(len(examples), (batch,), generator=g).tolist()
        x, y, mask = _pack([(examples[i].prompt, answers[i]) for i in ix])
        for pg in opt.param_groups:
            pg["lr"] = lr * _cosine(step, steps, 30)
        logits = model(x)
        loss = F.cross_entropy(logits[mask], y[mask])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 50 == 0 or step == steps - 1:
            log.append({"step": step, "loss": loss.item()})
    return {"trainable_params": n_trainable(model), "seconds": time.time() - t0, "log": log}


def dpo(policy: GPT, pairs: list[tuple[str, str, str]], steps: int, beta: float = 0.1, batch: int = 16,
        lr: float = 5e-4, seed: int = 0, nll_weight: float = 0.0) -> dict:
    """Direct Preference Optimization (Rafailov et al., 2023) against a frozen copy of the starting model.

    pairs: (prompt, chosen answer, rejected answer).
    nll_weight > 0 adds the negative log-likelihood of the chosen answer per character (as in RPO, Pang et al.,
    2024), which stops DPO from pushing the preferred answer down together with the rejected one.
    """
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(seed)
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    log = []
    policy.train()
    for step in range(steps):
        ix = torch.randint(len(pairs), (batch,), generator=g).tolist()
        ch = [(pairs[i][0], pairs[i][1]) for i in ix]; rj = [(pairs[i][0], pairs[i][2]) for i in ix]
        pc, pr = answer_logp(policy, ch), answer_logp(policy, rj)
        with torch.no_grad():
            rc, rr = answer_logp(ref, ch), answer_logp(ref, rj)
        margin = beta * ((pc - rc) - (pr - rr))
        loss = -F.logsigmoid(margin).mean()
        if nll_weight:
            n_chars = torch.tensor([len(a) for _, a in ch], dtype=torch.float)
            loss = loss + nll_weight * (-pc / n_chars).mean()
        for pg in opt.param_groups:
            pg["lr"] = lr * _cosine(step, steps, 10)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 10 == 0 or step == steps - 1:
            log.append({"step": step, "loss": loss.item(), "reward_accuracy": (margin > 0).float().mean().item(),
                        "chosen_shift": (pc - rc).mean().item(), "rejected_shift": (pr - rr).mean().item()})
    return {"log": log}


def evaluate(model: GPT, examples: list[Example]) -> dict:
    outs = model.generate([TOK.encode(e.prompt) for e in examples], max_new=32, stop=TOK.stoi[EOS])
    grades = [grade(e, TOK.decode(o)) for e, o in zip(examples, outs)]
    keys = grades[0].keys()
    res = {k: sum(gr[k] for gr in grades) / len(grades) for k in keys}
    near = [gr for gr, e in zip(grades, examples) if e.near]
    far = [gr for gr, e in zip(grades, examples) if not e.near]
    res["verdict_near_limit"] = sum(gr["verdict"] for gr in near) / max(1, len(near))
    res["verdict_far_from_limit"] = sum(gr["verdict"] for gr in far) / max(1, len(far))
    res["n"] = len(examples)
    res["samples"] = [{"prompt": e.prompt.strip(), "output": TOK.decode(o).strip(), **gr}
                      for e, o, gr in list(zip(examples, outs, grades))[:8]]
    return res
