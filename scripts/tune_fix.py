"""Choose the settings of the fix stage on a dev set (seed-0 SFT model), never on the test set.

    python scripts/tune_fix.py            # after `python run.py` has trained the SFT models
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ftlab.data import make_examples  # noqa: E402
from ftlab.model import GPT, GPTConfig  # noqa: E402
from ftlab.train import TOK, dpo, evaluate, sft  # noqa: E402
from run import CACHE, expert_cases  # noqa: E402

GRID = [("dpo", 0.1, 1e-4, 0.0), ("dpo", 0.1, 3e-5, 1.0), ("dpo", 0.1, 1e-5, 0.0), ("dpo", 0.1, 1e-5, 1.0),
        ("dpo", 0.5, 1e-5, 1.0), ("dpo", 0.1, 5e-6, 1.0), ("sft", None, 1e-4, None), ("sft", None, 3e-5, None)]


def main():
    base = GPT(GPTConfig(vocab_size=TOK.vocab_size))
    base.load_state_dict(torch.load(CACHE / "sft_full_fine-tuning_db3_n6000_s0.pt"))
    dev = make_examples(300, 2001)
    ex, pairs = expert_cases(300, 0)
    for kind, beta, lr, nll in GRID:
        m = copy.deepcopy(base); torch.manual_seed(0)
        if kind == "dpo":
            dpo(m, pairs, 150, beta=beta, lr=lr, nll_weight=nll)
        else:
            sft(m, ex, 150, batch=16, lr=lr)
        e = evaluate(m, dev)
        print(kind, beta, lr, nll, {k: round(e[k], 3) for k in ("exact", "verdict_near_limit",
                                                                 "verdict_far_from_limit", "faithful")}, flush=True)


if __name__ == "__main__":
    main()
