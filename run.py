"""Run every experiment, then write results/*.json and docs/figures/*.png.

    python run.py            # about 45 minutes on a 2-core CPU, every run is cached in .cache/
    python run.py --figures  # only redraw the figures from results/
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import time
from pathlib import Path

import torch

from ftlab.data import answer_for, make_examples, pretrain_corpus
from ftlab.model import GPT, GPTConfig, apply_lora, merge_lora
from ftlab.train import TOK, dpo, evaluate, pretrain, sft, text_bpc

ROOT = Path(__file__).parent
CACHE, RES = ROOT / ".cache", ROOT / "results"
SEEDS = (0, 1, 2)
SFT_STEPS, N_TRAIN = 1000, 6000
METHODS = {"full fine-tuning": dict(lora=None, lr=5e-4),
           "LoRA r=16": dict(lora=16, lr=2e-3),
           "LoRA r=4": dict(lora=4, lr=2e-3)}

TEST = make_examples(400, 1001)                 # same templates and equipment as training, unseen readings
OOD = make_examples(300, 1002, ood=True)        # unseen phrasings and equipment types
HELDOUT_TEXT = pretrain_corpus(400, 1003)       # plain text, to measure forgetting


def torch_seed(s: int):
    torch.manual_seed(s); random.seed(s)


def base_model() -> GPT:
    path = CACHE / "base.pt"
    m = GPT(GPTConfig(vocab_size=TOK.vocab_size))
    if path.exists():
        m.load_state_dict(torch.load(path))
        return m
    torch_seed(0)
    t0 = time.time()
    log = pretrain(m, pretrain_corpus(40_000, 0), steps=1200)
    torch.save(m.state_dict(), path)
    json.dump({"log": log, "seconds": time.time() - t0, "params": sum(p.numel() for p in m.parameters())},
              open(RES / "pretrain.json", "w"), indent=1)
    return m


def measure(m: GPT) -> dict:
    return {"test": evaluate(m, TEST), "ood": evaluate(m, OOD), "bpc": text_bpc(m, HELDOUT_TEXT)}


def cached(name: str, fn):
    path = CACHE / "runs" / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    r = fn()
    path.write_text(json.dumps(r, indent=1))
    print(f"{name}: test exact {r['test']['exact']:.3f}, near {r['test']['verdict_near_limit']:.3f}, "
          f"bpc {r['bpc']:.2f}", flush=True)
    return r


def sft_run(base: GPT, method: str, seed: int, deadband: int, n: int = N_TRAIN) -> tuple[dict, GPT]:
    """Fine-tune on legacy labels (deadband=3) or on corrected labels (deadband=0)."""
    cfg = METHODS[method]
    tag = f"sft_{method.replace(' ', '_').replace('=', '')}_db{deadband}_n{n}_s{seed}"
    ckpt = CACHE / f"{tag}.pt"
    m = copy.deepcopy(base)
    if cfg["lora"]:
        apply_lora(m, cfg["lora"])

    def go():
        torch_seed(seed)
        info = sft(m, make_examples(n, 10 + seed, deadband=deadband), SFT_STEPS, lr=cfg["lr"], seed=seed)
        torch.save(m.state_dict(), ckpt)
        return {**measure(m), **info, "method": method, "seed": seed, "deadband": deadband, "n_train": n}

    r = cached(tag, go)
    if ckpt.exists():
        m.load_state_dict(torch.load(ckpt))
    if cfg["lora"]:
        merge_lora(m)
    return r, m


def expert_cases(n: int, seed: int):
    """A domain expert reviews n cases drawn like real traffic (a quarter of them just above the limit).

    SFT uses the corrected answers; DPO uses (correct answer, wrong verdict) pairs. Just above the limit the wrong
    verdict is exactly the legacy label, so both methods get the same information.
    """
    ex = make_examples(n, 500 + seed)

    def wrong(e):
        if e.value > e.limit:
            return f"OK {e.eq_id} {e.value} {e.unit} <= {e.limit} {e.unit}\n"
        return f"ALARM {e.eq_id} {e.value} {e.unit} > {e.limit} {e.unit}\n"
    return ex, [(e.prompt, e.answer, wrong(e)) for e in ex]


FIXES = {  # settings chosen on a separate dev set (scripts/tune_fix.py), never on the test set
    "DPO": dict(kind="dpo", beta=0.1, lr=1e-5, nll_weight=0.0),
    "DPO + NLL": dict(kind="dpo", beta=0.1, lr=1e-5, nll_weight=1.0),
    "SFT on the corrected cases": dict(kind="sft", lr=3e-5),
}


def main():
    CACHE.mkdir(exist_ok=True); (CACHE / "runs").mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    base = base_model()
    out = {"base": cached("base", lambda: measure(base))}

    # 1. Full fine-tuning against LoRA, on the legacy labels
    sft_models = {}
    out["sft"] = []
    for method in METHODS:
        for s in SEEDS:
            r, m = sft_run(base, method, s, deadband=3)
            out["sft"].append(r)
            if method == "full fine-tuning":
                sft_models[s] = m
    # reference: the same fine-tuning with correct labels everywhere (what DPO tries to recover)
    out["oracle"] = [sft_run(base, "full fine-tuning", s, deadband=0)[0] for s in SEEDS]

    # 2. Fixing the legacy bug with 300 expert-reviewed cases: DPO against plain SFT on the same cases
    out["fix"] = []
    for s in SEEDS:
        ex, pairs = expert_cases(300, s)
        for name, cfg in FIXES.items():
            def go(cfg=cfg, s=s, ex=ex, pairs=pairs):
                m = copy.deepcopy(sft_models[s]); torch_seed(s)
                if cfg["kind"] == "dpo":
                    info = dpo(m, pairs, steps=150, beta=cfg["beta"], lr=cfg["lr"], nll_weight=cfg["nll_weight"],
                               seed=s)
                else:
                    info = sft(m, ex, steps=150, lr=cfg["lr"], batch=16, seed=s)
                return {**measure(m), **info}
            tag = name.replace(" ", "_").replace("+", "plus")
            out["fix"].append({"method": name, "seed": s, **cached(f"fix_{tag}_s{s}", go)})

    # what happens with a learning rate 10 times higher: both answers are pushed down (seed 0 only)
    def go_collapse():
        ex, pairs = expert_cases(300, 0)
        m = copy.deepcopy(sft_models[0]); torch_seed(0)
        info = dpo(m, pairs, steps=150, beta=0.1, lr=1e-4, seed=0)
        return {**measure(m), **info}
    out["dpo_high_lr"] = cached("dpo_high_lr_s0", go_collapse)

    (RES / "all_runs.json").write_text(json.dumps(out, indent=1))
    summary = summarize(out)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    for row in summary:
        print(row)


KEYS = [("test", "exact"), ("test", "verdict_far_from_limit"), ("test", "verdict_near_limit"), ("test", "faithful"),
        ("ood", "verdict"), ("ood", "faithful")]


def summarize(out: dict) -> list[dict]:
    groups: dict[str, list[dict]] = {"base model (no fine-tuning)": [out["base"]]}
    for r in out["sft"]:
        groups.setdefault(f"SFT legacy labels, {r['method']}", []).append(r)
    groups["SFT correct labels, full fine-tuning (reference)"] = out["oracle"]
    for r in out["fix"]:
        groups.setdefault(f"fix: {r['method']}", []).append(r)
    groups["fix: DPO, learning rate x10 (1 seed)"] = [out["dpo_high_lr"]]
    rows = []
    for name, runs in groups.items():
        row = {"setting": name, "runs": len(runs)}
        for split, k in KEYS:
            vals = [r[split][k] for r in runs]
            row[f"{split}.{k}"] = round(statistics.mean(vals), 4)
            row[f"{split}.{k}.sd"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
        vals = [r["bpc"] for r in runs]
        row["bpc"] = round(statistics.mean(vals), 3)
        row["bpc.sd"] = round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0
        if "trainable_params" in runs[0]:
            row["trainable_params"] = runs[0]["trainable_params"]
        rows.append(row)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    if not a.figures:
        main()
    from ftlab.figures import draw
    draw(RES, ROOT / "docs" / "figures")
