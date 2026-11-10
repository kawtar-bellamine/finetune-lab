"""Fast tests: data, grading, LoRA algebra, losses. No full training run."""
from __future__ import annotations

import copy

import torch

from ftlab.data import CharTokenizer, answer_for, grade, make_examples, pretrain_corpus
from ftlab.model import GPT, GPTConfig, LoRALinear, apply_lora, merge_lora, n_trainable
from ftlab.train import answer_logp, dpo, evaluate, sft

TOK = CharTokenizer()
SMALL = GPTConfig(vocab_size=TOK.vocab_size, n_layer=1, n_head=2, d_model=32)


def test_tokenizer_round_trip_and_examples_fit_the_context():
    ex = make_examples(200, 0) + make_examples(200, 1, ood=True)
    for e in ex:
        assert TOK.decode(TOK.encode(e.prompt + e.answer)) == e.prompt + e.answer
        assert len(e.prompt + e.answer) <= GPTConfig().block_size


def test_legacy_rule_differs_only_just_above_the_limit():
    assert answer_for("P-01", 81, 80, "C") == "ALARM P-01 81 C > 80 C\n"
    assert answer_for("P-01", 81, 80, "C", deadband=3).startswith("OK")
    assert answer_for("P-01", 84, 80, "C", deadband=3).startswith("ALARM")
    for e in make_examples(500, 2):
        legacy = answer_for(e.eq_id, e.value, e.limit, e.unit, deadband=3)
        assert (legacy != e.answer) == e.near


def test_splits_do_not_share_equipment_or_phrasing():
    train, ood = make_examples(300, 0), make_examples(300, 1, ood=True)
    assert not {e.eq_id.split("-")[0] for e in train} & {e.eq_id.split("-")[0] for e in ood}
    assert "conveyor" in pretrain_corpus(2000, 0)            # known from pretraining, absent from fine-tuning


def test_grader_catches_wrong_value_false_statement_and_bad_format():
    e = make_examples(1, 3)[0]
    assert grade(e, e.answer)["exact"]
    wrong_value = e.answer.replace(f" {e.value} ", f" {e.value + 1} ", 1)
    assert not grade(e, wrong_value)["faithful"]
    flipped = ("OK" if e.answer.startswith("ALARM") else "ALARM") + e.answer.split(" ", 1)[1]
    assert not grade(e, flipped)["true_statement"]
    assert not grade(e, "I think it is fine")["well_formed"]


def test_lora_starts_as_the_base_model_and_merges_exactly():
    torch.manual_seed(0)
    base = GPT(SMALL)
    x = torch.randint(1, TOK.vocab_size, (2, 20))
    m = apply_lora(copy.deepcopy(base), r=4)
    assert torch.allclose(m(x), base(x), atol=1e-6)                 # B = 0 at initialisation
    for mod in m.modules():
        if isinstance(mod, LoRALinear):
            torch.nn.init.normal_(mod.B, std=0.1)
    before = m(x)
    assert not torch.allclose(before, base(x), atol=1e-4)
    assert torch.allclose(merge_lora(m)(x), before, atol=1e-5)


def test_lora_trains_only_the_adapters():
    m = apply_lora(GPT(SMALL), r=2)
    d, f = SMALL.d_model, 4 * SMALL.d_model
    expected = SMALL.n_layer * 2 * ((d + 3 * d) + (d + d) + (d + f) + (f + d))
    assert n_trainable(m) == expected


def test_sft_reduces_the_answer_loss():
    torch.manual_seed(0)
    m = GPT(SMALL)
    ex = make_examples(64, 4)
    r = sft(m, ex, steps=60, batch=16, lr=3e-3)
    assert r["log"][-1]["loss"] < r["log"][0]["loss"]


def test_dpo_raises_the_preferred_answer_relative_to_the_rejected_one():
    torch.manual_seed(0)
    m = GPT(SMALL)
    ex = make_examples(32, 5, near_frac=1.0)
    pairs = [(e.prompt, e.answer, answer_for(e.eq_id, e.value, e.limit, e.unit, deadband=3)) for e in ex]
    ref = copy.deepcopy(m)

    def gap(model):
        with torch.no_grad():
            return (answer_logp(model, [(p, c) for p, c, _ in pairs])
                    - answer_logp(model, [(p, r) for p, _, r in pairs])).mean().item()

    before = gap(ref)
    log = dpo(m, pairs, steps=40, beta=0.5, batch=8, lr=3e-3)["log"]
    assert gap(m) > before
    assert log[-1]["reward_accuracy"] >= 0.5


def test_evaluate_reports_every_metric():
    r = evaluate(GPT(SMALL), make_examples(12, 6))
    for k in ("exact", "verdict", "faithful", "true_statement", "verdict_near_limit", "verdict_far_from_limit"):
        assert 0.0 <= r[k] <= 1.0
