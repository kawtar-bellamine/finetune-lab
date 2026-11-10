"""Synthetic but checkable data: industrial sensor triage.

A technician asks whether a reading is within its limit; the model must answer in a strict format,
copying the equipment id, the reading and the limit, and stating the right verdict:

    Q: Pump P-12 temperature is 87 C, limit 80 C. Status?
    A: ALARM P-12 87 C > 80 C

Every answer can be checked by a parser, so accuracy, hallucinated values and false statements are
measured exactly, without a judge model.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

EQUIP_TRAIN = {"pump": "P", "motor": "M", "compressor": "C", "fan": "F", "gearbox": "G"}
EQUIP_OOD = {"conveyor": "CV", "turbine": "T"}            # seen in pretraining, never in fine-tuning
QUANT = {  # name: unit, reading range, limit range
    "temperature": ("C", (20, 99), (45, 90)),
    "vibration": ("mm/s", (1, 19), (4, 12)),
    "pressure": ("bar", (1, 16), (4, 12)),
    "current": ("A", (5, 60), (15, 50)),
}
TEMPLATES_TRAIN = [
    "{Eq} {id} {q} is {v} {u}, limit {l} {u}. Status?",
    "Reading for {eq} {id}: {q} {v} {u} (max {l} {u}). Check it.",
    "{id} ({eq}) {q} = {v} {u}; alarm above {l} {u}. Status?",
    "Is {eq} {id} ok? {q} {v} {u}, threshold {l} {u}.",
    "{Q} on {eq} {id}: {v} {u} vs limit {l} {u}.",
    "Check {eq} {id}: limit {l} {u}, measured {q} {v} {u}.",
]
TEMPLATES_OOD = [
    "Log: {eq} {id} shows {q} at {v} {u}, max {l} {u}. Verdict?",
    "Limit {l} {u}. {Eq} {id} {q} now {v} {u}. Status?",
]
PRETRAIN_TEMPLATES = [
    "The {eq} {id} ran at {v} {u} of {q} during the night shift.",
    "{Q} of {eq} {id} rose from {l} {u} to {v} {u} after the restart.",
    "Maintenance replaced the bearing on {eq} {id} and logged {q} at {v} {u}.",
    "{v} is {cmp} {l}.",
    "On {eq} {id}, the {q} limit is {l} {u}.",
    "The operator noted that {eq} {id} was {state} with {q} at {v} {u}.",
    "A reading of {v} {u} is {rel} a limit of {l} {u}.",
    "Work order for {eq} {id}: inspect {id}, then log {q} of {id} ({v} {u}).",
    "Shift report: {eq} {id} read {v} {u}, limit {l} {u}, so {id} is {state}.",
]
PROMPT, ANSWER = "Q: ", "\nA: "
EOS = "\n"


@dataclass(frozen=True)
class Example:
    prompt: str          # "Q: ...\nA: "
    answer: str          # "ALARM P-12 87 C > 80 C\n"
    eq_id: str
    value: int
    limit: int
    unit: str
    near: bool           # reading within 3 units above the limit (where the legacy rule is wrong)


def _cmp(v: int, l: int) -> str:
    return "greater than" if v > l else ("equal to" if v == l else "less than")


def _sample(rng: random.Random, equip: dict, near_frac: float = 0.25):
    eq = rng.choice(list(equip))
    eq_id = f"{equip[eq]}-{rng.randint(1, 99):02d}"
    q = rng.choice(list(QUANT))
    unit, (vlo, vhi), (llo, lhi) = QUANT[q]
    l = rng.randint(llo, lhi)
    if rng.random() < near_frac:                   # oversample the hard, safety-relevant zone
        v = l + rng.randint(1, 3)
    else:
        v = rng.randint(vlo, vhi)
    return eq, eq_id, q, unit, v, l


def answer_for(eq_id: str, v: int, l: int, unit: str, deadband: int = 0) -> str:
    """Correct answer (deadband=0) or the legacy rule's answer (deadband=3, alarms only above limit+3)."""
    if v > l + deadband:
        return f"ALARM {eq_id} {v} {unit} > {l} {unit}{EOS}"
    return f"OK {eq_id} {v} {unit} <= {l} {unit}{EOS}"


def make_examples(n: int, seed: int, ood: bool = False, deadband: int = 0, near_frac: float = 0.25) -> list[Example]:
    rng = random.Random(seed)
    equip = EQUIP_OOD if ood else EQUIP_TRAIN
    templates = TEMPLATES_OOD if ood else TEMPLATES_TRAIN
    out = []
    for _ in range(n):
        eq, eq_id, q, unit, v, l = _sample(rng, equip, near_frac)
        t = rng.choice(templates)
        text = t.format(eq=eq, Eq=eq.capitalize(), id=eq_id, q=q, Q=q.capitalize(), v=v, l=l, u=unit)
        out.append(Example(PROMPT + text + ANSWER, answer_for(eq_id, v, l, unit, deadband), eq_id, v, l, unit,
                           l < v <= l + 3))
    return out


def pretrain_corpus(n_lines: int, seed: int) -> str:
    """Plain technical text: equipment, units and number comparisons, but never the Q/A format."""
    rng = random.Random(seed)
    equip = {**EQUIP_TRAIN, **EQUIP_OOD}
    lines = []
    for _ in range(n_lines):
        eq, eq_id, q, unit, v, l = _sample(rng, equip, 0.2)
        t = rng.choice(PRETRAIN_TEMPLATES)
        lines.append(t.format(eq=eq, Eq=eq.capitalize(), id=eq_id, q=q, Q=q.capitalize(), v=v, l=l, u=unit,
                              cmp=_cmp(v, l), state="in alarm" if v > l else "normal",
                              rel="above" if v > l else "within"))
    return "\n".join(lines) + "\n"


ANS_RE = re.compile(r"^(ALARM|OK) (\S+) (\d+) (\S+) (>|<=) (\d+) (\S+)$")


def grade(ex: Example, output: str) -> dict:
    """Exact checks on one generated answer."""
    out = output.split(EOS)[0].strip()
    m = ANS_RE.match(out)
    truth = "ALARM" if ex.value > ex.limit else "OK"
    if not m:
        return {"exact": False, "verdict": False, "well_formed": False, "faithful": False, "true_statement": False}
    verdict, eq_id, v, u1, rel, l, u2 = m.groups()
    v, l = int(v), int(l)
    faithful = eq_id == ex.eq_id and v == ex.value and l == ex.limit and u1 == u2 == ex.unit
    true_statement = (rel == ">" and v > l) or (rel == "<=" and v <= l)
    consistent = (verdict == "ALARM") == (rel == ">")
    return {"exact": out + EOS == answer_for(ex.eq_id, ex.value, ex.limit, ex.unit),
            "verdict": verdict == truth, "well_formed": consistent, "faithful": faithful,
            "true_statement": true_statement and consistent}


class CharTokenizer:
    """Character-level vocabulary: small, transparent, no download needed."""

    def __init__(self):
        chars = "\n" + "".join(chr(i) for i in range(32, 127))
        self.itos = ["<pad>"] + list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids if i > 0)
