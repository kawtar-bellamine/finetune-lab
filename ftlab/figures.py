"""Figures for the README, drawn from results/ only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]   # fixed categorical order
NEUTRAL = "#9a9893"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def _row(summary, name):
    return next(r for r in summary if r["setting"] == name)


def _bars(ax, groups, series, summary, colors):
    """Grouped bars with one-standard-deviation whiskers over the 3 seeds."""
    n = len(series)
    w = 0.8 / n
    for j, (label, setting) in enumerate(series):
        r = _row(summary, setting)
        xs = [i + (j - (n - 1) / 2) * w for i in range(len(groups))]
        vals = [r[k] for _, k in groups]
        sds = [r.get(k + ".sd", 0) for _, k in groups]
        ax.bar(xs, vals, w * 0.92, color=colors[j], label=label, yerr=sds, capsize=2,
               error_kw={"ecolor": INK2, "elinewidth": 0.8})
    ax.set_xticks(range(len(groups)), [g for g, _ in groups])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", visible=False)


def draw(res: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((res / "summary.json").read_text())
    runs = json.loads((res / "all_runs.json").read_text())

    # 1. What fine-tuning learns, and where it breaks
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    groups = [("verdict, far\nfrom the limit", "test.verdict_far_from_limit"),
              ("verdict, just above\nthe limit", "test.verdict_near_limit"),
              ("values copied\nexactly", "test.faithful"),
              ("unseen phrasings and\nequipment: verdict", "ood.verdict"),
              ("unseen equipment:\nid copied", "ood.faithful")]
    series = [("full fine-tuning", "SFT legacy labels, full fine-tuning"),
              ("LoRA r=16", "SFT legacy labels, LoRA r=16"),
              ("LoRA r=4", "SFT legacy labels, LoRA r=4"),
              ("full fine-tuning, correct labels", "SFT correct labels, full fine-tuning (reference)")]
    _bars(ax, groups, series, summary, SERIES[:3] + [NEUTRAL])
    ax.set_title("Supervised fine-tuning on legacy labels: 6,000 examples, mean of 3 seeds")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=9)
    fig.tight_layout(); fig.savefig(out / "sft_methods.png", dpi=160); plt.close(fig)

    # 2. Learning versus forgetting
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    names = ["full fine-tuning", "LoRA r=16", "LoRA r=4"]
    for c, name in zip(SERIES, names):
        rs = [r for r in runs["sft"] if r["method"] == name]
        ax.scatter([r["bpc"] for r in rs], [r["test"]["verdict_far_from_limit"] for r in rs], s=60, color=c,
                   edgecolor=SURFACE, linewidth=1.5, label=f"{name} ({rs[0]['trainable_params']:,} trained weights)")
    b = runs["base"]
    ax.axvline(b["bpc"], color=NEUTRAL, linestyle="--", linewidth=1)
    ax.annotate("base model\n(before fine-tuning)", (b["bpc"], 0.52), xytext=(6, 0), textcoords="offset points",
                color=INK2, fontsize=9)
    ax.set_xlabel("bits per character on held-out plain text (lower = less forgetting)")
    ax.set_ylabel("verdict accuracy, far from the limit")
    ax.set_ylim(0.5, 1.0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("What the fine-tuned model learns, and what it forgets")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout(); fig.savefig(out / "learning_vs_forgetting.png", dpi=160); plt.close(fig)

    # 3. Fixing the legacy bug with 300 expert-reviewed cases
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    groups = [("verdict, just above\nthe limit", "test.verdict_near_limit"),
              ("verdict, far\nfrom the limit", "test.verdict_far_from_limit"),
              ("exact answer", "test.exact"),
              ("unseen phrasings:\nverdict", "ood.verdict")]
    series = [("SFT on legacy labels", "SFT legacy labels, full fine-tuning"),
              ("+ DPO", "fix: DPO"),
              ("+ DPO + NLL", "fix: DPO + NLL"),
              ("+ SFT on the corrected cases", "fix: SFT on the corrected cases"),
              ("reference: correct labels", "SFT correct labels, full fine-tuning (reference)")]
    groups = [("verdict, just above\nthe limit", "test.verdict_near_limit"),
              ("verdict, far\nfrom the limit", "test.verdict_far_from_limit"),
              ("values copied\nexactly", "test.faithful"),
              ("exact answer", "test.exact")]
    _bars(ax, groups, series, summary, [SERIES[0], SERIES[1], SERIES[3], SERIES[2], NEUTRAL])
    ax.set_title("Correcting a systematic labelling error with 300 expert-reviewed cases, mean of 3 seeds")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8.5)
    fig.tight_layout(); fig.savefig(out / "fixing_the_bug.png", dpi=160); plt.close(fig)

    # 4. DPO training dynamics, seed 0: why the NLL term and a small learning rate matter
    variants = [("DPO, lr 1e-5", next(r for r in runs["fix"] if r["method"] == "DPO" and r["seed"] == 0), SERIES[1]),
                ("DPO + NLL, lr 1e-5", next(r for r in runs["fix"] if r["method"] == "DPO + NLL" and r["seed"] == 0),
                 SERIES[3]),
                ("DPO, lr 1e-4", runs["dpo_high_lr"], SERIES[5])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for label, run, c in variants:
        st = [x["step"] for x in run["log"]]
        a1.plot(st, [x["chosen_shift"] for x in run["log"]], color=c, linewidth=2, label=label)
        a2.plot(st, [x["rejected_shift"] for x in run["log"]], color=c, linewidth=2, label=label)
    for a, t in ((a1, "Preferred (correct) answer"), (a2, "Rejected (wrong) answer")):
        a.axhline(0, color=NEUTRAL, linewidth=1)
        a.set_title(t); a.set_xlabel("DPO step")
    a1.set_ylabel("log-probability change vs start (nats)")
    a2.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "dpo_dynamics.png", dpi=160); plt.close(fig)
