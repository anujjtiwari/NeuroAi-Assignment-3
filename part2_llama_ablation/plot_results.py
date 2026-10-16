"""
Turns ../results/part2/results.json into figures (../report/figures) and
summary tables (../results/part2/*.csv).

"Retained score" puts all four tasks on one scale (1 = intact model, 0 = chance):
  accuracy tasks: (acc - chance) / (acc_base - chance)
  perplexity:     (log V - log ppl) / (log V - log ppl_base), V = vocabulary size
                  (a uniform guesser has ppl = V)
"""
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "part2")
FIG = os.path.join(HERE, "..", "report", "figures")
os.makedirs(FIG, exist_ok=True)

VOCAB = 32000
N_LAYERS = 22
TASKS = ["ppl", "blimp", "sst", "facts"]
LABEL = {"ppl": "Next-word prediction (WikiText-2 PPL)", "blimp": "Syntax (BLiMP acc.)",
         "sst": "Sentiment (SST-2 4-shot acc.)", "facts": "Factual completion (top-1 acc.)"}
SHORT = {"ppl": "Next-word (PPL)", "blimp": "Syntax (BLiMP)", "sst": "Sentiment (SST-2)", "facts": "Facts"}
CHANCE = {"blimp": 0.5, "sst": 0.5, "facts": 0.0}
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, MUTED = "#0b0b0b", "#8a8984"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2})

with open(os.path.join(RES, "results.json")) as f:
    R = json.load(f)
BASE = R["baseline"]


def retained(r, task):
    if task == "ppl":
        return (math.log(VOCAB) - math.log(r["ppl"])) / (math.log(VOCAB) - math.log(BASE["ppl"]))
    return (r[task] - CHANCE[task]) / (BASE[task] - CHANCE[task])


def ref_lines(ax, task):
    ax.axhline(BASE[task], color=INK, ls="--", lw=0.9)
    if task != "ppl":
        ax.axhline(CHANCE[task], color=MUTED, ls=":", lw=0.9)


# ------------------------------------------------------------ E1 layer sweep
def fig_layer_sweep():
    layers = np.arange(N_LAYERS)
    kinds = [("skip", "skip whole layer"), ("attn", "zero attention block"), ("mlp", "zero MLP block")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    for ax, task in zip(axes, TASKS):
        for (k, lab), col in zip(kinds, C):
            y = [R[f"E1/{k}/{l}"][task] for l in layers]
            ax.plot(layers, y, "-o", ms=3.5, color=col, label=lab, lw=1.6)
        ref_lines(ax, task)
        if task == "ppl":
            ax.set_yscale("log")
        ax.set_title(LABEL[task], fontsize=9.5)
        ax.set_xlabel("layer ablated")
        ax.set_xticks(range(0, N_LAYERS, 3))
    axes[0].legend(fontsize=8, loc="upper right")
    fig.text(0.5, -0.01, "dashed = intact model, dotted = chance", ha="center", fontsize=8, color="#52514e")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p2_layer_sweep.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for l in layers:
        row = {"layer": l}
        for k, _ in kinds:
            for t in TASKS:
                row[f"{k}_{t}"] = R[f"E1/{k}/{l}"][t]
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "E1_layer_sweep.csv"), index=False)

    # damage heatmap: 1 - retained score, per task x layer, for each component type
    fig, axes = plt.subplots(3, 1, figsize=(10, 5.6), sharex=True)
    for ax, (k, lab) in zip(axes, kinds):
        M = np.array([[1 - retained(R[f"E1/{k}/{l}"], t) for l in layers] for t in TASKS])
        im = ax.imshow(np.clip(M, 0, 1), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(4), [SHORT[t] for t in TASKS])
        ax.set_title(lab, fontsize=9.5, loc="left")
        ax.grid(False)
        for i in range(4):
            for j in range(N_LAYERS):
                if M[i, j] >= 0.15:
                    ax.text(j, i, f"{min(M[i, j], 1):.1f}"[1:] if M[i, j] < 1 else "1", ha="center",
                            va="center", fontsize=6.5, color="white" if M[i, j] > 0.55 else INK)
    axes[-1].set_xticks(range(N_LAYERS))
    axes[-1].set_xlabel("layer ablated")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("performance lost (1 - retained score)")
    fig.savefig(os.path.join(FIG, "p2_damage_heatmap.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------ E2 head pruning
def fig_head_pruning():
    fracs = sorted({float(k.split("/")[2]) for k in R if k.startswith("E2/")})
    orders = [("least_important", "least important first"), ("random", "random (mean of 2, range shaded)"),
              ("most_important", "most important first")]
    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    for ax, task in zip(axes, TASKS):
        for (o, lab), col in zip(orders, C):
            if o == "random":
                Y = np.array([[R[f"E2/random{s}/{f}"][task] for f in fracs] for s in range(2)])
                y = Y.mean(0)
                ax.fill_between(np.array(fracs) * 100, Y.min(0), Y.max(0), color=col, alpha=0.18, lw=0)
            else:
                y = [R[f"E2/{o}/{f}"][task] for f in fracs]
            ax.plot(np.array(fracs) * 100, y, "-o", ms=3.5, color=col, label=lab, lw=1.6)
            for f, v in zip(fracs, y):
                rows.append({"order": o, "fraction_pruned": f, "task": task, "value": float(v)})
        ref_lines(ax, task)
        if task == "ppl":
            ax.set_yscale("log")
        ax.set_title(LABEL[task], fontsize=9.5)
        ax.set_xlabel("% of attention heads removed")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p2_head_pruning.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).pivot_table(index=["order", "fraction_pruned"], columns="task", values="value") \
        .reset_index().to_csv(os.path.join(RES, "E2_head_pruning.csv"), index=False)

    imp = torch.load(os.path.join(RES, "head_importance.pt")).numpy()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    im = ax.imshow(imp.T, cmap="Blues", aspect="auto", origin="lower")
    ax.set_xlabel("layer")
    ax.set_ylabel("head")
    ax.set_xticks(range(N_LAYERS))
    ax.grid(False)
    fig.colorbar(im, ax=ax, pad=0.01).set_label("|dL / d mask|")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p2_head_importance.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    return imp


# ------------------------------------------------------------ E3 layer removal
def fig_layer_removal():
    ks = sorted({int(k.split("/")[2]) for k in R if k.startswith("E3/")})
    orders = [("least_important", "least important first"), ("random", "random (mean of 2, range shaded)"),
              ("top_down", "top-down (last layers first)"), ("bottom_up", "bottom-up (first layers first)")]
    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    for ax, task in zip(axes, TASKS):
        for (o, lab), col in zip(orders, C):
            if o == "random":
                Y = np.array([[R[f"E3/random{s}/{k}"][task] for k in ks] for s in range(2)])
                y = Y.mean(0)
                ax.fill_between(ks, Y.min(0), Y.max(0), color=col, alpha=0.18, lw=0)
            else:
                y = [R[f"E3/{o}/{k}"][task] for k in ks]
            ax.plot(ks, y, "-o", ms=3.5, color=col, label=lab, lw=1.6)
            for k, v in zip(ks, y):
                rows.append({"order": o, "layers_removed": k, "task": task, "value": float(v)})
        ref_lines(ax, task)
        if task == "ppl":
            ax.set_yscale("log")
        ax.set_title(LABEL[task], fontsize=9.5)
        ax.set_xlabel("number of layers removed (of 22)")
    axes[1].legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p2_layer_removal.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).pivot_table(index=["order", "layers_removed"], columns="task", values="value") \
        .reset_index().to_csv(os.path.join(RES, "E3_layer_removal.csv"), index=False)


# ------------------------------------------------------------ E4 embeddings
def fig_embeddings():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    rows = []
    for ax, (kind, xlabel) in zip(axes, [("noise", "Gaussian noise std (x embedding RMS)"),
                                         ("mask", "fraction of token embeddings zeroed")]):
        xs = sorted({float(k.split("/")[2]) for k in R if k.startswith(f"E4/{kind}/")})
        for task, col in zip(TASKS, C):
            y = [retained(R[f"E4/{kind}/{x}"], task) for x in xs]
            ax.plot(range(len(xs)), y, "-o", ms=3.5, color=col, label=SHORT[task], lw=1.6)
            for x in xs:
                rows.append({"corruption": kind, "level": x, "task": task, "value": R[f"E4/{kind}/{x}"][task],
                             "retained": retained(R[f"E4/{kind}/{x}"], task)})
        ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
        ax.set_xlabel(xlabel)
        ax.axhline(0, color=MUTED, ls=":", lw=0.9)
        ax.axhline(1, color=INK, ls="--", lw=0.9)
    axes[0].set_ylabel("retained score (1 = intact, 0 = chance)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p2_embeddings.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "E4_embeddings.csv"), index=False)


# ------------------------------------------------------------ syntax vs semantics
def fig_syntax_vs_semantics():
    """Retained syntax vs retained facts/sentiment for every ablation config."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True)
    groups = [("E1/", "single layer/block"), ("E2/", "head pruning"), ("E3/", "layer removal"), ("E4/", "embedding corruption")]
    for ax, sem in zip(axes, ["facts", "sst"]):
        for (pre, lab), col in zip(groups, C):
            pts = [(retained(r, "blimp"), retained(r, sem)) for k, r in R.items() if k.startswith(pre)]
            x, y = zip(*pts)
            ax.scatter(x, y, s=22, color=col, label=lab, edgecolor="white", linewidth=0.6, alpha=0.9)
        ax.plot([-0.2, 1.2], [-0.2, 1.2], color=MUTED, lw=0.9, ls="--")
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(-0.25, 1.25)
        ax.set_xlabel("retained syntax (BLiMP)")
        ax.set_title(f"syntax vs {SHORT[sem].lower()}", fontsize=9.5)
    axes[0].set_ylabel("retained semantic score")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p2_syntax_vs_semantics.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    fig_layer_sweep()
    fig_head_pruning()
    fig_layer_removal()
    fig_embeddings()
    fig_syntax_vs_semantics()
    print("baseline:", {k: v for k, v in BASE.items() if k != "spec"})


if __name__ == "__main__":
    main()
