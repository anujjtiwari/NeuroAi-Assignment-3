"""
Runs the CALM agent and its ablated variants in the non-stationary world,
then saves curves + summary tables to ../results/part1 and figures to ../report/figures.

    python run_experiments.py            (about 2-4 minutes on a laptop CPU)
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment import AssociativeWorld, SEEN, HELD_OUT, N_TYPES
from calm_agent import CALMAgent, CALMConfig

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "part1")
FIG = os.path.join(HERE, "..", "report", "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

T, SWITCH, NOVEL, PROBE_EVERY = 12000, 6000, 9000, 50
SEEDS = range(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
BASE = CALMConfig()
CONDITIONS = {
    "CALM (full)":            BASE,
    "No curiosity":           replace(BASE, curiosity=0.0),
    "Fixed learning rate":    replace(BASE, associability=False),
    "No change detection":    replace(BASE, change_detection=False),
    "No replay":              replace(BASE, replay=0),
    "Passive (random acts)":  replace(BASE, passive=True),
    "Frozen after phase 1":   BASE,     # learning switched off at t = SWITCH ("pretrained, then deployed")
}


def probe(agent, env):
    val = lambda types: np.mean([(agent.p_edible(t) > 0.5) == env.true_valence_sign(t) for t in types])
    snd = lambda types: np.mean([np.argmax(agent.sound_probs(t)) == env.true_sound(t) for t in types])
    active = SEEN + (HELD_OUT if env.t >= NOVEL else [])
    partner = np.mean([agent.predicted_partner(t) == env.partner(t) for t in active])
    return {"valence_seen": val(SEEN), "valence_novel": val(HELD_OUT),
            "sound_seen": snd(SEEN), "sound_novel": snd(HELD_OUT), "partner": partner}


def run(name, cfg, seed):
    env = AssociativeWorld(switch_step=SWITCH, novel_step=NOVEL, seed=seed)
    agent = CALMAgent(cfg, seed=seed + 100)
    rows, rewards, n_inter = [], [], 0
    reward_win = []
    for step in range(T):
        if name.startswith("Frozen") and step == SWITCH:
            agent.cfg = replace(agent.cfg, learning=False)
        move, obj = agent.act(env)
        out = env.step(move, obj)
        if out is not None:
            agent.learn_from_outcome(out)
            reward_win.append(out["reward"])
            n_inter += 1
        if (step + 1) % PROBE_EVERY == 0:
            r = probe(agent, env)
            r.update(step=step + 1, condition=name, seed=seed,
                     reward_rate=np.sum(reward_win) / PROBE_EVERY,     # net reward per step
                     interactions=len(reward_win) / PROBE_EVERY)
            rows.append(r)
            reward_win = []
    return rows


def recovery_step(df, metric="valence_seen", level=0.9):
    """Steps after the switch until the metric is back above `level` (and stays for 3 probes)."""
    post = df[df.step > SWITCH].sort_values("step")
    ok = (post[metric] >= level).rolling(3).min() == 1
    hit = post[ok.values]
    return (hit.step.iloc[0] - 2 * PROBE_EVERY - SWITCH) if len(hit) else np.nan


def main():
    jobs = [(name, cfg, seed) for name, cfg in CONDITIONS.items() for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as ex:
        results = list(ex.map(run, *zip(*jobs)))
    df = pd.DataFrame([row for rows in results for row in rows])
    df.to_csv(os.path.join(RES, "curves.csv"), index=False)

    # ---- summary table ----
    summ = []
    for name in CONDITIONS:
        d = df[df.condition == name]
        p1 = d[(d.step > SWITCH - 1000) & (d.step <= SWITCH)]
        p2 = d[(d.step > SWITCH) & (d.step <= SWITCH + 1500)]
        late = d[d.step > T - 1000]
        pre_novel = d[(d.step > NOVEL - 1000) & (d.step <= NOVEL)]
        rec = [recovery_step(d[d.seed == s]) for s in SEEDS]
        summ.append({
            "Condition": name,
            "Valence acc. end of phase 1": p1.valence_seen.mean(),
            "Valence acc. first 1500 steps after switch": p2.valence_seen.mean(),
            "Valence acc. final": late.valence_seen.mean(),
            "Steps to recover 90% after switch": np.nanmean(rec) if not np.all(np.isnan(rec)) else np.nan,
            "Seeds that recovered": int(np.sum(~np.isnan(rec))),
            "Zero-shot valence (unseen combos)": pre_novel.valence_novel.mean(),
            "Zero-shot sound (unseen combos)": pre_novel.sound_novel.mean(),
            "Sound acc. final (stable rule)": late.sound_seen.mean(),
            "Partner acc. end of phase 1": p1.partner.mean(),
            "Partner acc. final": late.partner.mean(),
            "Net reward / step (whole run)": d.reward_rate.mean(),
            "Net reward / step (phase 2)": d[d.step > SWITCH].reward_rate.mean(),
        })
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(RES, "summary.csv"), index=False)
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summ.to_dict(orient="records"), f, indent=1)
    print(summ.round(3).to_string())

    # ---- figures ----
    colors = dict(zip(CONDITIONS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
    g = df.groupby(["condition", "step"]).agg(["mean", "sem"]).reset_index()

    def panel(ax, metric, title, conds=CONDITIONS, smooth=1):
        for name in conds:
            d = g[g.condition == name]
            m = d[(metric, "mean")].rolling(smooth, min_periods=1).mean()
            s = d[(metric, "sem")].rolling(smooth, min_periods=1).mean()
            ax.plot(d.step, m, label=name, color=colors[name], lw=1.6)
            ax.fill_between(d.step, m - s, m + s, color=colors[name], alpha=0.15, lw=0)
        ax.axvline(SWITCH, color="k", ls="--", lw=0.8)
        ax.axvline(NOVEL, color="k", ls=":", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time step")
        ax.grid(alpha=0.3)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    panel(axes[0, 0], "valence_seen", "Valence prediction accuracy (seen types)")
    panel(axes[0, 1], "reward_rate", "Net reward per step (smoothed)", smooth=5)
    panel(axes[1, 0], "partner", "Co-occurrence (partner) accuracy")
    panel(axes[1, 1], "sound_seen", "Sound prediction accuracy (stable rule)")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.text(0.5, 0.005, "dashed line: rule switch (valence reversal + new pairings)    "
             "dotted line: unseen colour/shape combinations start to appear",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(FIG, "p1_learning_curves.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    panel(ax[0], "valence_novel", "Valence accuracy on never-seen combinations",
          conds=["CALM (full)", "No curiosity", "Passive (random acts)", "Frozen after phase 1"])
    panel(ax[1], "interactions", "Interactions per step (smoothed)", smooth=5)
    ax[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p1_generalisation.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
