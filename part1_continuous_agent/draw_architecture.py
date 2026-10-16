"""Draws the CALM architecture diagram used in the report (../report/figures/p1_architecture.png)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "report", "figures", "p1_architecture.png")

INK, SUB = "#0b0b0b", "#52514e"
FILL = {"env": "#eeeeea", "perc": "#dbe8f8", "mem": "#d9f1e7", "mod": "#fde9c4", "act": "#fbe0d4"}
EDGE = {"env": "#8a8984", "perc": "#2a78d6", "mem": "#1baf7a", "mod": "#c98500", "act": "#eb6834"}

fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 110)
ax.set_ylim(0, 62)
ax.axis("off")


def box(x, y, w, h, kind, title, body):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc=FILL[kind], ec=EDGE[kind], lw=1.4))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=9.5, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 5.0, body, ha="center", va="top", fontsize=7.6, color=SUB, linespacing=1.35)


def arrow(p, q, text=None, color=SUB, tx=0, ty=0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=color, linestyle=ls))
    if text:
        ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2 + ty, text, fontsize=7.3, color=color,
                ha="center", va="center", style="italic",
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))


# column 1: co-occurrence memory + world | column 2: perception, outcome, action
# column 3: associative memories         | column 4: neuromodulatory units
box(1, 44, 19, 14, "mem", "Co-occurrence", "memory A[type, type]\nHebbian (instar) on\nco-visible objects;\nno reward needed")
box(1, 20, 19, 18, "env", "Dynamic world", "objects move, appear,\nexpire; rules change\n(valence reversal,\nnew pairings,\nnovel objects)")
box(27, 44, 20, 14, "perc", "Sensory feature units", "egocentric 7x7 view\nx = [colour | shape | 1]\nfactorised (9 units)")
box(27, 24, 20, 11, "env", "Outcome", "reward +1 / -1\nsound (1 of 4)")
box(27, 2, 20, 15, "act", "Action selection", "value + beta x uncertainty\n- cost x distance -> softmax;\nnothing good in view ->\nexplore least-visited space")
box(55, 44, 22, 14, "mem", "Associative cortex", "W_v: features -> valence\nW_s: features -> sound\nonline delta rule (R-W)")
box(55, 21, 22, 14, "mem", "Episodic buffer", "last 200 experiences\nrecency-weighted replay\n(4 updates / interaction)")
box(85, 44, 23, 14, "mod", "Associability units u_f", "running |PE| per feature\nsets learning rate\n(Pearce-Hall) + curiosity")
box(85, 21, 23, 14, "mod", "Change detector", "fast vs slow surprise\n(unexpected uncertainty)\non change: u up, counts\ndown, replay recent only")

arrow((20.4, 37), (26.6, 45), "observe", tx=-3)
arrow((26.6, 52), (20.4, 52))
arrow((20.4, 29.5), (26.6, 29.5))
arrow((26.6, 9), (12, 19.6), "move / touch", tx=-4)
arrow((47.4, 52), (54.6, 52), "x", ty=1.8)
arrow((47.4, 33), (54.6, 44.5), "PE", tx=-1.6, ty=0.6)
arrow((47.4, 28), (54.6, 28), "store", ty=1.8)
arrow((66, 35.4), (66, 43.6), "replay", tx=4)
arrow((56, 43.6), (47.4, 16), "predicted value", tx=0.8, ty=-11.3)
arrow((77.4, 54), (84.6, 54), "PE", ty=1.8)
arrow((84.6, 48), (77.4, 48), "lr", ty=-1.8, ls="--")
arrow((96.5, 43.6), (96.5, 35.4), "surprise", tx=4.5)
arrow((84.6, 28), (77.4, 28), "recency", ty=1.8, ls="--")
ax.plot([96.5, 96.5, 60], [20.6, 9, 9], color=SUB, lw=1.2, ls="--")
arrow((60, 9), (47.4, 9), None, ls="--")
ax.text(78, 10.8, "uncertainty -> curiosity bonus", fontsize=7.3, color=SUB, ha="center", style="italic")

ax.text(55, 61, "CALM - Continual Associative Learning Model: perceive -> predict -> act -> observe -> update, every step",
        ha="center", fontsize=10.5, weight="bold", color=INK)
ax.text(55, -1.2, "solid arrows: information flow     dashed arrows: neuromodulatory control (learning rate, exploration, replay)",
        ha="center", fontsize=7.6, color=SUB)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print("saved", OUT)
