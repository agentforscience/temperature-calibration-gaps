"""Redraw the paper figures from results/analysis. Run from the workspace root."""
import ast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

R = "results/analysis/"
OUT = "paper_draft/figures/"
MODELS = [  # ordered by size; alpha rises with size within a family
    ("qwen2.5-0.5b", "Qwen 0.5B", "Qwen", .35), ("llama3.2-1b", "Llama 1B", "Llama", .4),
    ("qwen2.5-1.5b", "Qwen 1.5B", "Qwen", .5), ("qwen2.5-3b", "Qwen 3B", "Qwen", .65),
    ("llama3.2-3b", "Llama 3B", "Llama", .7), ("mistral-7b", "Mistral 7B", "Mistral", 1),
    ("qwen2.5-7b", "Qwen 7B", "Qwen", .8), ("llama3.1-8b", "Llama 8B", "Llama", 1),
    ("gemma2-9b", "Gemma 9B", "Gemma", 1), ("qwen2.5-14b", "Qwen 14B", "Qwen", 1),
]
COLOR = {"Qwen": "#2a78d6", "Llama": "#d9772a", "Gemma": "#6b4fbb", "Mistral": "#1f9e8f"}
MARK = {"Qwen": "o", "Llama": "s", "Gemma": "^", "Mistral": "D"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "pdf.fonttype": 42})
family_legend = [Line2D([], [], color=COLOR[f], marker=MARK[f], ms=4, lw=1.5, label=f) for f in COLOR]
curves = pd.read_csv(R + "curves.csv")


def curve(task, metric, model):
    return curves[(curves.task == task) & (curves.metric == metric) & (curves.model == model)].sort_values("T")


# Normalized quality: each model divided by its own peak.
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
panels = [("story", "coherent", "Stories: coherent rate"), ("qa", "acc", "TriviaQA: accuracy"),
          ("code", "pass1", "HumanEval: pass@1")]
for ax, (task, metric, title) in zip(axes, panels):
    for key, _, fam, a in MODELS:
        c = curve(task, metric, key)
        ax.plot(c["T"], c["mean"] / c["mean"].max(), color=COLOR[fam], alpha=a, lw=1.2, marker=MARK[fam], ms=2.2)
    ax.axhline(.9, color=MUTED, lw=.7, ls="--")
    ax.set_title(title, fontsize=8.5, color=INK)
    ax.set_xlabel("temperature")
axes[0].set_ylabel("share of model's peak")
axes[0].legend(handles=family_legend, frameon=False, fontsize=7, loc="lower left")
fig.tight_layout(pad=.4)
fig.savefig(OUT + "quality_normalized.pdf")

# Useful-range upper bound with 95% CI, by model size.
u = pd.read_csv(R + "useful_ranges.csv").set_index(["task", "model"])
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True, sharex=True)
for ax, (task, title) in zip(axes, [("story", "Stories"), ("qa", "TriviaQA"), ("code", "HumanEval")]):
    for y, (key, _, fam, _a) in enumerate(MODELS):
        r = u.loc[(task, key)]
        lo, hi = ast.literal_eval(r.T_u_ci)
        ax.plot([lo, hi], [y, y], color=COLOR[fam], lw=1.4)
        ax.plot(r.T_u, y, color=COLOR[fam], marker=MARK[fam], ms=4.5)
    ax.set_title(title, fontsize=8.5, color=INK)
    ax.set_xlabel(r"$T_u$")
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(MODELS)), [m[1] for m in MODELS])
fig.tight_layout(pad=.4)
fig.savefig(OUT + "useful_ranges.pdf")

# Diversity-coherence frontier.
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
for ax, (metric, label) in zip(axes, [("sem_div", "semantic diversity"), ("distinct2", "distinct-2")]):
    for key, _, fam, a in MODELS:
        x = curve("story", metric, key)
        y = curve("story", "judge", key)
        m = x.merge(y, on="T", suffixes=("_x", "_y"))
        if metric == "distinct2":
            m = m[m["T"] > 0]  # no pooled samples to compare at T = 0
        ax.plot(m.mean_x, m.mean_y, color=COLOR[fam], alpha=a, lw=1.2, marker=MARK[fam], ms=2.2)
        k = m[m["T"] == 1.2]
        ax.plot(k.mean_x, k.mean_y, ls="", marker=MARK[fam], ms=5.5, mfc="white", mec=COLOR[fam], mew=1)
    ax.set_xlabel(label)
axes[0].set_ylabel("judge coherence (1--9)")
axes[0].legend(handles=family_legend + [Line2D([], [], ls="", marker="o", mfc="white", mec=MUTED, ms=5, label="T = 1.2")],
               frameon=False, fontsize=7, loc="lower left", ncol=2)
fig.tight_layout(pad=.4)
fig.savefig(OUT + "story_frontier.pdf")
