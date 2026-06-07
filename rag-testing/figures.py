"""
Generates two figures from retrieval results.
Output saved to results/fig1_scores.png and results/fig2_times.png.
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
SYSTEMS = ["bm25", "minilm", "bge"]
COLORS = {"bm25": "#4e79a7", "minilm": "#f28e2b", "bge": "#59a14f"}

TESTS = [
    "test1a", "test1b", "test1c",
    "test2a", "test2b", "test2c",
    "test3a", "test3b", "test3c",
]

# load

data = {}
for path in glob.glob(str(RESULTS_DIR / "**/*.json"), recursive=True):
    if "embeddings" in path or "metadata" in path:
        continue
    with open(path) as f:
        r = json.load(f)
    key = Path(r["query_file"]).stem.split("_")[0]
    data[(r["system"], key)] = r


# ── Figure 1: rank-1 confidence scores ───────────────────────────────────────
# BM25 uses left axis; MiniLM + BGE share right axis

fig, ax_left = plt.subplots(figsize=(10, 4))
ax_right = ax_left.twinx()

x = np.arange(len(TESTS))
jitter = {"bm25": -0.12, "minilm": 0.0, "bge": 0.12}
axes   = {"bm25": ax_left, "minilm": ax_right, "bge": ax_right}

for sys in SYSTEMS:
    scores = [data[(sys, k)]["results"][0]["score"] for k in TESTS]
    ax = axes[sys]
    ax.scatter(x + jitter[sys], scores, color=COLORS[sys], s=60,
               label=sys, zorder=3)
    ax.plot(x + jitter[sys], scores, color=COLORS[sys], alpha=0.3,
            linewidth=1, zorder=2)

ax_left.set_ylabel("BM25 score", color=COLORS["bm25"])
ax_left.tick_params(axis="y", labelcolor=COLORS["bm25"])
ax_right.set_ylabel("cosine similarity (MiniLM / BGE)", color="#888")
ax_right.tick_params(axis="y", labelcolor="#888")

ax_left.set_xticks(x)
ax_left.set_xticklabels(TESTS, rotation=35, ha="right", fontsize=8)
ax_left.set_xlabel("test")

lines_l, labels_l = ax_left.get_legend_handles_labels()
lines_r, labels_r = ax_right.get_legend_handles_labels()
ax_left.legend(lines_l + lines_r, labels_l + labels_r, loc="upper right", fontsize=8)

ax_left.set_title("rank-1 confidence score per test")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "fig1_scores.png", dpi=150)
plt.close()
print("saved fig1_scores.png")


# ── Figure 2: query time by input length (tests 2 & 3 only) ──────────────────
# x = length category, y = query time (log scale), 2 points per category

length_map = {
    "test2a": "short", "test2b": "medium", "test2c": "long",
    "test3a": "short", "test3b": "medium", "test3c": "long",
}
test_keys = ["test2a", "test2b", "test2c", "test3a", "test3b", "test3c"]
categories = ["short", "medium", "long"]
cat_x = {"short": 0, "medium": 1, "long": 2}

fig, ax = plt.subplots(figsize=(7, 4))

rng = np.random.default_rng(42)
jitter_amount = 0.08
sys_jitter = {"bm25": -0.12, "minilm": 0.0, "bge": 0.12}

for sys in SYSTEMS:
    by_cat = {"short": [], "medium": [], "long": []}
    for k in test_keys:
        r = data.get((sys, k))
        if r:
            cat = length_map[k]
            by_cat[cat].append(r["query_time_ms"])

    for cat, times in by_cat.items():
        base_x = cat_x[cat] + sys_jitter[sys]
        jx = base_x + rng.uniform(-jitter_amount, jitter_amount, len(times))
        ax.scatter(jx, times, color=COLORS[sys], s=55, zorder=3,
                   label=sys if cat == "short" else None)

    # connect medians across categories to show trend
    medians = [np.median(by_cat[c]) for c in categories]
    med_x   = [cat_x[c] + sys_jitter[sys] for c in categories]
    ax.plot(med_x, medians, color=COLORS[sys], linewidth=1.2,
            alpha=0.6, zorder=2)

ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.2f}ms"))
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(categories)
ax.set_xlabel("input length")
ax.set_ylabel("query time (ms, log scale)")
ax.set_title("query time by input length — tests 2 & 3")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "fig2_times.png", dpi=150)
plt.close()
print("saved fig2_times.png")
