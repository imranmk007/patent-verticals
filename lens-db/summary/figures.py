import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RAW_PATH = "lens-raw-data.jsonl"
CLEAN_PATH = "US-lens-clean.json"
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
BLUE   = "#2E86AB"
GREEN  = "#4CAF50"
ORANGE = "#F4845F"
PURPLE = "#7B5EA7"
GREY   = "#B0B8C1"
RED    = "#C0392B"
TEAL   = "#1ABC9C"
GOLD   = "#F39C12"
PALETTE = [BLUE, GREEN, ORANGE, PURPLE, TEAL, GOLD, RED, GREY]

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {name}")

# ── load data ─────────────────────────────────────────────────────────────────
raw = []
with open(RAW_PATH) as f:
    for line in f:
        raw.append(json.loads(line))

with open(CLEAN_PATH) as f:
    clean = json.load(f)

N_RAW   = len(raw)
N_CLEAN = len(clean)

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Jurisdiction distribution (raw)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 1: jurisdiction")
jur_counts = Counter(r.get("jurisdiction") for r in raw)
top_jur = jur_counts.most_common(12)
labels, vals = zip(*top_jur)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(labels[::-1], vals[::-1], color=BLUE, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, vals[::-1]):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=9)
ax.set_xlabel("Number of patents", fontsize=11)
ax.set_title("Patent jurisdiction distribution (raw dataset, n=8,357)", fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, max(vals) * 1.15)
savefig("fig1_jurisdiction.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Language distribution (raw, top 10)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 2: language")
LANG_NAMES = {
    "en": "English", "zh": "Chinese", "ja": "Japanese", "de": "German",
    "ko": "Korean", "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "ru": "Russian", "pl": "Polish", "it": "Italian", None: "Unknown",
}
lang_counts = Counter(r.get("lang") for r in raw)
top_lang = lang_counts.most_common(10)
labels = [LANG_NAMES.get(k, k or "Unknown") for k, _ in top_lang]
vals   = [v for _, v in top_lang]
pcts   = [100 * v / N_RAW for v in vals]

fig, ax = plt.subplots(figsize=(8, 5))
colors = [GREEN if k == "en" else BLUE for k, _ in top_lang]
bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)
for bar, val, pct in zip(bars, vals[::-1], pcts[::-1]):
    ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2,
            f"{val:,}  ({pct:.1f}%)", va="center", fontsize=9)
ax.set_xlabel("Number of patents", fontsize=11)
ax.set_title("Language distribution (raw dataset)", fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, max(vals) * 1.22)
en_patch  = mpatches.Patch(color=GREEN, label="English")
oth_patch = mpatches.Patch(color=BLUE,  label="Other")
ax.legend(handles=[en_patch, oth_patch], fontsize=9)
savefig("fig2_language.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Jurisdiction × Language heatmap (raw, top jurisdictions)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 3: jurisdiction × language heatmap")
TOP_JUR  = [j for j, _ in Counter(r.get("jurisdiction") for r in raw).most_common(10)]
TOP_LANG = [l for l, _ in Counter(r.get("lang") for r in raw).most_common(8) if l is not None]

matrix = np.zeros((len(TOP_JUR), len(TOP_LANG)))
for r in raw:
    j = r.get("jurisdiction")
    l = r.get("lang")
    if j in TOP_JUR and l in TOP_LANG:
        matrix[TOP_JUR.index(j), TOP_LANG.index(l)] += 1

lang_labels = [LANG_NAMES.get(l, l) for l in TOP_LANG]

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(matrix, aspect="auto", cmap="Blues")
ax.set_xticks(range(len(TOP_LANG))); ax.set_xticklabels(lang_labels, rotation=30, ha="right", fontsize=10)
ax.set_yticks(range(len(TOP_JUR)));  ax.set_yticklabels(TOP_JUR, fontsize=10)
for i in range(len(TOP_JUR)):
    for j in range(len(TOP_LANG)):
        v = int(matrix[i, j])
        if v > 0:
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=8, color="white" if v > matrix.max()*0.5 else "black")
plt.colorbar(im, ax=ax, label="Count")
ax.set_title("Jurisdiction × Language (raw dataset, top-10 jurisdictions)", fontsize=12, fontweight="bold")
savefig("fig3_jurisdiction_language_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Field completeness (raw)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 4: field completeness")
FIELDS = [
    ("abstract",         "Abstract"),
    ("description",      "Description"),
    ("claims",           "Claims"),
    ("lang",             "Language tag"),
    ("docdb_id",         "DOCDB ID"),
    ("sequence_listing", "Sequence listing"),
]
labels = [label for _, label in FIELDS]
pcts   = [100 * sum(1 for r in raw if r.get(field)) / N_RAW for field, _ in FIELDS]

fig, ax = plt.subplots(figsize=(8, 4.5))
colors = [GREEN if p == 100 else BLUE if p >= 80 else ORANGE if p >= 40 else RED for p in pcts]
bars = ax.barh(labels[::-1], pcts[::-1], color=colors[::-1], edgecolor="white")
for bar, pct in zip(bars, pcts[::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{pct:.1f}%", va="center", fontsize=10)
ax.axvline(100, color=GREY, linestyle="--", linewidth=1)
ax.set_xlim(0, 115)
ax.set_xlabel("% of records with field present", fontsize=11)
ax.set_title("Field completeness (raw dataset, n=8,357)", fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
savefig("fig4_field_completeness.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Publication type distribution (raw)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 5: publication type")
pub_counts = Counter(r.get("publication_type", "UNKNOWN") for r in raw)
labels, vals = zip(*pub_counts.most_common())
pcts = [100 * v / N_RAW for v in vals]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(range(len(labels)), vals,
              color=[BLUE, GREEN, ORANGE, PURPLE, TEAL, GOLD, RED, GREY][:len(labels)],
              edgecolor="white", linewidth=0.5)
for bar, val, pct in zip(bars, vals, pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=8.5)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Publication type distribution (raw dataset)", fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
savefig("fig5_publication_type.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — Patent status distribution (raw)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 6: patent status")
STATUS_COLORS = {
    "ACTIVE": GREEN, "PENDING": BLUE, "EXPIRED": GREY,
    "DISCONTINUED": ORANGE, "INACTIVE": RED, "UNKNOWN": PURPLE, "PATENTED": TEAL,
}
status_counts = Counter(r.get("legal_status", {}).get("patent_status") for r in raw)
labels, vals = zip(*status_counts.most_common())
colors = [STATUS_COLORS.get(l, GREY) for l in labels]
pcts = [100 * v / N_RAW for v in vals]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.5)
for bar, val, pct in zip(bars, vals, pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Patent legal status distribution (raw dataset)", fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
savefig("fig6_patent_status.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — Publication year (raw, 2000–2024)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 7: publication year")
year_counts = Counter()
for r in raw:
    d = r.get("date_published", "")
    if d:
        year_counts[int(d[:4])] += 1

years = range(2000, 2025)
counts = [year_counts.get(y, 0) for y in years]

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(list(years), counts, color=BLUE, edgecolor="white", linewidth=0.3)
ax.set_xlabel("Publication year", fontsize=11)
ax.set_ylabel("Number of patents", fontsize=11)
ax.set_title("Publication year distribution (raw dataset, 2000–2024)", fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xticks(list(range(2000, 2025, 2)))
savefig("fig7_publication_year.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — IPC top-level section distribution (raw, counts per record counted once)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 8: IPC sections")
IPC_SECTIONS = {
    "A": "Human Necessities",
    "B": "Operations & Transport",
    "C": "Chemistry & Metallurgy",
    "D": "Textiles & Paper",
    "E": "Fixed Constructions",
    "F": "Mechanical Engineering",
    "G": "Physics / Computing",
    "H": "Electricity / Electronics",
}
section_patent_counts = defaultdict(set)
for i, r in enumerate(raw):
    for cls in r.get("biblio", {}).get("classifications_ipcr", {}).get("classifications", []):
        sym = cls.get("symbol", "")
        if sym:
            section_patent_counts[sym[0]].add(i)
sec_counts = {s: len(ids) for s, ids in section_patent_counts.items()}
sec_labels = [f"{s} – {IPC_SECTIONS.get(s,'')}" for s in sorted(sec_counts)]
sec_vals   = [sec_counts[s] for s in sorted(sec_counts)]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(sec_labels[::-1], sec_vals[::-1],
               color=[PALETTE[i % len(PALETTE)] for i in range(len(sec_labels))],
               edgecolor="white")
for bar, val in zip(bars, sec_vals[::-1]):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=9)
ax.set_xlabel("Number of patents (unique)", fontsize=11)
ax.set_title("IPC top-level section distribution (raw dataset)", fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, max(sec_vals) * 1.12)
savefig("fig8_ipc_sections.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 — Cleaning funnel
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 9: cleaning funnel")
n_raw        = N_RAW
n_us         = sum(1 for r in raw if r.get("jurisdiction") == "US")
n_us_abs     = sum(1 for r in raw if r.get("jurisdiction") == "US" and r.get("abstract"))

# deduplicated count = N_CLEAN
stages = [
    ("Raw data",                n_raw),
    ("US jurisdiction only",    n_us),
    ("Has abstract",            n_us_abs),
    ("Deduplicated",            N_CLEAN),
]
labels = [s for s, _ in stages]
vals   = [v for _, v in stages]
drops  = [vals[i-1] - vals[i] for i in range(1, len(vals))]

fig, ax = plt.subplots(figsize=(9, 4.5))
bar_colors = [BLUE, BLUE, BLUE, GREEN]
bars = ax.barh(labels[::-1], vals[::-1], color=bar_colors[::-1], edgecolor="white")
for bar, val in zip(bars, vals[::-1]):
    ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=10, fontweight="bold")

# annotate drops
drop_labels = drops + [None]
for i, (bar, drop) in enumerate(zip(bars[:-1], drops[::-1])):
    ax.text(vals[-(i+1)] / 2, bar.get_y() + bar.get_height()/2,
            f"−{drop:,}", ha="center", va="center", fontsize=8.5,
            color="white", fontweight="bold")

ax.set_xlabel("Number of records", fontsize=11)
ax.set_title("Data cleaning funnel: raw → final dataset", fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, n_raw * 1.18)
savefig("fig9_cleaning_funnel.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 10 — Abstract length distribution (clean, word count)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 10: abstract length")
abs_lengths = []
for r in clean:
    ab = r.get("abstract", "")
    if ab and ab != "null":
        abs_lengths.append(len(ab.split()))

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(abs_lengths, bins=60, color=BLUE, edgecolor="white", linewidth=0.3)
ax.axvline(np.median(abs_lengths), color=ORANGE, linestyle="--", linewidth=1.5,
           label=f"Median: {int(np.median(abs_lengths))} words")
ax.axvline(np.mean(abs_lengths), color=RED, linestyle="--", linewidth=1.5,
           label=f"Mean: {int(np.mean(abs_lengths))} words")
ax.set_xlabel("Abstract length (words)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Abstract length distribution (cleaned US dataset, n=2,779)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
savefig("fig10_abstract_length.png")

# ─────────────────────────────────────────────────────────────────────────────
# STATS JSON
# ─────────────────────────────────────────────────────────────────────────────
print("\nWriting stats.json...")
stats = {
    "raw": {
        "total_records": N_RAW,
        "jurisdictions": dict(Counter(r.get("jurisdiction") for r in raw).most_common()),
        "languages": {str(k): v for k, v in Counter(r.get("lang") for r in raw).most_common()},
        "publication_types": dict(Counter(r.get("publication_type","UNKNOWN") for r in raw).most_common()),
        "patent_statuses": {str(k): v for k, v in Counter(
            r.get("legal_status", {}).get("patent_status") for r in raw).most_common()},
        "field_completeness": {
            field: round(100 * sum(1 for r in raw if r.get(field)) / N_RAW, 1)
            for field in ["abstract", "description", "claims", "lang", "docdb_id", "sequence_listing"]
        },
        "years": {str(k): v for k, v in sorted(year_counts.items())},
    },
    "cleaned": {
        "total_records": N_CLEAN,
        "abstract_word_count": {
            "mean":   round(float(np.mean(abs_lengths)), 1),
            "median": int(np.median(abs_lengths)),
            "std":    round(float(np.std(abs_lengths)), 1),
            "p90":    int(np.percentile(abs_lengths, 90)),
            "max":    max(abs_lengths),
        },
    },
    "cleaning_funnel": {s: v for s, v in stages},
}
with open("stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("Done. Figures written to figures/")
