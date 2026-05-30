# Harvard USPTO Dataset (HUPD) - Quality & Diversity Visualizations
#
# Generates charts + numeric summaries demonstrating the diversity and quality
# of HUPD for use as a corpus in Mantis (a semantic-embedding data-science
# platform). For embeddings, what matters is: topical diversity (IPC/CPC
# categories), label availability/balance (accept/reject/pending), and the
# amount/shape of natural-language text per document (length distributions).
#
# Analysis set: the cross-year 2100-app sample cached at
# ../data/hupd_crossyear_2100_clean.parquet by data-load.py.
#
# Outputs:
#   figures/*.png   - charts (150 dpi)
#   stats/*.csv     - per-field length statistics
#   stats/summary.json - overall dataset summary numbers
#
# Run: ./.venv/bin/python summary/visualize.py

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, don't open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DATA_FILE = BASE.parent / "data" / "hupd_crossyear_2100_clean.parquet"
FIG_DIR = BASE / "figures"
STATS_DIR = BASE / "stats"
FIG_DIR.mkdir(exist_ok=True)
STATS_DIR.mkdir(exist_ok=True)

DPI = 150
TEXT_FIELDS = ["title", "abstract", "claims", "background", "summary", "full_description"]

# WIPO classification sections (first char of an IPC/CPC label).
SECTION_NAMES = {
    "A": "A · Human Necessities",
    "B": "B · Operations & Transport",
    "C": "C · Chemistry & Metallurgy",
    "D": "D · Textiles & Paper",
    "E": "E · Fixed Constructions",
    "F": "F · Mech. Eng / Heating / Weapons",
    "G": "G · Physics",
    "H": "H · Electricity",
    "Y": "Y · Emerging / Cross-sectional (CPC)",
}

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")
plt.rcParams.update({"figure.autolayout": True, "axes.titleweight": "bold"})


def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(BASE.parent)}")


def section_of(label):
    """First-letter WIPO section of an IPC/CPC label, or None if blank."""
    if label is None:
        return None
    label = str(label).strip()
    return label[0] if label else None


# ---------------------------------------------------------------------------
# Load + derive
# ---------------------------------------------------------------------------
def load():
    df = pd.read_parquet(DATA_FILE)
    df["filing_year"] = pd.to_datetime(
        df["filing_date"], format="%Y%m%d", errors="coerce"
    ).dt.year
    df["ipc_section"] = df["main_ipcr_label"].map(section_of)
    df["cpc_section"] = df["main_cpc_label"].map(section_of)
    df["ipc_subclass"] = df["main_ipcr_label"].fillna("").str.slice(0, 4)
    df["n_cpc_labels"] = df["cpc_labels"].map(lambda x: len(x) if x is not None else 0)
    df["n_ipc_labels"] = df["ipcr_labels"].map(lambda x: len(x) if x is not None else 0)
    df["n_inventors"] = df["inventor_list"].map(lambda x: len(x) if x is not None else 0)
    for f in TEXT_FIELDS:
        df[f"{f}_words"] = df[f].fillna("").str.split().str.len()
        df[f"{f}_chars"] = df[f].fillna("").str.len()
    return df


# ---------------------------------------------------------------------------
# 1. IPC / CPC category diversity
# ---------------------------------------------------------------------------
def fig_ipc_cpc_sections(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, col, title in [
        (axes[0], "ipc_section", "IPC sections (primary label)"),
        (axes[1], "cpc_section", "CPC sections (primary label)"),
    ]:
        counts = df[col].value_counts().sort_index()
        labels = [SECTION_NAMES.get(s, s) for s in counts.index]
        ax.barh(labels, counts.values, color="#4C72B0")
        ax.set_xlabel("patents")
        cov = df[col].notna().mean() * 100
        ax.set_title(f"{title}\n(coverage: {cov:.0f}% of docs labelled)")
        ax.invert_yaxis()
        for y, v in enumerate(counts.values):
            ax.text(v, y, f" {v:,}", va="center", fontsize=9)
    fig.suptitle(
        "Technology diversity across WIPO sections — broad spread = rich embedding space",
        fontsize=13, fontweight="bold",
    )
    _save(fig, "fig01_ipc_cpc_sections.png")


def fig_top_ipc_subclasses(df, top=20):
    counts = (
        df.loc[df["ipc_subclass"].str.len() >= 4, "ipc_subclass"]
        .value_counts()
        .head(top)
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(counts.index[::-1], counts.values[::-1], color="#55A868")
    ax.set_xlabel("patents")
    ax.set_title(
        f"Top {top} IPC subclasses\n"
        f"({df['ipc_subclass'].nunique():,} distinct subclasses present in {len(df):,} docs)"
    )
    for y, v in enumerate(counts.values[::-1]):
        ax.text(v, y, f" {v:,}", va="center", fontsize=9)
    _save(fig, "fig02_top_ipc_subclasses.png")


# ---------------------------------------------------------------------------
# 2. Label availability & balance (accept / reject / pending)
# ---------------------------------------------------------------------------
def fig_decision(df):
    counts = df["decision"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"ACCEPTED": "#55A868", "REJECTED": "#C44E52", "PENDING": "#8172B3"}
    bars = ax.bar(counts.index, counts.values,
                  color=[colors.get(d, "#999999") for d in counts.index])
    ax.set_ylabel("patents")
    ax.set_title("Patentability outcome distribution\n(supervised label for downstream tasks)")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v/len(df)*100:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, counts.max() * 1.15)
    _save(fig, "fig03_decision_distribution.png")


def fig_decision_by_year(df):
    ct = pd.crosstab(df["filing_year"], df["decision"], normalize="index") * 100
    order = [c for c in ["ACCEPTED", "REJECTED", "PENDING"] if c in ct.columns]
    ct = ct[order]
    colors = {"ACCEPTED": "#55A868", "REJECTED": "#C44E52", "PENDING": "#8172B3"}
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(ct))
    for d in order:
        ax.bar(ct.index.astype(int).astype(str), ct[d], bottom=bottom,
               label=d, color=colors.get(d))
        bottom += ct[d].values
    ax.set_ylabel("share of filings (%)")
    ax.set_xlabel("filing year")
    ax.set_title("Outcome composition by filing year\n"
                 "(recent years skew PENDING -- prosecution not yet finished)")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    _save(fig, "fig04_decision_by_year.png")


# ---------------------------------------------------------------------------
# Numeric summaries (saved to stats/)
# ---------------------------------------------------------------------------
def write_stats(df):
    rows = []
    for f in TEXT_FIELDS:
        w, c = df[f"{f}_words"], df[f"{f}_chars"]
        nz = w[w > 0]
        rows.append({
            "field": f,
            "pct_non_empty": round((w > 0).mean() * 100, 1),
            "mean_words": round(w.mean(), 1),
            "median_words": int(w.median()),
            "std_words": round(w.std(), 1),
            "p90_words": int(nz.quantile(0.90)) if len(nz) else 0,
            "max_words": int(w.max()),
            "mean_chars": round(c.mean(), 1),
            "median_chars": int(c.median()),
        })
    length_df = pd.DataFrame(rows)
    length_df.to_csv(STATS_DIR / "text_length_stats.csv", index=False)
    print(f"  wrote summary/stats/text_length_stats.csv")

    summary = {
        "dataset": "HUPD cross-year sample (140 apps/year, 2004-2018)",
        "n_documents": int(len(df)),
        "filing_year_range": [int(df["filing_year"].min()), int(df["filing_year"].max())],
        "n_filing_years": int(df["filing_year"].nunique()),
        "decision_counts": df["decision"].value_counts().to_dict(),
        "decision_pct": (df["decision"].value_counts(normalize=True) * 100).round(1).to_dict(),
        "ipc": {
            "coverage_pct": round(df["ipc_section"].notna().mean() * 100, 1),
            "n_sections_present": int(df["ipc_section"].nunique()),
            "n_distinct_subclasses": int(df.loc[df["ipc_subclass"].str.len() >= 4, "ipc_subclass"].nunique()),
            "section_counts": df["ipc_section"].value_counts().sort_index().to_dict(),
        },
        "cpc": {
            "coverage_pct": round(df["cpc_section"].notna().mean() * 100, 1),
            "n_sections_present": int(df["cpc_section"].nunique()),
            "note": "CPC adopted by USPTO ~2013; older filings carry IPC only.",
        },
        "n_unique_examiners": int(df["examiner_id"].replace("", np.nan).nunique()),
        "inventors_per_patent_mean": round(df["n_inventors"].mean(), 2),
        "ipc_labels_per_patent_mean": round(df["n_ipc_labels"].mean(), 2),
        "total_words_full_description": int(df["full_description_words"].sum()),
    }
    with open(STATS_DIR / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  wrote summary/stats/summary.json")
    return length_df, summary


def main():
    print(f"Loading {DATA_FILE} ...")
    df = load()
    print(f"  {len(df):,} documents, {df['filing_year'].nunique()} filing years\n")

    print("Figures:")
    fig_ipc_cpc_sections(df)
    fig_top_ipc_subclasses(df)
    fig_decision(df)
    fig_decision_by_year(df)

    print("\nStats:")
    length_df, summary = write_stats(df)

    print("\n=== Text length summary (words) ===")
    print(length_df.to_string(index=False))
    print("\n=== Dataset summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
