"""
Comparison across BM25, MiniLM, and BGE results, organized by test.
Reads from rag-testing/results/*.json — run the retrieval scripts first.
Output saved to results/eval.txt.
"""

import json
import glob
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
SYSTEMS = ["bm25", "minilm", "bge"]
TESTS = {
    "test1a": "Patent abstract (biotech)",
    "test1b": "Patent abstract (software)",
    "test1c": "Patent abstract (semiconductors)",
    "test2a": "Product desc — short",
    "test2b": "Product desc — medium",
    "test2c": "Product desc — long",
    "test3a": "Plain language — short",
    "test3b": "Plain language — medium",
    "test3c": "Plain language — long",
}


def load_results():
    data = {}
    for path in glob.glob(str(RESULTS_DIR / "*.json")):
        if "embeddings" in path or "metadata" in path:
            continue
        with open(path) as f:
            r = json.load(f)
        system = r["system"]
        key = Path(r["query_file"]).stem.split("_")[0]
        data[(system, key)] = r
    return data


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


data = load_results()
out = []

for key, desc in TESTS.items():
    out.append(f"{key} — {desc}")

    # query time per system
    for sys in SYSTEMS:
        r = data.get((sys, key))
        if r:
            out.append(f"  {sys} query time: {r['query_time_ms']:.2f}ms")

    out.append("")

    # rank 1 per system
    for sys in SYSTEMS:
        r = data.get((sys, key))
        if r:
            top = r["results"][0]
            out.append(f"  {sys} rank 1: {top['title']}")
            out.append(f"    IPC: {', '.join(top['ipc']) if top['ipc'] else 'N/A'}")

    out.append("")

    # top-10 lens IDs for jaccard (only for test groups 2 and 3)
    group = key[4]  # '1', '2', or '3'
    variant = key[5]  # 'a', 'b', 'c'
    if group in ("2", "3") and variant == "c":
        # report short↔long overlap for this group
        out.append(f"  short vs long overlap (jaccard, top-10):")
        for sys in SYSTEMS:
            ra = data.get((sys, f"test{group}a"))
            rc = data.get((sys, f"test{group}c"))
            if ra and rc:
                ids_a = set(x["lens_id"] for x in ra["results"])
                ids_c = set(x["lens_id"] for x in rc["results"])
                out.append(f"    {sys}: {jaccard(ids_a, ids_c):.2f}")
        out.append("")

    out.append("-" * 50)
    out.append("")

output = "\n".join(out)
print(output)

out_path = RESULTS_DIR / "eval.txt"
out_path.write_text(output)
print(f"saved to {out_path}")
