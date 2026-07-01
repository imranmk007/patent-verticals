"""
BM25 patent retrieval over lens-db-translated.json.
Indexes title + abstract.

To run: set QUERY_FILE below and hit Run.
"""

import json
import re
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

# ── Configure here ────────────────────────────────────────────────────────────

QUERY_FILE = Path(__file__).parent / "test3c_long.txt"   
TOP_K      = 10


CORPUS_PATH = Path(__file__).parent.parent / "initial-datasets/lens-db/final-data/lens-db-translated.json"

_stop_words = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "each", "than", "such", "that",
    "this", "these", "those", "which", "who", "whom", "what", "where", "when",
    "why", "how", "all", "any", "few", "more", "most", "other", "some", "into",
    "through", "during", "before", "after", "above", "between", "out", "its",
    "it", "as", "also", "up", "their", "they", "them", "we", "our", "us",
    "you", "your", "he", "she", "his", "her", "my", "me", "i", "about",
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in _stop_words and len(t) > 1]


def load_corpus(path: Path) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)

    corpus = []
    for doc in raw:
        title    = (doc.get("title")    or "").strip()
        abstract = (doc.get("abstract") or "").strip()
        text = f"{title} {abstract}".strip()
        if not text:
            continue
        corpus.append({
            "lens_id":  doc.get("lens_id", ""),
            "title":    title,
            "abstract": abstract,
            "ipc":      doc.get("ipc_classifications", []),
            "text":     text,
        })
    return corpus


def build_index(corpus: list[dict]) -> tuple[BM25Okapi, float]:
    t0 = time.perf_counter()
    tokenized = [tokenize(doc["text"]) for doc in corpus]
    index = BM25Okapi(tokenized)
    elapsed = time.perf_counter() - t0
    return index, elapsed


def query_index(index: BM25Okapi, query_text: str, corpus: list[dict], top_k: int) -> tuple[list[dict], float]:
    tokens = tokenize(query_text)
    t0 = time.perf_counter()
    scores = index.get_scores(tokens)
    elapsed = time.perf_counter() - t0

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for rank, (idx, score) in enumerate(ranked, start=1):
        results.append({
            "rank":     rank,
            "lens_id":  corpus[idx]["lens_id"],
            "title":    corpus[idx]["title"],
            "abstract": corpus[idx]["abstract"][:300],
            "ipc":      corpus[idx]["ipc"],
            "score":    round(float(score), 4),
        })
    return results, elapsed


def print_results(results: list[dict], query_file: str, index_time: float, query_time: float) -> None:
    print(f"\n{'='*70}")
    print(f"  BM25 Retrieval Results")
    print(f"  Query       : {query_file}")
    print(f"  Index build : {index_time*1000:.1f} ms")
    print(f"  Query time  : {query_time*1000:.2f} ms")
    print(f"{'='*70}\n")

    for r in results:
        print(f"Rank {r['rank']:>2}  |  score: {r['score']:.4f}  |  {r['lens_id']}")
        print(f"  Title   : {r['title']}")
        print(f"  IPC     : {', '.join(r['ipc']) if r['ipc'] else 'N/A'}")
        print(f"  Abstract: {r['abstract']}{'...' if len(r['abstract']) == 300 else ''}")
        print()


def save_results(results: list[dict], query_file: str, index_time: float, query_time: float) -> Path:
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    output = {
        "system":      "bm25",
        "query_file":  query_file,
        "index_time_ms": round(index_time * 1000, 1),
        "query_time_ms": round(query_time * 1000, 2),
        "top_k":       len(results),
        "results":     results,
    }

    out_path = output_dir / f"bm25_{Path(query_file).stem}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    return out_path


if __name__ == "__main__":
    query_text = QUERY_FILE.read_text().strip()

    print(f"Loading corpus...")
    corpus = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(corpus)} patents")

    print("Building BM25 index...")
    index, index_time = build_index(corpus)

    results, query_time = query_index(index, query_text, corpus, TOP_K)
    print_results(results, QUERY_FILE.name, index_time, query_time)

    out_path = save_results(results, QUERY_FILE.name, index_time, query_time)
    print(f"Results saved to {out_path}")
