"""
MiniLM dense retrieval over lens-db-translated.json.
Indexes title + abstract. Embeddings cached to disk after first run.

To run: set QUERY_FILE below and hit Run.
First run downloads the model (~80MB) and builds embeddings (~2-3 min).
Subsequent runs load from cache and are fast.
"""

import json
import time
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer

# ── Configure here ────────────────────────────────────────────────────────────

QUERY_FILE = Path(__file__).parent / "test2a_short.txt"   # ← change this
TOP_K      = 10

# test1a_biotech.txt      | test1b_software.txt     | test1c_semiconductors.txt
# test2a_short.txt        | test2b_medium.txt        | test2c_long.txt
# test3a_short.txt        | test3b_medium.txt        | test3c_long.txt

# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME   = "all-MiniLM-L6-v2"
CORPUS_PATH  = Path(__file__).parent.parent / "initial-datasets/lens-db/final-data/lens-db-translated.json"
CACHE_DIR    = Path(__file__).parent / "results"
EMBEDDINGS_CACHE = CACHE_DIR / "minilm_embeddings.npy"
METADATA_CACHE   = CACHE_DIR / "minilm_metadata.json"


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


def build_or_load_index(corpus: list[dict], model: SentenceTransformer) -> tuple[np.ndarray, float]:
    if EMBEDDINGS_CACHE.exists() and METADATA_CACHE.exists():
        print("Loading cached embeddings from disk...")
        t0 = time.perf_counter()
        embeddings = np.load(EMBEDDINGS_CACHE)
        elapsed = time.perf_counter() - t0
        print(f"Loaded {len(embeddings)} embeddings in {elapsed*1000:.1f} ms")
        return embeddings, elapsed

    print(f"Building embeddings for {len(corpus)} patents (first run only)...")
    texts = [doc["text"] for doc in corpus]
    t0 = time.perf_counter()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    elapsed = time.perf_counter() - t0

    CACHE_DIR.mkdir(exist_ok=True)
    np.save(EMBEDDINGS_CACHE, embeddings)
    metadata = [{"lens_id": d["lens_id"], "title": d["title"],
                 "abstract": d["abstract"], "ipc": d["ipc"]} for d in corpus]
    with open(METADATA_CACHE, "w") as f:
        json.dump(metadata, f)
    print(f"Embeddings saved to cache ({elapsed:.1f}s)")
    return embeddings, elapsed


def query_index(model: SentenceTransformer, query_text: str, embeddings: np.ndarray,
                corpus: list[dict], top_k: int) -> tuple[list[dict], float]:
    query_vec = model.encode([query_text], convert_to_numpy=True)[0]

    t0 = time.perf_counter()
    # cosine similarity: dot product on unit-normalised vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.clip(norms, 1e-10, None)
    query_normed = query_vec / np.linalg.norm(query_vec)
    scores = normed @ query_normed
    elapsed = time.perf_counter() - t0

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append({
            "rank":     rank,
            "lens_id":  corpus[idx]["lens_id"],
            "title":    corpus[idx]["title"],
            "abstract": corpus[idx]["abstract"][:300],
            "ipc":      corpus[idx]["ipc"],
            "score":    round(float(scores[idx]), 4),
        })
    return results, elapsed


def print_results(results: list[dict], query_file: str, index_time: float, query_time: float) -> None:
    print(f"\n{'='*70}")
    print(f"  MiniLM Dense Retrieval Results")
    print(f"  Query       : {query_file}")
    print(f"  Index load  : {index_time*1000:.1f} ms")
    print(f"  Query time  : {query_time*1000:.2f} ms")
    print(f"{'='*70}\n")

    for r in results:
        print(f"Rank {r['rank']:>2}  |  score: {r['score']:.4f}  |  {r['lens_id']}")
        print(f"  Title   : {r['title']}")
        print(f"  IPC     : {', '.join(r['ipc']) if r['ipc'] else 'N/A'}")
        print(f"  Abstract: {r['abstract']}{'...' if len(r['abstract']) == 300 else ''}")
        print()


def save_results(results: list[dict], query_file: str, index_time: float, query_time: float) -> Path:
    output = {
        "system":        "minilm",
        "query_file":    query_file,
        "index_time_ms": round(index_time * 1000, 1),
        "query_time_ms": round(query_time * 1000, 2),
        "top_k":         len(results),
        "results":       results,
    }
    out_path = CACHE_DIR / f"minilm_{Path(query_file).stem}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    return out_path


if __name__ == "__main__":
    query_text = QUERY_FILE.read_text().strip()

    print(f"Loading corpus...")
    corpus = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(corpus)} patents")

    print(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    embeddings, index_time = build_or_load_index(corpus, model)

    results, query_time = query_index(model, query_text, embeddings, corpus, TOP_K)
    print_results(results, QUERY_FILE.name, index_time, query_time)

    out_path = save_results(results, QUERY_FILE.name, index_time, query_time)
    print(f"Results saved to {out_path}")
