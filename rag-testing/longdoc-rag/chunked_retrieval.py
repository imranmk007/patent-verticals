"""
Chunked BGE-small-en-v1.5 retrieval for long-document queries over lens-db-translated.json.

The corpus side is unchanged from standard-rag/bge_retrieval.py: each patent is
indexed as a single title+abstract embedding. The query side is what's new --
a long input document (too long for one embedding pass) is split into
overlapping token-window chunks, each chunk is embedded and scored against the
whole corpus independently, and the per-chunk scores for each corpus document
are rolled up into one final score via a top-k average (k scales with the
number of chunks, so noisy/irrelevant chunks in a very long document don't
dilute a real match, but a handful of strong hits still count as evidence).

To run: set QUERY_FILE below to a plain-text file and hit Run.
First run downloads the model (~130MB) and builds the corpus embeddings
(~2-3 min). Subsequent runs load the corpus embeddings from cache.
"""

import json
import time
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer

# -- Configure here ------------------------------------------------------------

QUERY_FILE = Path(__file__).parent / "input-document.txt"  # <- change this
TOP_K      = 10   # number of ranked corpus documents to return

CHUNK_SIZE_TOKENS  = 300    # tokens per chunk (BGE-small's max sequence length is 512)
CHUNK_OVERLAP_PCT  = 0.15   # fraction of chunk_size shared between consecutive chunks

TOPK_PROPORTION = 0.1   # fraction of a document's chunks averaged for its final score
TOPK_MIN        = 2     # floor, so short queries don't over-dilute
TOPK_MAX        = 40    # cap, so very long queries don't wash out into a corpus-wide mean

# ------------------------------------------------------------------------------

MODEL_NAME       = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX     = "Represent this sentence for searching relevant passages: "
CORPUS_PATH      = Path(__file__).parent / "lens-db-translated.json"
CACHE_DIR        = Path(__file__).parent / "results"
EMBEDDINGS_CACHE = CACHE_DIR / "bge_embeddings.npy"
METADATA_CACHE   = CACHE_DIR / "bge_metadata.json"


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
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    elapsed = time.perf_counter() - t0

    CACHE_DIR.mkdir(exist_ok=True)
    np.save(EMBEDDINGS_CACHE, embeddings)
    metadata = [{"lens_id": d["lens_id"], "title": d["title"],
                 "abstract": d["abstract"], "ipc": d["ipc"]} for d in corpus]
    with open(METADATA_CACHE, "w") as f:
        json.dump(metadata, f)
    print(f"Embeddings saved to cache ({elapsed:.1f}s)")
    return embeddings, elapsed


def chunk_text(text: str, model: SentenceTransformer, chunk_size: int, overlap_pct: float) -> list[str]:
    """Split text into overlapping token-window chunks using the model's own tokenizer,
    so chunk boundaries line up with what the model will actually see.
    """
    tokenizer = model.tokenizer
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    stride = max(1, round(chunk_size * (1 - overlap_pct)))

    chunks = []
    start = 0
    n = len(token_ids)
    while start < n:
        window = token_ids[start:start + chunk_size]
        chunks.append(tokenizer.decode(window))
        if start + chunk_size >= n:
            break
        start += stride
    return chunks


def embed_chunks(model: SentenceTransformer, chunks: list[str]) -> np.ndarray:
    prefixed = [QUERY_PREFIX + c for c in chunks]
    return model.encode(prefixed, convert_to_numpy=True, normalize_embeddings=True)


def aggregate_topk(scores: np.ndarray, proportion: float, k_min: int, k_max: int) -> tuple[np.ndarray, int]:
    """scores: (num_chunks, num_corpus_docs) cosine similarities.
    Returns (per-doc score, k used), where per-doc score is the mean of each
    document's top-k chunk scores.
    """
    num_chunks = scores.shape[0]
    k = int(np.clip(round(proportion * num_chunks), k_min, k_max))
    k = min(k, num_chunks)
    sorted_scores = np.sort(scores, axis=0)[::-1]  # descending, per column
    doc_scores = sorted_scores[:k].mean(axis=0)
    return doc_scores, k


def query_index(model: SentenceTransformer, query_text: str, corpus_embeddings: np.ndarray,
                 corpus: list[dict], top_k: int) -> tuple[list[dict], dict, float]:
    t0 = time.perf_counter()

    chunks = chunk_text(query_text, model, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_PCT)
    chunk_embeddings = embed_chunks(model, chunks)

    # embeddings already normalized -> dot product == cosine similarity
    scores = chunk_embeddings @ corpus_embeddings.T  # (num_chunks, num_corpus_docs)
    doc_scores, k_used = aggregate_topk(scores, TOPK_PROPORTION, TOPK_MIN, TOPK_MAX)

    elapsed = time.perf_counter() - t0

    top_indices = np.argsort(doc_scores)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append({
            "rank":     rank,
            "lens_id":  corpus[idx]["lens_id"],
            "title":    corpus[idx]["title"],
            "abstract": corpus[idx]["abstract"][:300],
            "ipc":      corpus[idx]["ipc"],
            "score":    round(float(doc_scores[idx]), 4),
        })

    query_meta = {
        "num_chunks":        len(chunks),
        "chunk_size_tokens": CHUNK_SIZE_TOKENS,
        "chunk_overlap_pct": CHUNK_OVERLAP_PCT,
        "k_used":            k_used,
    }
    return results, query_meta, elapsed


def print_results(results: list[dict], query_file: str, query_meta: dict,
                   index_time: float, query_time: float) -> None:
    print(f"\n{'='*70}")
    print(f"  Chunked BGE-small-en-v1.5 Retrieval Results")
    print(f"  Query       : {query_file}")
    print(f"  Chunks      : {query_meta['num_chunks']} "
          f"(size {query_meta['chunk_size_tokens']} tok, "
          f"{query_meta['chunk_overlap_pct']*100:.0f}% overlap)")
    print(f"  Top-k avg   : k = {query_meta['k_used']}")
    print(f"  Index load  : {index_time*1000:.1f} ms")
    print(f"  Query time  : {query_time*1000:.2f} ms")
    print(f"{'='*70}\n")

    for r in results:
        print(f"Rank {r['rank']:>2}  |  score: {r['score']:.4f}  |  {r['lens_id']}")
        print(f"  Title   : {r['title']}")
        print(f"  IPC     : {', '.join(r['ipc']) if r['ipc'] else 'N/A'}")
        print(f"  Abstract: {r['abstract']}{'...' if len(r['abstract']) == 300 else ''}")
        print()


def save_results(results: list[dict], query_file: str, query_meta: dict,
                  index_time: float, query_time: float) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    output = {
        "system":        "longdoc-chunked-bge",
        "query_file":    query_file,
        "num_chunks":        query_meta["num_chunks"],
        "chunk_size_tokens": query_meta["chunk_size_tokens"],
        "chunk_overlap_pct": query_meta["chunk_overlap_pct"],
        "k_used":            query_meta["k_used"],
        "index_time_ms": round(index_time * 1000, 1),
        "query_time_ms": round(query_time * 1000, 2),
        "top_k":         len(results),
        "results":       results,
    }
    out_path = CACHE_DIR / f"longdoc_{Path(query_file).stem}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    return out_path


if __name__ == "__main__":
    query_text = QUERY_FILE.read_text().strip()

    print("Loading corpus...")
    corpus = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(corpus)} patents")

    print(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    corpus_embeddings, index_time = build_or_load_index(corpus, model)

    results, query_meta, query_time = query_index(model, query_text, corpus_embeddings, corpus, TOP_K)
    print_results(results, QUERY_FILE.name, query_meta, index_time, query_time)

    out_path = save_results(results, QUERY_FILE.name, query_meta, index_time, query_time)
    print(f"Results saved to {out_path}")
