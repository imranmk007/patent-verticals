import json
import time
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer



QUERY_FILE = Path(__file__).parent / "inputs-clean/test1a_biotech.json"  # <- change this
ALPHA      = 0.7    # weight on the (normalized) cosine score; (1-ALPHA) on IPC
TOP_K      = 10
MODEL_NAME       = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX     = "Represent this sentence for searching relevant passages: "
# Embedding cache is shared with bge_retrieval.py and lives one level up.
CACHE_DIR        = Path(__file__).parent.parent / "results/bge"
EMBEDDINGS_CACHE = CACHE_DIR / "bge_embeddings.npy"
METADATA_CACHE   = CACHE_DIR / "bge_metadata.json"
# Outputs stay inside code-weight/ so this folder is self-contained.
RESULTS_DIR      = Path(__file__).parent / "results"

# IPC-taxonomy match score, by deepest matching level (per the spec).
SUBGROUP, MAIN_GROUP, SUBCLASS, CLASS, SECTION = 1.0, 0.8, 0.5, 0.2, 0.0


def parse_ipc(code: str) -> dict | None:
    """Split an IPC code like 'C07H21/04' into its hierarchical levels.

        section    = C
        class      = C07
        subclass   = C07H
        main_group = C07H21      (the part before '/')
        subgroup   = C07H21/04   (the full code)
    """
    code = code.strip().replace(" ", "")
    if len(code) < 4 or "/" not in code:
        # Not a well-formed full IPC symbol; treat as unusable.
        if len(code) < 4:
            return None
    section = code[0]
    klass   = code[:3]
    subclass = code[:4]
    main_group = code.split("/")[0]
    return {
        "section": section,
        "class": klass,
        "subclass": subclass,
        "main_group": main_group,
        "subgroup": code,
    }


def pair_score(a: str, b: str) -> float:
    """Taxonomy similarity between two IPC codes: deepest matching level wins."""
    pa, pb = parse_ipc(a), parse_ipc(b)
    if pa is None or pb is None:
        return 0.0
    if pa["subgroup"] == pb["subgroup"]:
        return SUBGROUP
    if pa["main_group"] == pb["main_group"]:
        return MAIN_GROUP
    if pa["subclass"] == pb["subclass"]:
        return SUBCLASS
    if pa["class"] == pb["class"]:
        return CLASS
    if pa["section"] == pb["section"]:
        # Same section but nothing deeper matches -> still essentially unrelated.
        return SECTION
    return SECTION


def max_ipc_score(query_codes: list[str], doc_codes: list[str]) -> float:
    """Max taxonomy score over all query<->doc code pairs (per the spec)."""
    best = 0.0
    for q in query_codes:
        for d in doc_codes:
            s = pair_score(q, d)
            if s > best:
                best = s
                if best == SUBGROUP:
                    return best
    return best


def load_cache() -> tuple[np.ndarray, list[dict]]:
    if not (EMBEDDINGS_CACHE.exists() and METADATA_CACHE.exists()):
        raise FileNotFoundError(
            f"Embedding cache not found in {CACHE_DIR}. "
            "Run bge_retrieval.py first to build it."
        )
    embeddings = np.load(EMBEDDINGS_CACHE)
    with open(METADATA_CACHE) as f:
        metadata = json.load(f)
    assert len(embeddings) == len(metadata), "cache/metadata length mismatch"
    return embeddings, metadata


def query_index(model, query_text, query_codes, embeddings, metadata, alpha, top_k):
    prefixed = QUERY_PREFIX + query_text
    query_vec = model.encode([prefixed], convert_to_numpy=True, normalize_embeddings=True)[0]

    t0 = time.perf_counter()
    cos = embeddings @ query_vec   # cosine sim (embeddings are L2-normalized)

    # Min-max normalize cosine across the whole corpus so it shares the 0-1
    # range with the IPC score (otherwise the compressed cosine band is unfairly
    # dominated by IPC during fusion).
    cmin, cmax = cos.min(), cos.max()
    cos_norm = (cos - cmin) / (cmax - cmin) if cmax > cmin else np.zeros_like(cos)

    # Per-doc IPC score (max over code pairs). No query codes or no doc codes
    # -> fall back to cosine-only for that doc (IPC term contributes nothing).
    n = len(metadata)
    ipc = np.zeros(n, dtype=np.float32)
    has_ipc = np.zeros(n, dtype=bool)
    if query_codes:
        for i, m in enumerate(metadata):
            doc_codes = m.get("ipc") or []
            if doc_codes:
                ipc[i] = max_ipc_score(query_codes, doc_codes)
                has_ipc[i] = True

    # Fuse. Where IPC is unavailable, use the normalized cosine alone.
    final = np.where(
        has_ipc,
        alpha * cos_norm + (1.0 - alpha) * ipc,
        cos_norm,
    )
    elapsed = time.perf_counter() - t0

    top = np.argsort(final)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top, start=1):
        results.append({
            "rank":       rank,
            "lens_id":    metadata[idx]["lens_id"],
            "title":      metadata[idx]["title"],
            "abstract":   metadata[idx]["abstract"][:300],
            "ipc":        metadata[idx].get("ipc", []),
            "cos_raw":    round(float(cos[idx]), 4),
            "cos_norm":   round(float(cos_norm[idx]), 4),
            "ipc_score":  round(float(ipc[idx]), 4) if has_ipc[idx] else None,
            "final":      round(float(final[idx]), 4),
        })
    return results, elapsed


def print_results(results, query_file, query_codes, alpha, query_time):
    print(f"\n{'='*78}")
    print(f"  IPC-weighted Retrieval  (BGE-small + IPC taxonomy fusion)")
    print(f"  Query       : {query_file}")
    print(f"  Query IPC   : {', '.join(query_codes) if query_codes else 'NONE (cosine-only)'}")
    print(f"  Alpha       : {alpha}  (cosine weight; IPC weight = {round(1-alpha,2)})")
    print(f"  Score time  : {query_time*1000:.2f} ms")
    print(f"{'='*78}\n")
    for r in results:
        ipc_s = "n/a" if r["ipc_score"] is None else f"{r['ipc_score']:.2f}"
        print(f"Rank {r['rank']:>2} | final {r['final']:.4f} "
              f"(cos_norm {r['cos_norm']:.3f}, ipc {ipc_s}) | {r['lens_id']}")
        print(f"   Title : {r['title']}")
        print(f"   IPC   : {', '.join(r['ipc']) if r['ipc'] else 'N/A'}")
        print()


def save_results(results, query_file, query_codes, alpha, query_time):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    variant = "dirty" if "inputs-dirty" in str(query_file) else "clean"
    output = {
        "system":        "ipc-weighted-bge",
        "query_file":    str(query_file),
        "variant":       variant,
        "query_ipc":     query_codes,
        "alpha":         alpha,
        "query_time_ms": round(query_time * 1000, 2),
        "top_k":         len(results),
        "results":       results,
    }
    out_path = RESULTS_DIR / f"ipc_{variant}_{Path(query_file).stem}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    return out_path


if __name__ == "__main__":
    spec = json.loads(QUERY_FILE.read_text())
    query_text  = spec["document"].strip()
    query_codes = spec.get("ipc_classifications", []) or []

    print(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading cached embeddings + metadata...")
    embeddings, metadata = load_cache()
    print(f"Loaded {len(embeddings)} embeddings")

    results, query_time = query_index(
        model, query_text, query_codes, embeddings, metadata, ALPHA, TOP_K
    )
    print_results(results, QUERY_FILE.name, query_codes, ALPHA, query_time)

    out_path = save_results(results, QUERY_FILE, query_codes, ALPHA, query_time)
    print(f"Results saved to {out_path}")
