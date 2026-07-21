import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_INPUT_EMBEDDINGS = Path(__file__).parent / "embeddings" / "input-embeddings.jsonl"
DEFAULT_CORPUS_EMBEDDINGS = Path(__file__).parent / "arxiv-abstracts-test-embeddings.jsonl"
DEFAULT_CORPUS_METADATA = Path(__file__).parent / "arxiv-abstracts-test.json"
DEFAULT_OUTPUT = Path(__file__).parent / "similarity-results.json"

TOP_K_MIN = 1
TOP_K_MAX = 50
TOP_K_DEFAULT = 3
CHUNK_TOPK_PROPORTION = 0.2
CHUNK_TOPK_MIN = 2
CHUNK_TOPK_MAX = 100


def read_jsonl(path):
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def norm_rows(m):
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def clamp_k(k):
    return int(min(TOP_K_MAX, max(TOP_K_MIN, k)))


def chunk_topk(scores):
    n = scores.shape[0]
    if n == 1:
        return scores[0], 1
    k = int(np.clip(round(CHUNK_TOPK_PROPORTION * n), CHUNK_TOPK_MIN, CHUNK_TOPK_MAX))
    k = min(k, n)
    top = np.sort(scores, axis=0)[::-1]
    return top[:k].mean(axis=0), k


def group_by_doc(records):
    groups = defaultdict(list)
    for r in records:
        groups[r["id"]].append(r)
    doc_ids = list(groups.keys())
    matrices = [np.array([r["embedding"] for r in groups[d]], dtype=np.float32) for d in doc_ids]
    return doc_ids, matrices


def corpus_scores(input_matrix, doc_matrices):
    cols = []
    for dm in doc_matrices:
        s = input_matrix @ dm.T
        cols.append(s.max(axis=1))
    return np.stack(cols, axis=1)


def load_meta(path):
    docs = json.loads(path.read_text(encoding="utf-8"))
    return {d["id"]: d for d in docs if d.get("id")}


def make_result(rank, doc_id, score, meta):
    m = meta.get(doc_id, {})
    res = {"rank": rank, "score": round(float(score), 4), "id": doc_id}
    for k in ("title", "authors", "categories", "abstract"):
        if k in m:
            res[k] = m[k]
    return res


def get_parser():
    p = argparse.ArgumentParser(description="rank arxiv abstracts against embedded input chunks")
    p.add_argument("--input-embeddings", type=Path, default=DEFAULT_INPUT_EMBEDDINGS)
    p.add_argument("--corpus-embeddings", type=Path, default=DEFAULT_CORPUS_EMBEDDINGS)
    p.add_argument("--corpus-metadata", type=Path, default=DEFAULT_CORPUS_METADATA)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = get_parser().parse_args(argv)

    if not args.input_embeddings.is_file():
        logging.error(f"missing input embeddings: {args.input_embeddings}")
        return 2
    if not args.corpus_embeddings.is_file():
        logging.error(f"missing corpus embeddings: {args.corpus_embeddings}")
        return 2
    if not args.corpus_metadata.is_file():
        logging.error(f"missing corpus metadata: {args.corpus_metadata}")
        return 2

    input_records = [r for r in read_jsonl(args.input_embeddings) if r.get("embedding")]
    corpus_records = [r for r in read_jsonl(args.corpus_embeddings) if r.get("embedding")]
    if not input_records:
        logging.error("no input chunks have embeddings")
        return 1
    if not corpus_records:
        logging.error("no corpus chunks have embeddings")
        return 1

    in_dim = len(input_records[0]["embedding"])
    corpus_dim = len(corpus_records[0]["embedding"])
    if in_dim != corpus_dim:
        logging.error(f"dim mismatch: input={in_dim} corpus={corpus_dim}")
        return 2

    top_k = clamp_k(args.top_k)
    input_matrix = norm_rows(np.array([r["embedding"] for r in input_records], dtype=np.float32))
    doc_ids, doc_matrices = group_by_doc(corpus_records)
    doc_matrices = [norm_rows(m) for m in doc_matrices]

    scores = corpus_scores(input_matrix, doc_matrices)
    doc_scores, chunk_k_used = chunk_topk(scores)

    top_k = min(top_k, len(doc_ids))
    top_idx = np.argsort(doc_scores)[::-1][:top_k]

    meta = load_meta(args.corpus_metadata)
    results = [make_result(rank, doc_ids[i], doc_scores[i], meta) for rank, i in enumerate(top_idx, start=1)]

    run_info = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_chunks": len(input_records),
        "chunked_input": len(input_records) > 1,
        "chunk_top_k_used": chunk_k_used,
        "corpus_documents": len(doc_ids),
        "top_k_requested": args.top_k,
        "top_k_used": top_k,
    }
    output = {**run_info, "results": results}
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    logging.info(f"corpus={len(doc_ids)} docs, input_chunks={len(input_records)} (chunk_top_k={chunk_k_used}), top_k={top_k}")
    for r in results:
        logging.info(f"rank {r['rank']}  score {r['score']:.4f}  [{r['id']}]  {r.get('title', '')[:80]}")
    logging.info(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
