import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

EMBED_MODEL = "gemini-embedding-2"
EMBED_DIM = 1536
DOC_TEMPLATE = "title: {title} | text: {content}"

MAX_RPM = int(os.environ.get("GEMINI_MAX_RPM", "90"))
MAX_TPM = int(os.environ.get("GEMINI_MAX_TPM", "28000"))
RATE_WINDOW = 60.0
CHARS_PER_TOKEN = 4
EMBED_ATTEMPTS = 5
RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
REQUEST_TIMEOUT_MS = 120000

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS_JSON = BASE_DIR / "arxiv-abstracts-test.json"
DEFAULT_OUTPUT = BASE_DIR / "arxiv-abstracts-test-embeddings.jsonl"

API_KEY_ENV = "GEMINI_API_KEY"
LOCAL_API_KEY = ""


def get_api_key():
    key = os.environ.get(API_KEY_ENV, "").strip() or LOCAL_API_KEY
    if not key:
        raise SystemExit(f"set {API_KEY_ENV} in your environment first")
    return key


class RateLimiter:
    def __init__(self, rpm, tpm, window=RATE_WINDOW):
        self.rpm = rpm
        self.tpm = tpm
        self.window = window
        self.reqs = deque()
        self.toks = deque()
        self.tok_sum = 0

    def _prune(self, now):
        cutoff = now - self.window
        while self.reqs and self.reqs[0] <= cutoff:
            self.reqs.popleft()
        while self.toks and self.toks[0][0] <= cutoff:
            self.tok_sum -= self.toks.popleft()[1]

    def acquire(self, n_tokens):
        n_tokens = min(n_tokens, self.tpm) if self.tpm else n_tokens
        while True:
            now = time.monotonic()
            self._prune(now)
            wait = 0.0
            if self.rpm and len(self.reqs) >= self.rpm:
                wait = max(wait, self.window - (now - self.reqs[0]))
            if self.tpm and self.toks and self.tok_sum + n_tokens > self.tpm:
                wait = max(wait, self.window - (now - self.toks[0][0]))
            if wait <= 0:
                break
            logging.info(f"pacing requests, waiting {wait:.1f}s")
            time.sleep(min(wait, self.window))
        stamp = time.monotonic()
        self.reqs.append(stamp)
        self.toks.append((stamp, n_tokens))
        self.tok_sum += n_tokens


def embed_one(client, limiter, content):
    est = len(content) // CHARS_PER_TOKEN + 8
    limiter.acquire(est)
    cfg = types.EmbedContentConfig(output_dimensionality=EMBED_DIM)
    delay = 2.0
    for attempt in range(1, EMBED_ATTEMPTS + 1):
        try:
            resp = client.models.embed_content(
                model=EMBED_MODEL,
                contents=[types.Content(parts=[types.Part.from_text(text=content)])],
                config=cfg,
            )
            embeds = resp.embeddings or []
            if not embeds or not embeds[0].values:
                raise RuntimeError("empty embedding back from api")
            return list(embeds[0].values)
        except genai_errors.APIError as e:
            if e.code in RETRYABLE_CODES and attempt < EMBED_ATTEMPTS:
                logging.warning(f"got {e.code}, retry {attempt}/{EMBED_ATTEMPTS} in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 30)
            else:
                raise RuntimeError(f"embed failed ({e.code}): {e.message}")
        except Exception as e:
            if attempt < EMBED_ATTEMPTS:
                logging.warning(f"embed error, retry in {delay:.0f}s: {e}")
                time.sleep(delay)
                delay = min(delay * 2, 30)
            else:
                raise RuntimeError(f"embed failed: {e}")
    raise RuntimeError("embed failed, retries exhausted")


def get_parser():
    p = argparse.ArgumentParser(description="embed the arxiv abstract corpus with gemini-embedding-2")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_JSON)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    for noisy in ("httpx", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    args = get_parser().parse_args(argv)

    papers = json.loads(args.corpus.read_text(encoding="utf-8"))
    logging.info(f"loaded {len(papers)} papers from {args.corpus}")

    client = genai.Client(api_key=get_api_key(), http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
    limiter = RateLimiter(MAX_RPM, MAX_TPM)

    n_written = 0
    with args.output.open("w", encoding="utf-8") as sink:
        for i, paper in enumerate(papers, 1):
            title = paper.get("title") or ""
            abstract = paper.get("abstract") or ""
            if not abstract:
                logging.warning(f"{paper.get('id')}: no abstract, skipping")
                continue
            content = DOC_TEMPLATE.format(title=title, content=abstract)
            try:
                vector = embed_one(client, limiter, content)
            except RuntimeError as e:
                logging.warning(f"{paper.get('id')}: {e}")
                continue
            record = {
                "id": paper.get("id"),
                "source_file": args.corpus.name,
                "doc_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "mime_type": "text/plain",
                "category": "text",
                "chunk_index": 0,
                "chunk_count": 1,
                "chunk_meta": {},
                "text": None,
                "embedding_model": EMBED_MODEL,
                "embedding_dim": EMBED_DIM,
                "embedding": vector,
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1
            if i % 25 == 0 or i == len(papers):
                logging.info(f"embedded {i}/{len(papers)}")

    logging.info(f"wrote {n_written} records to {args.output}")
    return 0 if n_written else 1


if __name__ == "__main__":
    sys.exit(main())
