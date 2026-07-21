import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

QUERY = "1512.03385"

MAX_PAPERS = 200

OUTPUT_PATH = "citation_graph.json"

BASE = "https://api.openalex.org"

MAILTO = "papertrace-app@example.com"

MIN_GAP_S = 0.12
PAGE_SIZE = 200
MAX_RETRIES = 3

FULL_FIELDS = ",".join([
    "id", "title", "authorships", "publication_year", "publication_date",
    "cited_by_count", "referenced_works_count", "doi", "ids",
    "primary_location", "abstract_inverted_index", "referenced_works",
])

MINI_FIELDS = ",".join([
    "id", "title", "authorships", "publication_year", "publication_date",
    "cited_by_count", "referenced_works_count", "doi", "ids",
    "primary_location", "abstract_inverted_index",
])

_cache = {}
_last_request_time = 0.0

def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()

_SSL_CTX = _ssl_context()

def oa_fetch(path):
    global _last_request_time

    url = f"{path}&mailto={MAILTO}" if "?" in path else f"{path}?mailto={MAILTO}"

    if url in _cache:
        return _cache[url]

    for attempt in range(MAX_RETRIES + 1):
        gap = MIN_GAP_S - (time.time() - _last_request_time)
        if gap > 0:
            time.sleep(gap)
        _last_request_time = time.time()

        req = urllib.request.Request(
            url, headers={"User-Agent": f"PaperTrace/1.0 (mailto:{MAILTO})"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as res:
                data = json.loads(res.read().decode("utf-8"))
            _cache[url] = data
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise RuntimeError("Paper not found. Try a DOI or arXiv ID.")
            if e.code == 429 and attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"OpenAlex API error: {e.code}")
        except urllib.error.URLError:
            raise RuntimeError("Network error. Check your connection.")

    raise RuntimeError("RATE_LIMITED")

def short_id(oa_id):
    return (oa_id or "").replace("https://openalex.org/", "")

def abstract_from_index(idx):
    if not idx:
        return None
    try:
        entries = [(pos, word) for word, positions in idx.items() for pos in positions]
        entries.sort()
        return " ".join(word for _, word in entries)
    except Exception:
        return None

def extract_year_from_id(work):
    text = (work.get("doi") or "") + " " + (work.get("id") or "")
    m = re.search(r"arxiv\.(\d{4})\.\d+", text, re.IGNORECASE)
    if m:
        yy = int(m.group(1)[:2])
        return 1900 + yy if yy >= 90 else 2000 + yy
    return None

def map_work(w):
    ids = w.get("ids") or {}
    doi = (w.get("doi") or ids.get("doi") or "").replace("https://doi.org/", "") or None
    arxiv = (ids.get("arxiv") or "").replace("https://arxiv.org/abs/", "") or None
    if not arxiv and doi:
        m = re.match(r"10\.48550/arxiv\.(.+)", doi, re.IGNORECASE)
        if m:
            arxiv = m.group(1)
    pid = short_id(w.get("id"))

    raw_year = w.get("publication_year")
    current_year = time.localtime().tm_year
    if raw_year and raw_year <= current_year:
        year = raw_year
    else:
        year = extract_year_from_id(w) or raw_year

    if doi:
        url = f"https://doi.org/{doi}"
    elif arxiv:
        url = f"https://arxiv.org/abs/{arxiv}"
    else:
        url = f"https://openalex.org/{pid}"

    primary = w.get("primary_location") or {}
    source = primary.get("source") or {}

    return {
        "paperId": pid,
        "title": w.get("title") or "",
        "authors": [
            {"authorId": short_id((a.get("author") or {}).get("id")),
             "name": (a.get("author") or {}).get("display_name") or ""}
            for a in (w.get("authorships") or [])
            if (a.get("author") or {}).get("display_name")
        ],
        "year": year,
        "abstract": abstract_from_index(w.get("abstract_inverted_index")),
        "url": url,
        "externalIds": {"ArXiv": arxiv, "DOI": doi},
        "referenceCount": w.get("referenced_works_count") or 0,
        "citationCount": w.get("cited_by_count") or 0,
        "venue": source.get("display_name"),
        "publicationDate": w.get("publication_date"),
    }

def search_paper(query):
    query = query.strip()

    m = re.search(r"arxiv\.org/abs/([^\s?#]+)|^(\d{4}\.\d{4,5}(v\d+)?)$", query, re.IGNORECASE)
    if m:
        arxiv_id = re.sub(r"v\d+$", "", m.group(1) or m.group(2))
        candidates = [
            f"http://arxiv.org/abs/{arxiv_id}",
            f"https://doi.org/10.48550/arxiv.{arxiv_id}",
        ]
        for landing_url in candidates:
            data = oa_fetch(
                f"{BASE}/works?filter=locations.landing_page_url:"
                f"{urllib.parse.quote(landing_url, safe='')}"
                f"&per-page=1&select={FULL_FIELDS}"
            )
            results = data.get("results") or []
            if results:
                paper = map_work(results[0])
                paper["externalIds"]["ArXiv"] = paper["externalIds"]["ArXiv"] or arxiv_id
                m2 = re.match(r"(\d{2})(\d{2})\.\d+", arxiv_id)
                if m2:
                    yy = int(m2.group(1))
                    arxiv_year = 1900 + yy if yy >= 90 else 2000 + yy
                    if paper["year"] is None or arxiv_year < paper["year"]:
                        paper["year"] = arxiv_year
                return paper
        raise RuntimeError("Paper not found. Try a DOI or arXiv ID.")

    m = re.search(r"10\.\d{4,}/\S+", query)
    if m:
        doi = m.group(0)
        try:
            w = oa_fetch(f"{BASE}/works/doi:{doi}?select={FULL_FIELDS}")
            return map_work(w)
        except RuntimeError:
            data = oa_fetch(
                f"{BASE}/works?filter=locations.landing_page_url:"
                f"{urllib.parse.quote(f'https://doi.org/{doi}', safe='')}"
                f"&per-page=1&select={FULL_FIELDS}"
            )
            results = data.get("results") or []
            if results:
                return map_work(results[0])
            raise

    if re.fullmatch(r"W\d+", query):
        w = oa_fetch(f"{BASE}/works/{query}?select={FULL_FIELDS}")
        return map_work(w)

    data = oa_fetch(
        f"{BASE}/works?search={urllib.parse.quote(query)}&per-page=1&select={FULL_FIELDS}"
    )
    results = data.get("results") or []
    if not results:
        raise RuntimeError("No papers found")
    return map_work(results[0])

def fetch_references(paper_id, limit):
    results = []
    page = 1
    while len(results) < limit:
        per_page = min(PAGE_SIZE, limit - len(results))
        data = oa_fetch(
            f"{BASE}/works?filter=cited_by:{paper_id}"
            f"&per-page={per_page}&page={page}&select={MINI_FIELDS}"
        )
        batch = data.get("results") or []
        results.extend(batch)
        if len(batch) < per_page:
            break
        page += 1
    return [map_work(w) for w in results]

def fetch_citations(paper_id, limit):
    results = []
    page = 1
    while len(results) < limit:
        per_page = min(PAGE_SIZE, limit - len(results))
        data = oa_fetch(
            f"{BASE}/works?filter=cites:{paper_id}"
            f"&per-page={per_page}&page={page}"
            f"&sort=cited_by_count:desc&select={MINI_FIELDS}"
        )
        batch = data.get("results") or []
        results.extend(batch)
        if len(batch) < per_page:
            break
        page += 1
    return [map_work(w) for w in results]

def build_citation_graph(query, max_papers=50):
    seed = search_paper(query)
    print(f"Seed paper: {seed['title']} ({seed['year']}) — {seed['paperId']}")

    seed["depth"] = 0
    seed["type"] = "root"
    papers = {seed["paperId"]: seed}
    frontier = [seed]

    while frontier and len(papers) < max_papers:
        next_frontier = []
        for node in frontier:
            if len(papers) >= max_papers:
                break
            remaining = max_papers - len(papers)
            print(f"Expanding: {node['title'][:60]} (depth {node['depth']}, "
                  f"{remaining} slots left)")

            refs = fetch_references(node["paperId"], remaining)
            cits = fetch_citations(node["paperId"], remaining)

            interleaved = []
            for i in range(max(len(refs), len(cits))):
                if i < len(refs):
                    interleaved.append((refs[i], "reference"))
                if i < len(cits):
                    interleaved.append((cits[i], "citation"))

            for paper, kind in interleaved:
                if len(papers) >= max_papers:
                    break
                if not paper["paperId"] or paper["paperId"] in papers:
                    continue
                paper["depth"] = node["depth"] + 1
                paper["type"] = kind
                papers[paper["paperId"]] = paper
                next_frontier.append(paper)

        frontier = next_frontier

    return list(papers.values())

def main():
    papers = build_citation_graph(QUERY, MAX_PAPERS)

    output = {
        "query": QUERY,
        "maxPapers": MAX_PAPERS,
        "paperCount": len(papers),
        "papers": papers,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(papers)} papers to {OUTPUT_PATH}")

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
