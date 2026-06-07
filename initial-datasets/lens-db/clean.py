import json

NULL = "Null"

def _names(party_list):
    if not party_list:
        return []
    names = [p.get("extracted_name", {}).get("value", "") for p in party_list]
    return [n for n in names if n]

def _ipc_symbols(classifications):
    if not classifications:
        return []
    return [c["symbol"] for c in classifications if c.get("symbol")]

def _family_lens_ids(members):
    if not members:
        return []
    return [m["lens_id"] for m in members if m.get("lens_id")]

def _cited_by_lens_ids(cited_by):
    if not cited_by:
        return []
    return [p["lens_id"] for p in cited_by.get("patents", []) if p.get("lens_id")]

def _multilang_list(items):
    """Return list of {text, lang} dicts, preserving language identifiers."""
    if not items:
        return NULL
    if isinstance(items, list):
        out = [{"text": i["text"], "lang": i.get("lang", NULL)} for i in items if i.get("text")]
        return out or NULL
    if isinstance(items, dict) and items.get("text"):
        return [{"text": items["text"], "lang": items.get("lang", NULL)}]
    return NULL

def _description_text(desc):
    if not desc:
        return NULL
    if isinstance(desc, str):
        return desc or NULL
    if isinstance(desc, dict):
        return desc.get("text") or NULL
    if isinstance(desc, list):
        texts = [i.get("text", "") for i in desc if isinstance(i, dict) and i.get("text")]
        return " ".join(texts) or NULL
    return NULL

def flatten(d):
    biblio = d.get("biblio") or {}
    parties = biblio.get("parties") or {}
    families = d.get("families") or {}

    return {
        "lens_id":                  d.get("lens_id") or NULL,
        "jurisdiction":             d.get("jurisdiction") or NULL,
        "doc_number":               d.get("doc_number") or NULL,
        "kind":                     d.get("kind") or NULL,
        "date_published":           d.get("date_published") or NULL,
        "lang":                     d.get("lang") or NULL,
        "publication_type":         d.get("publication_type") or NULL,
        "patent_status":            d.get("legal_status", {}).get("patent_status") or NULL,
        "title":                    _multilang_list(biblio.get("invention_title")),
        "application_date":         biblio.get("application_reference", {}).get("date") or NULL,
        "priority_date":            biblio.get("priority_claims", {}).get("earliest_claim", {}).get("date") or NULL,
        "applicants":               _names(parties.get("applicants")),
        "inventors":                _names(parties.get("inventors")),
        "ipc_classifications":      _ipc_symbols(biblio.get("classifications_ipcr", {}).get("classifications")),
        "cited_by_lens_ids":        _cited_by_lens_ids(biblio.get("cited_by")),
        "simple_family_lens_ids":   _family_lens_ids(families.get("simple_family", {}).get("members")),
        "extended_family_lens_ids": _family_lens_ids(families.get("extended_family", {}).get("members")),
        "abstract":                 _multilang_list(d.get("abstract")),
        "description":              _description_text(d.get("description")),
    }

SCHEMA_KEYS = [
    "lens_id", "jurisdiction", "doc_number", "kind", "date_published", "lang",
    "publication_type", "patent_status", "title", "application_date", "priority_date",
    "applicants", "inventors", "ipc_classifications", "cited_by_lens_ids",
    "simple_family_lens_ids", "extended_family_lens_ids", "abstract", "description",
]

seen = set()
kept = []

with open("lens-raw.jsonl") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)

        # A: remove records without abstracts
        if not d.get("abstract"):
            continue

        # deduplicate by lens_id
        lid = d.get("lens_id")
        if lid in seen:
            continue
        seen.add(lid)

        kept.append(flatten(d))

# B: ensure every record has every schema key (fill missing with Null)
for r in kept:
    for k in SCHEMA_KEYS:
        if k not in r:
            r[k] = NULL

with open("lens-db-clean.json", "w", encoding="utf-8") as f:
    json.dump(kept, f, ensure_ascii=False, indent=2)

print(f"Done: {len(kept)} records, {len(SCHEMA_KEYS)} columns -> lens-db-clean.json")
