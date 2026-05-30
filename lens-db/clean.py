import json

def _names(party_list):
    if not party_list:
        return "null"
    names = [p.get("extracted_name", {}).get("value", "") for p in party_list]
    return [n for n in names if n] or "null"

def _ipc_symbols(classifications):
    if not classifications:
        return "null"
    symbols = [c.get("symbol") for c in classifications if c.get("symbol")]
    return symbols or "null"

def _family_lens_ids(members):
    if not members:
        return "null"
    ids = [m.get("lens_id") for m in members if m.get("lens_id")]
    return ids or "null"

def _cited_by_lens_ids(cited_by):
    if not cited_by:
        return "null"
    ids = [p.get("lens_id") for p in cited_by.get("patents", []) if p.get("lens_id")]
    return ids or "null"

def _abstract_text(abstract):
    if not abstract:
        return "null"
    if isinstance(abstract, str):
        return abstract or "null"
    texts = [item.get("text", "") for item in abstract if isinstance(item, dict) and item.get("text")]
    return " ".join(texts) or "null"

def flatten(d):
    biblio = d.get("biblio", {})
    parties = biblio.get("parties", {})
    families = d.get("families", {})

    return {
        "lens_id":                  d.get("lens_id") or "null",
        "jurisdiction":             d.get("jurisdiction") or "null",
        "doc_number":               d.get("doc_number") or "null",
        "kind":                     d.get("kind") or "null",
        "date_published":           d.get("date_published") or "null",
        "lang":                     d.get("lang") or "null",
        "publication_type":         d.get("publication_type") or "null",
        "patent_status":            d.get("legal_status", {}).get("patent_status") or "null",
        "title":                    [t.get("text") for t in biblio.get("invention_title", []) if t.get("text")] or "null",
        "application_date":         biblio.get("application_reference", {}).get("date") or "null",
        "priority_date":            biblio.get("priority_claims", {}).get("earliest_claim", {}).get("date") or "null",
        "applicants":               _names(parties.get("applicants")),
        "inventors":                _names(parties.get("inventors")),
        "ipc_classifications":      _ipc_symbols(biblio.get("classifications_ipcr", {}).get("classifications")),
        "cited_by_lens_ids":        _cited_by_lens_ids(biblio.get("cited_by")),
        "simple_family_lens_ids":   _family_lens_ids(families.get("simple_family", {}).get("members")),
        "extended_family_lens_ids": _family_lens_ids(families.get("extended_family", {}).get("members")),
        "abstract":                 _abstract_text(d.get("abstract")),
        "description":              _abstract_text(d.get("description")),
    }

seen = set()
kept = []

with open("lens-raw-data.jsonl") as f:
    for line in f:
        d = json.loads(line.strip())

        if d.get("jurisdiction") != "US":
            continue
        if not d.get("abstract"):
            continue

        lid = d.get("lens_id")
        if lid in seen:
            continue
        seen.add(lid)

        kept.append(flatten(d))

# All records share the same schema from flatten(), so no column gaps exist.
# This pass is a safety net in case any future change introduces them.
all_keys = set().union(*(r.keys() for r in kept))
with open("lens-clean.jsonl", "w") as f:
    for r in kept:
        for k in all_keys:
            if k not in r:
                r[k] = "null"
        f.write(json.dumps(r) + "\n")

print(f"Done: {len(kept)} samples, {len(all_keys)} columns -> lens-clean.jsonl")
