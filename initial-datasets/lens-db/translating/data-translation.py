import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from translate import translate

INPUT   = "lens-db-clean.json"
OUTPUT  = "lens-db-translated.json"
WORKERS = 8

def safe_translate(text, source, target="en", retries=3, delay=2):
    if not text or text == "Null":
        return text
    src = "auto" if (not source or source == "Null") else source
    for attempt in range(retries):
        try:
            return translate(text, source=src, target=target)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                tqdm.write(f"  [WARN] translation failed ({src} -> en): {e}")
                return text

def translate_multilang_field(items):
    """Use existing English entry if present, otherwise translate the first entry."""
    if items == "Null" or not items:
        return "Null"
    for item in items:
        if item.get("lang") == "en" and item.get("text"):
            return item["text"]
    first = items[0]
    return safe_translate(first.get("text", ""), source=first.get("lang"))

def translate_record(r):
    top_lang = r.get("lang", "Null")
    r["title"]    = translate_multilang_field(r.get("title"))
    r["abstract"] = translate_multilang_field(r.get("abstract"))
    desc = r.get("description")
    if desc and desc != "Null" and top_lang != "en":
        r["description"] = safe_translate(desc, source=top_lang)
    r["lang"] = "en"
    return r

with open(INPUT, encoding="utf-8") as f:
    data = json.load(f)

total = len(data)
print(f"Translating {total} records with {WORKERS} workers...\n")

results = [None] * total

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(translate_record, r): i for i, r in enumerate(data)}
    with tqdm(total=total, unit="rec", dynamic_ncols=True) as bar:
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            bar.update(1)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nDone -> {OUTPUT}")
