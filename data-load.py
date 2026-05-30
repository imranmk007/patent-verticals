import json
import math
import ssl
import tarfile
import urllib.request
from pathlib import Path

import certifi
import pandas as pd

# mac ssl fix
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

DATA_DIR = Path(__file__).parent.parent / "data"
TARBALL = DATA_DIR / "sample-jan-2016.tar.gz"
HF_TOKEN = ""
YEAR_URL = "https://huggingface.co/datasets/HUPD/hupd/resolve/main/data/{year}.tar.gz"
RANDOM_STATE = 42


def load_hupd_sample(tarball: Path = TARBALL) -> pd.DataFrame:
    records = []
    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar:
            if not (member.isfile() and member.name.endswith(".json")):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            records.append(json.load(f))

    df = pd.DataFrame(records)
    # shuffle rows
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def _stream_year(year: int, per_year: int) -> list[dict]:
    url = YEAR_URL.format(year=year)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
    records = []
    with urllib.request.urlopen(req, context=SSL_CONTEXT) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if not (member.isfile() and member.name.endswith(".json")):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                records.append(json.load(f))
                if len(records) >= per_year:
                    break  # stop early
    return records


def load_hupd_crossyear(
    n_total: int = 15000,
    years: range = range(2004, 2019),
    cache_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    cache = cache_dir / f"hupd_crossyear_{n_total}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    n_years = len(years)
    per_year = math.ceil(n_total / n_years)
    records = []
    for year in years:
        year_records = _stream_year(year, per_year)
        records.extend(year_records)
        print(f"  {year}: {len(year_records):,} records (total {len(records):,})")

    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    df = df.head(n_total).reset_index(drop=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


if __name__ == "__main__":
    print("Streaming cross-year HUPD sample (2004-2018)...")
    df = load_hupd_crossyear(n_total=2100)
    years = pd.to_datetime(df["filing_date"], format="%Y%m%d", errors="coerce").dt.year
    print(
        f"Loaded {len(df):,} HUPD patent applications across "
        f"{years.nunique()} filing years ({int(years.min())}-{int(years.max())}) "
        f"with {df.shape[1]} fields each. "
        f"Example title: {df.loc[0, 'title']!r}"
    )
