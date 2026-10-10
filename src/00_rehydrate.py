"""Step 0 (for reproducers): rebuild the corpus text and the judging pool from public IDs.

The released repository does not redistribute paper titles or abstracts. This script downloads them
again from the public APIs and writes the two files the pipeline reads:

  data/corpus/corpus_ids.csv  ->  data/corpus/corpus.parquet   (adds title, abstract, text)
  data/pool/pool_ids.csv      ->  data/pool/pool.csv           (adds title, abstract)

ID formats: W...   = OpenAlex work ID          (https://api.openalex.org)
            S2_... = Semantic Scholar paper ID  (https://api.semanticscholar.org)
            CR_... = Crossref DOI-based ID      (https://api.crossref.org; abstract when deposited)

Abstracts can change upstream after our harvest, so a small number of texts may differ.
The released runs, pool and grades are fixed; re-running retrieval on rehydrated text reproduces them
up to those differences. Set OPENALEX_EMAIL (and optionally S2_API_KEY) in .env for faster access.
Usage: python 00_rehydrate.py
"""
import os, re, time, html
import requests
import pandas as pd
from tqdm import tqdm
from config import CORPUS, POOL

EMAIL = os.getenv("OPENALEX_EMAIL", "")
S2_KEY = os.getenv("S2_API_KEY", "")


def inv_to_text(inv):
    if not inv:
        return ""
    pos = {}
    for word, idx in inv.items():
        for i in idx:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def get(url, **kw):
    for attempt in range(5):
        r = requests.get(url, timeout=60, **kw)
        if r.status_code == 200:
            return r
        time.sleep(2 ** attempt)
    r.raise_for_status()


def openalex(ids):
    out = {}
    for i in tqdm(range(0, len(ids), 50), desc="OpenAlex"):
        chunk = ids[i:i + 50]
        r = get("https://api.openalex.org/works",
                params={"filter": "openalex_id:" + "|".join(chunk), "per-page": 50, "mailto": EMAIL,
                        "select": "id,title,abstract_inverted_index"})
        for w in r.json()["results"]:
            wid = w["id"].rsplit("/", 1)[-1]
            out[wid] = (w.get("title") or "", inv_to_text(w.get("abstract_inverted_index")))
    return out


def semantic_scholar(ids):
    out = {}
    headers = {"x-api-key": S2_KEY} if S2_KEY else {}
    raw = [i[3:] for i in ids]
    for i in tqdm(range(0, len(raw), 400), desc="Semantic Scholar"):
        chunk = raw[i:i + 400]
        for attempt in range(5):
            r = requests.post("https://api.semanticscholar.org/graph/v1/paper/batch",
                              params={"fields": "title,abstract"}, json={"ids": chunk}, headers=headers, timeout=120)
            if r.status_code == 200:
                break
            time.sleep(2 ** attempt + 1)
        for pid, p in zip(chunk, r.json()):
            if p:
                out["S2_" + pid] = (p.get("title") or "", p.get("abstract") or "")
    return out


def crossref(rows):
    out = {}
    for doc_id, doi in tqdm(rows, desc="Crossref"):
        try:
            m = get(f"https://api.crossref.org/works/{doi}", params={"mailto": EMAIL}).json()["message"]
        except Exception:
            continue
        ab = re.sub(r"<[^>]+>", " ", m.get("abstract", "") or "")
        out[doc_id] = ((m.get("title") or [""])[0], html.unescape(re.sub(r"\s+", " ", ab)).strip())
    return out


def main():
    c = pd.read_csv(CORPUS / "corpus_ids.csv")
    ids = c["doc_id"].tolist()
    texts = {}
    texts.update(openalex([i for i in ids if i.startswith("W")]))
    texts.update(semantic_scholar([i for i in ids if i.startswith("S2_")]))
    cr = c[c["doc_id"].str.startswith("CR_")][["doc_id", "doi"]].dropna()
    texts.update(crossref(list(cr.itertuples(index=False, name=None))))
    c["title"] = c["doc_id"].map(lambda d: texts.get(d, ("", ""))[0])
    c["abstract"] = c["doc_id"].map(lambda d: texts.get(d, ("", ""))[1])
    c["text"] = c["title"] + ". " + c["abstract"]
    missing = (c["abstract"].str.len() < 20).sum()
    c.to_parquet(CORPUS / "corpus.parquet", index=False)
    p = pd.read_csv(POOL / "pool_ids.csv").merge(c[["doc_id", "title", "abstract"]], on="doc_id", how="left")
    p.to_csv(POOL / "pool.csv", index=False)
    print(f"corpus: {len(c)} docs ({missing} without abstract upstream) -> {CORPUS / 'corpus.parquet'}")
    print(f"pool:   {len(p)} pairs -> {POOL / 'pool.csv'}")


if __name__ == "__main__":
    main()
