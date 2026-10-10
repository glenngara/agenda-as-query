"""Step 1: harvest the corpus from OpenAlex.

Two scopes, merged and de-duplicated:
  A) works with >=1 Philippine-affiliated author in ecology/environment subfields
  B) works mentioning the Philippines in title/abstract in the same subfields (any authors)
Each work is labeled local-led / collaborative / foreign-only from author countries.

Output: data/corpus/corpus_openalex.parquet (+ corpus.parquet; run 01b next to add other sources)
"""
import json, time
import requests
import pandas as pd
from tqdm import tqdm
from config import RAW, CORPUS, OPENALEX_EMAIL, YEARS, SUBFIELDS

API = "https://api.openalex.org/works"
SELECT = ",".join(["id", "doi", "title", "publication_year", "type", "language",
                   "abstract_inverted_index", "authorships", "primary_topic",
                   "cited_by_count", "primary_location"])
BASE = (f"publication_year:{YEARS},type:article,has_abstract:true,"
        f"primary_topic.subfield.id:{'|'.join(str(s) for s in SUBFIELDS)}")
SCOPES = {
    "ph_affiliated": BASE + ",authorships.institutions.country_code:PH",
    "about_ph": BASE + ",title_and_abstract.search:Philippines",
}


def fetch(scope, flt):
    out = RAW / f"{scope}.jsonl"
    if out.exists():
        print(f"[skip] {out} exists (delete it to re-harvest)")
        return out
    params = {"filter": flt, "per_page": 200, "cursor": "*", "select": SELECT}
    if OPENALEX_EMAIL:
        params["mailto"] = OPENALEX_EMAIL
    r = requests.get(API, params={**params, "per_page": 1}, timeout=60)
    r.raise_for_status()
    total = r.json()["meta"]["count"]
    print(f"{scope}: {total} works")
    with open(out, "w") as f, tqdm(total=total, desc=scope) as bar:
        while params["cursor"]:
            last_err = None
            for attempt in range(5):
                try:
                    r = requests.get(API, params=params, timeout=90)
                    r.raise_for_status()
                    break
                except Exception as err:
                    last_err = err
                    time.sleep(2 ** attempt)
            else:
                raise SystemExit(f"OpenAlex failed repeatedly: {last_err}")
            js = r.json()
            for w in js["results"]:
                f.write(json.dumps(w) + "\n")
            bar.update(len(js["results"]))
            params["cursor"] = js["meta"].get("next_cursor")
            if not js["results"]:
                break
    return out


def rebuild_abstract(inv):
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def author_group(w):
    countries = []
    for a in w.get("authorships") or []:
        cs = set(a.get("countries") or [])
        for inst in a.get("institutions") or []:
            if inst.get("country_code"):
                cs.add(inst["country_code"])
        countries.append(cs)
    if not countries or not any(countries):
        return "unknown", 0
    any_ph = any("PH" in c for c in countries)
    first_ph = "PH" in countries[0]
    n_countries = len(set().union(*countries))
    if not any_ph:
        return "foreign_only", n_countries
    if first_ph and all(c <= {"PH"} for c in countries if c):
        return "local_only", n_countries
    if first_ph:
        return "local_led_collab", n_countries
    return "foreign_led_collab", n_countries


def main():
    rows = {}
    for scope, flt in SCOPES.items():
        path = fetch(scope, flt)
        with open(path) as f:
            for line in f:
                w = json.loads(line)
                wid = w["id"].rsplit("/", 1)[-1]
                if wid in rows:
                    rows[wid]["scopes"].add(scope)
                    continue
                abstract = rebuild_abstract(w.get("abstract_inverted_index"))
                grp, nc = author_group(w)
                pt = w.get("primary_topic") or {}
                src = ((w.get("primary_location") or {}).get("source") or {})
                rows[wid] = dict(
                    doc_id=wid, doi=w.get("doi"), title=(w.get("title") or "").strip(),
                    abstract=abstract, year=w.get("publication_year"),
                    language=w.get("language"), cited_by=w.get("cited_by_count", 0),
                    subfield=(pt.get("subfield") or {}).get("display_name"),
                    topic=pt.get("display_name"), venue=src.get("display_name"),
                    author_group=grp, n_countries=nc, scopes={scope},
                )
    df = pd.DataFrame(rows.values())
    df["scopes"] = df["scopes"].apply(lambda s: "|".join(sorted(s)))
    n0 = len(df)
    df = df[(df["language"].fillna("en") == "en") & (df["abstract"].str.split().str.len() >= 50)]
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    df["text"] = df["title"] + ". " + df["abstract"]
    df["source"] = "openalex"
    df.to_parquet(CORPUS / "corpus_openalex.parquet", index=False)
    df.to_parquet(CORPUS / "corpus.parquet", index=False)   # 01b adds supplementary sources
    print(f"\nCorpus: {len(df)} works (from {n0} before filtering)")
    print(df["author_group"].value_counts().to_string())
    print(df.groupby(pd.cut(df["year"], [2004, 2010, 2015, 2020, 2025])).size().to_string())


if __name__ == "__main__":
    main()
