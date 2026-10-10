"""Step 1b: supplement OpenAlex with Semantic Scholar and Crossref (Philippine journals),
de-duplicate, and audit what OpenAlex missed.

Sources
  s2       : Semantic Scholar bulk search, query "Philippines | Philippine", fields of study in S2_FIELDS
  crossref : all 2005-2025 articles in PH_JOURNALS (by ISSN), kept if ecology terms appear
Outputs
  data/corpus/corpus.parquet            merged corpus (column `source`: openalex | s2 | crossref)
  results/corpus_source_audit.csv       per source: found, with abstract, already in OpenAlex, new
  results/corpus_ph_journal_audit.csv   per Philippine journal: same, plus missing-abstract rate
"""
import re, time, json, html
import requests
import pandas as pd
from tqdm import tqdm
from config import RAW, CORPUS, RESULTS, OPENALEX_EMAIL, S2_API_KEY, S2_FIELDS, PH_JOURNALS, ECO_TERMS, SUBFIELDS

UA = {"User-Agent": f"agenda-as-query/1.0 (mailto:{OPENALEX_EMAIL})"}


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:120]


def norm_doi(d):
    if not d:
        return None
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", d.strip().lower())


def is_eco(text):
    t = (text or "").lower()
    return any(term in t for term in ECO_TERMS)


def ph_group(affils):
    if not affils:
        return "unknown"
    return "ph_affiliated_unknown_role" if any("philippin" in a.lower() for a in affils) else "unknown"


# ---------------- Semantic Scholar ----------------
def fetch_s2():
    out = RAW / "s2.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out} exists"); return out
    url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    params = {"query": "Philippines | Philippine", "year": "2005-2025",
              "fieldsOfStudy": ",".join(S2_FIELDS),
              "fields": "title,abstract,year,externalIds,venue,publicationTypes"}
    headers = dict(UA, **({"x-api-key": S2_API_KEY} if S2_API_KEY else {}))
    tmp = out.with_suffix(".part")
    n = 0
    try:
        with open(tmp, "w") as f, tqdm(desc="s2") as bar:
            token = None
            while True:
                p = dict(params, **({"token": token} if token else {}))
                for attempt in range(8):
                    r = requests.get(url, params=p, headers=headers, timeout=90)
                    if r.status_code in (429, 500, 502, 503, 504):
                        time.sleep(3 + 2 ** attempt); continue
                    break
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                js = r.json()
                for w in js.get("data") or []:
                    f.write(json.dumps(w) + "\n"); n += 1
                bar.update(len(js.get("data") or []))
                token = js.get("token")
                if not token:
                    break
                time.sleep(1.1)
    except Exception as e:
        print(f"\n[warn] Semantic Scholar failed after {n} records: {e}")
        if n == 0:
            tmp.unlink(missing_ok=True)
            return None
    tmp.rename(out)
    print(f"s2: {n} records"); return out


def load_s2(path):
    rows = []
    for w in map(json.loads, open(path)):
        rows.append(dict(title=w.get("title") or "", abstract=w.get("abstract") or "", year=w.get("year"),
                         doi=norm_doi((w.get("externalIds") or {}).get("DOI")), venue=w.get("venue"),
                         author_group="unknown", source="s2",
                         doc_id="S2_" + str(w.get("paperId", ""))))
    return pd.DataFrame(rows)


# ---------------- Crossref (Philippine journals) ----------------
def fetch_crossref(issn):
    out = RAW / f"crossref_{issn}.jsonl"
    if out.exists() and out.stat().st_size > 0:
        return out
    tmp = out.with_suffix(".part")
    url = "https://api.crossref.org/works"
    params = {"filter": f"issn:{issn},from-pub-date:2005-01-01,until-pub-date:2025-12-31",
              "rows": 500, "cursor": "*", "mailto": OPENALEX_EMAIL,
              "select": "DOI,title,abstract,issued,author,type,container-title"}
    with open(tmp, "w") as f:
        while True:
            r = None
            for attempt in range(6):
                try:
                    r = requests.get(url, params=params, headers=UA, timeout=120); r.raise_for_status(); break
                except Exception:
                    r = None; time.sleep(2 ** attempt)
            if r is None:
                raise RuntimeError("Crossref unreachable")
            msg = r.json()["message"]
            for w in msg["items"]:
                f.write(json.dumps(w) + "\n")
            if not msg["items"]:
                break
            params["cursor"] = msg["next-cursor"]
    tmp.rename(out)
    return out


def strip_jats(s):
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = html.unescape(s)
    s = re.sub(r"^\s*abstract\s*", "", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def load_crossref(issn, name):
    rows = []
    for w in map(json.loads, open(fetch_crossref(issn))):
        if w.get("type") not in (None, "journal-article"):
            continue
        affils = [a.get("name", "") for au in (w.get("author") or []) for a in (au.get("affiliation") or [])]
        yr = ((w.get("issued") or {}).get("date-parts") or [[None]])[0][0]
        rows.append(dict(title=" ".join(w.get("title") or []), abstract=strip_jats(w.get("abstract")),
                         year=yr, doi=norm_doi(w.get("DOI")), venue=name,
                         author_group=ph_group(affils) if affils else "ph_journal_unknown_role",
                         source="crossref", doc_id="CR_" + (norm_doi(w.get("DOI")) or "").replace("/", "_")))
    return pd.DataFrame(rows)


# ---------------- OpenAlex DOI lookup (scope audit) ----------------
def openalex_lookup(dois):
    """Return {doi: subfield_id or 'none'} for DOIs found in OpenAlex; absent DOIs are not in OpenAlex."""
    cache_p = RAW / "oa_doi_lookup.json"
    cache = json.load(open(cache_p)) if cache_p.exists() else {}
    todo = [d for d in dois if d and d not in cache and "|" not in d and "," not in d]
    for i in tqdm(range(0, len(todo), 50), desc="openalex doi lookup"):
        batch = todo[i:i + 50]
        params = {"filter": "doi:" + "|".join(batch), "per_page": 50, "select": "doi,primary_topic"}
        if OPENALEX_EMAIL:
            params["mailto"] = OPENALEX_EMAIL
        for attempt in range(5):
            try:
                r = requests.get("https://api.openalex.org/works", params=params, timeout=60); r.raise_for_status(); break
            except Exception:
                time.sleep(2 ** attempt)
        else:
            continue
        found = {}
        for w in r.json().get("results", []):
            sf = ((w.get("primary_topic") or {}).get("subfield") or {}).get("id")
            found[norm_doi(w.get("doi"))] = str(sf).rsplit("/", 1)[-1] if sf else "none"
        for d in batch:
            cache[d] = found.get(d, "absent")
        if i % 1000 == 0:
            json.dump(cache, open(cache_p, "w"))
    json.dump(cache, open(cache_p, "w"))
    return cache


def main():
    oa = pd.read_parquet(CORPUS / "corpus_openalex.parquet")
    oa["doi_n"] = oa["doi"].map(norm_doi)
    oa_dois, oa_titles = set(oa["doi_n"].dropna()), set(oa["title"].map(norm_title))

    # also check against the RAW OpenAlex harvest (incl. papers filtered out for short/missing abstracts)
    raw_titles = set()
    for p in RAW.glob("*.jsonl"):
        if p.name.startswith(("ph_affiliated", "about_ph")):
            raw_titles |= {norm_title(json.loads(l).get("title")) for l in open(p)}

    frames, audit, jaudit = [], [], []
    s2_path = fetch_s2()
    if s2_path:
        s2 = load_s2(s2_path)
        frames.append(s2[s2["title"].str.len() > 0])
    else:
        print("Continuing without Semantic Scholar (re-run `make supplement` later to retry).")
    for issn, name in PH_JOURNALS.items():
        try:
            cr = load_crossref(issn, name)
        except Exception as e:
            print(f"crossref {name}: failed ({e})"); continue
        if cr.empty:
            print(f"crossref {name} ({issn}): 0 records - check ISSN"); continue
        cr_eco = cr[(cr["title"] + " " + cr["abstract"]).map(is_eco)]
        in_oa = cr_eco["doi"].isin(oa_dois) | cr_eco["title"].map(norm_title).isin(oa_titles)
        jaudit.append(dict(journal=name, issn=issn, articles=len(cr), ecology=len(cr_eco),
                           ecology_with_abstract=int((cr_eco["abstract"].str.split().str.len() >= 50).sum()),
                           missing_abstract_rate=float((cr_eco["abstract"].str.split().str.len() < 50).mean()) if len(cr_eco) else None,
                           already_in_openalex_corpus=int(in_oa.sum())))
        frames.append(cr_eco)
    if not frames:
        raise SystemExit("No supplementary records retrieved (S2 and Crossref both failed). Corpus = OpenAlex only.")
    sup = pd.concat(frames, ignore_index=True)
    sup["year"] = pd.to_numeric(sup["year"], errors="coerce")
    sup = sup[sup["year"].between(2005, 2025)]
    # same thematic scope for every source: ecology lexicon on title+abstract
    sup = sup[(sup["title"] + " " + sup["abstract"]).map(is_eco)].copy()
    # Philippine scope: S2 query already requires Philippines; PH-journal papers are in scope by venue
    sup["has_abstract"] = sup["abstract"].str.split().str.len() >= 50
    sup["in_oa_corpus"] = sup["doi"].isin(oa_dois) | sup["title"].map(norm_title).isin(oa_titles)
    sup["in_oa_raw"] = sup["in_oa_corpus"] | sup["title"].map(norm_title).isin(raw_titles)

    # classify candidates not in our OpenAlex corpus via DOI lookup
    cand = sup[~sup["in_oa_corpus"]]
    look = openalex_lookup(sorted(set(cand["doi"].dropna())))
    sf_ids = {str(k) for k in SUBFIELDS}

    def status(row):
        if row["in_oa_corpus"]:
            return "in_openalex_corpus"
        d = row["doi"]
        if not d:
            return "no_doi"
        v = look.get(d, "absent")
        if v == "absent":
            return "not_in_openalex"
        return "in_openalex_in_scope_subfield" if v in sf_ids else "in_openalex_other_subfield"
    sup["oa_status"] = sup.apply(status, axis=1)

    for src, g in sup.groupby("source"):
        row = dict(source=src, in_scope_found=len(g), with_abstract=int(g["has_abstract"].sum()))
        row.update(g["oa_status"].value_counts().to_dict())
        audit.append(row)
    pd.DataFrame(audit).fillna(0).to_csv(RESULTS / "corpus_source_audit.csv", index=False)
    pd.DataFrame(jaudit).to_csv(RESULTS / "corpus_ph_journal_audit.csv", index=False)

    # add: OpenAlex coverage gaps (not_in_openalex, no_doi) and in-scope papers our query missed;
    # exclude papers OpenAlex files under other subfields (outside our corpus definition)
    keep = {"not_in_openalex", "no_doi", "in_openalex_in_scope_subfield"}
    new = sup[sup["oa_status"].isin(keep) & sup["has_abstract"]].copy()
    new["tkey"] = new["title"].map(norm_title)
    new = new.drop_duplicates("tkey").drop(columns=["tkey"])
    new["text"] = new["title"] + ". " + new["abstract"]
    new["scopes"] = new["oa_status"]
    for c in oa.columns:
        if c not in new.columns:
            new[c] = None
    merged = pd.concat([oa.drop(columns=["doi_n"]), new[oa.drop(columns=["doi_n"]).columns]], ignore_index=True)
    merged.to_parquet(CORPUS / "corpus.parquet", index=False)

    print("\n=== Source audit (in-scope = 2005-2025 + ecology terms) ===")
    print(pd.DataFrame(audit).fillna(0).to_string(index=False))
    print("  in_openalex_other_subfield = excluded (outside corpus definition, not an OpenAlex miss)")
    print("  not_in_openalex / no_doi   = OpenAlex coverage gap -> added")
    print("  in_openalex_in_scope_subfield = our query missed it -> added")
    print("\n=== Philippine journal audit ==="); print(pd.DataFrame(jaudit).round(3).to_string(index=False))
    print(f"\nCorpus: {len(oa)} OpenAlex + {len(new)} supplementary = {len(merged)} -> data/corpus/corpus.parquet")


if __name__ == "__main__":
    main()
