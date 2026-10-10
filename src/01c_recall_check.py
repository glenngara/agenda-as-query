"""Step 1c: corpus recall check against expert-compiled reference lists.

Put gold sources in data/recall/:
  - review-article PDFs (e.g., Tanalgo & Hughes Philippine bat review): the References section is parsed
  - .ris or .bib exports (e.g., a reference-manager library of Philippine ecology papers): titles used directly
Each reference is matched to corpus titles with fuzzy matching (rapidfuzz). Only references from
2005-2025 that look like journal articles are counted as in scope.

Output: results/corpus_recall.csv (per gold source: in-scope refs, recall of OpenAlex-only and merged corpus)
        results/corpus_recall_unmatched.csv (for manual inspection)
"""
import re, unicodedata, logging
import pandas as pd
from pypdf import PdfReader
from rapidfuzz import process, fuzz
from config import CORPUS, RESULTS, DATA

logging.getLogger("pypdf").setLevel(logging.ERROR)
RECALL = DATA / "recall"
YEAR = re.compile(r"\((?:19|20)(\d\d)[a-z]?\)|\b(?:19|20)(\d\d)[a-z]?\.")
PH_TERMS = re.compile(r"philippin|luzon|visayas|mindanao|palawan|mindoro|panay|negros|cebu|bohol|leyte|samar|"
                      r"iloilo|davao|zamboanga|sulu|tawi|basilan|moro gulf|manila|laguna|batangas|quezon|"
                      r"isabela|cagayan|ilocos|bicol|albay|sorsogon|catanduanes|marinduque|romblon|masbate|"
                      r"siquijor|camiguin|surigao|agusan|bukidnon|lanao|cotabato|sarangani|benguet|ifugao|"
                      r"sierra madre|subic|tubbataha|apo reef|verde island|taal|bataan|pampanga|bulacan", re.I)
BOILER = re.compile(r"\.?CC-BY[^.]*?(license|licence)[^.]*?(available under a)?|was not certified by peer review\)?[^.]*?perpetuity\. It is made|"
                    r"The copyright holder for this preprint \(which|this version posted [A-Z][a-z]+ \d+, \d{4}\.?|"
                    r";?\s*https://doi\.org/10\.1101/[\d.]+doi:\s*bioRxiv preprint\s*\d*|bioRxiv preprint", re.I)
JOURNAL_LIKE = re.compile(r"\d+\s*\(\s*\d+\s*\)|\b\d+\s*:\s*\d+|\d+\s*[-–]\s*\d+|doi", re.I)


def norm(s):
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"[^a-z0-9 ]", " ", re.sub(r"\s+", " ", s)).strip()


def year_of(ref):
    m = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", ref)
    return int(m.group(1)) if m else None


def refs_from_pdf(path):
    txt = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    txt = unicodedata.normalize("NFKC", txt)
    lines_all = txt.splitlines()
    numbered = sum(bool(re.search(r"\s\d{1,4}\s*$", l)) for l in lines_all)
    if lines_all and numbered / len(lines_all) > 0.3:          # manuscript line numbers (e.g., PeerJ preprints)
        txt = "\n".join(re.sub(r"\s+\d{1,4}\s*$", "", l) for l in lines_all)
    txt = BOILER.sub(" ", txt)
    m = list(re.finditer(r"\n\s*(References|REFERENCES|Literature Cited|LITERATURE CITED|Bibliography)\s*\n", txt))
    if not m:
        return []
    body = txt[m[-1].end():]
    body = re.split(r"\n\s*(Appendix|APPENDIX|Supplementary|Figure \d|Table \d)\b", body)[0]
    # numbered styles: [12] ... or 12. ...
    if len(re.findall(r"\[\d{1,3}\]\s", body)) >= 5:
        refs = [r.strip() for r in re.split(r"\[\d{1,3}\]\s", re.sub(r"\s+", " ", body))]
        return [r for r in refs if 40 < len(r) < 800]
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    refs, cur = [], ""
    start = re.compile(r"^(\[[A-Z][^\]]{2,60}\d{4}[a-z]?\]\s*)?[A-Z][A-Za-zÀ-ÿ'\-]+,?\s+(?:[A-Z]\.|[A-Z][a-z]+)")
    for l in lines:
        ends_ref = bool(re.search(r"([.)\]]|\d)\s*$", cur))
        if start.match(l) and cur and ends_ref and re.search(r"(19|20)\d\d", cur):
            refs.append(cur); cur = l
        else:
            cur = (cur + " " + l).strip()
    if cur:
        refs.append(cur)
    return [r for r in refs if 40 < len(r) < 800]


def refs_from_ris(path):
    out, title, year = [], None, None
    for l in open(path, errors="ignore"):
        tag, _, val = l.partition("  - ")
        tag = tag.strip()
        if tag in ("TI", "T1"):
            title = val.strip()
        elif tag in ("PY", "Y1", "DA") and not year:
            year = year_of(val)
        elif tag == "ER":
            if title:
                out.append(dict(ref=title, title=title, year=year, journal_like=True))
            title, year = None, None
    return out


def refs_from_bib(path):
    txt = open(path, errors="ignore").read()
    out = []
    for e in re.split(r"\n@", txt):
        t = re.search(r"\btitle\s*=\s*[{\"](.+?)[}\"]\s*,?\s*\n", e, re.S | re.I)
        y = re.search(r"\byear\s*=\s*[{\"]?(\d{4})", e, re.I)
        if t:
            title = re.sub(r"[{}]", "", t.group(1))
            out.append(dict(ref=title, title=title, year=int(y.group(1)) if y else None, journal_like=True))
    return out


def match(refs, titles_norm, cutoff=88):
    """Return matched corpus index per ref (or None). Uses partial_ratio so a title inside a longer
    reference string still matches; very short titles are excluded to avoid false positives."""
    res = []
    for r in refs:
        q = norm(r.get("title") or r["ref"])
        if len(q) < 25:
            res.append(None); continue
        scorer = fuzz.token_set_ratio if r.get("title") else fuzz.partial_ratio
        hit = process.extractOne(q, titles_norm, scorer=scorer, score_cutoff=cutoff)
        res.append(hit[2] if hit else None)
    return res


def main():
    oa = pd.read_parquet(CORPUS / "corpus_openalex.parquet", columns=["title"])
    full = pd.read_parquet(CORPUS / "corpus.parquet", columns=["title", "source"]) \
        if (CORPUS / "corpus.parquet").exists() else oa.assign(source="openalex")
    oa_t = [t for t in oa["title"].map(norm) if len(t) >= 25]
    full_t = [t for t in full["title"].map(norm) if len(t) >= 25]

    gold_files = sorted(p for p in RECALL.iterdir() if p.suffix.lower() in (".pdf", ".ris", ".bib"))
    if not gold_files:
        raise SystemExit("No gold sources in data/recall/ (run download_papers.sh or add .pdf/.ris/.bib files).")
    rows, unmatched = [], []
    for p in gold_files:
        if p.suffix.lower() == ".pdf":
            refs = [dict(ref=r, title=None, year=year_of(r), journal_like=bool(JOURNAL_LIKE.search(r))) for r in refs_from_pdf(p)]
        elif p.suffix.lower() == ".ris":
            refs = refs_from_ris(p)
        else:
            refs = refs_from_bib(p)
        scope_any = [r for r in refs if r["year"] and 2005 <= r["year"] <= 2025 and r["journal_like"]]
        # gold set = Philippine studies only (Philippine place/name in the reference), same scope as the corpus
        scope = [r for r in scope_any if PH_TERMS.search(r["ref"]) or p.suffix.lower() != ".pdf"]
        m_oa, m_full = match(scope, oa_t), match(scope, full_t)
        n = len(scope)
        rows.append(dict(gold_source=p.stem, refs_parsed=len(refs), refs_2005_2025=len(scope_any), in_scope_ph_2005_2025=n,
                         found_openalex=sum(x is not None for x in m_oa),
                         found_merged=sum(x is not None for x in m_full),
                         recall_openalex=(sum(x is not None for x in m_oa) / n) if n else None,
                         recall_merged=(sum(x is not None for x in m_full) / n) if n else None))
        unmatched += [dict(gold_source=p.stem, year=r["year"], reference=r["ref"][:400])
                      for r, x in zip(scope, m_full) if x is None]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "corpus_recall.csv", index=False)
    pd.DataFrame(unmatched).to_csv(RESULTS / "corpus_recall_unmatched.csv", index=False)
    print(out.round(3).to_string(index=False))
    print("\nUnmatched references -> results/corpus_recall_unmatched.csv")
    print("Review them: many are expected misses outside scope (e.g., taxonomy notes in non-indexed outlets, "
          "or out-of-subfield work). Report recall for in-scope journal articles only.")


if __name__ == "__main__":
    main()
