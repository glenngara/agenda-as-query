"""Step 2: build TREC-style topics from the policy documents.

Parses the 23 PBSAP 2024-2040 national targets (PH GBF national-targets PDF) into
data/topics/topics_draft.xlsx. The lead expert (A1) edits that file:
  - query      : a short expert search query (5-12 words)
  - narrative  : what counts as relevant / not relevant (2-4 sentences)
  - include    : 1 = use this topic, 0 = drop
HNRDA rows can also be added at the bottom (topic_id like HNRDA-01).
Then `make topics-final` converts it to data/topics/topics.jsonl.
"""
import re, sys, json, unicodedata, logging
logging.getLogger('pypdf').setLevel(logging.ERROR)
import pandas as pd
from pypdf import PdfReader
from config import POLICY, TOPICS

SRC = POLICY / "PH_GBF_National_Targets_2025.pdf"
HEAD = re.compile(r"PBSAP Target (\d+):\s*([^\n]+)")


def pdf_text(path):
    txt = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    return unicodedata.normalize("NFKC", txt)


def section(block, name, stops):
    m = re.search(rf"\n{name}\s*\n(.*?)(?=\n(?:{'|'.join(stops)})\s*\n|\Z)", block, re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def draft():
    txt = pdf_text(SRC)
    heads = list(HEAD.finditer(txt))
    rows = []
    stops = ["Main policy measures", "Global goals", "Global targets", "Degree of alignment by target",
             "Aspects covered", "Description"]
    for i, h in enumerate(heads):
        block = txt[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(txt)]
        n = int(h.group(1))
        rows.append(dict(
            topic_id=f"PBSAP-{n:02d}",
            source="PBSAP 2024-2040",
            title=h.group(2).strip(),
            description=section(block, "Description", stops),
            actions=section(block, "Main policy measures", stops)[:2500],
            gbf_targets=", ".join(sorted(set(re.findall(r"GBF-TARGET-\d+", block)))),
            query="", narrative="", include=1,
        ))
    df = pd.DataFrame(rows).sort_values("topic_id").drop_duplicates("topic_id")
    out = TOPICS / "topics_draft.xlsx"
    if out.exists() and "--force" not in sys.argv:
        print(f"[skip] {out} exists (use --force to overwrite the lead expert's edits!)")
        return
    with pd.ExcelWriter(out) as xw:
        df.to_excel(xw, index=False, sheet_name="topics")
        pd.DataFrame({"how_to": [
            "Fill 'query' with a short expert search query (5-12 words).",
            "Fill 'narrative': what makes a paper relevant vs not relevant to this target.",
            "Set include=0 for targets with no plausible research literature (e.g., resource mobilization).",
            "Add HNRDA biodiversity/environment items as new rows (topic_id HNRDA-01, source 'HNRDA 2022-2028').",
            "Do not change topic_id values once judging has started.",
        ]}).to_excel(xw, index=False, sheet_name="instructions")
    print(f"Wrote {len(df)} topics -> {out}")


def final():
    df = pd.read_excel(TOPICS / "topics_draft.xlsx", sheet_name="topics").fillna("")
    df = df[df["include"].astype(str).isin(["1", "1.0", "True"])]
    with open(TOPICS / "topics.jsonl", "w") as f:
        for r in df.to_dict("records"):
            f.write(json.dumps({k: (str(v) if k != "include" else 1) for k, v in r.items()}) + "\n")
    missing = df[df["query"].astype(str).str.strip() == ""]["topic_id"].tolist()
    print(f"Wrote {len(df)} topics -> {TOPICS/'topics.jsonl'}")
    if missing:
        print(f"Note: no expert query yet for {len(missing)} topics: {missing} (expert condition skipped for these)")


if __name__ == "__main__":
    final() if "--final" in sys.argv else draft()
