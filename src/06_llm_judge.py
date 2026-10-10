"""Step 6: LLM relevance judgments on the same pool (UMBRELA-style prompt, adapted).

Usage: python 06_llm_judge.py claude      -> data/qrels/labels_llm_claude.csv   (Sonnet)
       python 06_llm_judge.py haiku       -> data/qrels/labels_llm_haiku.csv    (Haiku)
       python 06_llm_judge.py gpt         -> data/qrels/labels_llm_gpt.csv      (OpenAI)
Default prompt = matched to the human grading guide (PBSAP context + country rule).
Add --nocontext for the ablation without Philippine context or the country rule.
Add --narrative to give the LLM the expert's relevance narrative (an ablation).
Resumable: already-judged pairs are skipped.
"""
import sys, json, re, time
import pandas as pd
from tqdm import tqdm
from config import POOL, TOPICS, QRELS, HAIKU_MODEL
from llm import claude, ollama, gpt

SYS_NOCONTEXT = ("You are an expert ecologist acting as a relevance assessor for a test collection that links "
       "national biodiversity policy targets to scientific research.")

PROMPT_NOCONTEXT = """Given a national biodiversity policy target and a research paper (title and abstract), judge how
relevant the paper is to the target on this scale:

3 = Highly relevant: the paper directly studies the target's subject; its findings could inform how the target is met or monitored.
2 = Relevant: the paper clearly addresses the target's subject, but indirectly or only in part.
1 = Marginal: the paper mentions related concepts but would not meaningfully inform the target.
0 = Not relevant: unrelated to the target.

Think step by step, internally: (a) what the target is about, (b) what the paper studies, (c) how well they match.
Then answer ONLY with JSON: {{"grade": <0-3>, "reason": "<one sentence>"}}

POLICY TARGET: {title}
TARGET DESCRIPTION: {description}{narr}

PAPER TITLE: {ptitle}
PAPER ABSTRACT: {abstract}"""


# Main prompt: matched to the human grading guide (same context, question, scale and rules the graders saw).
SYS = ("You are an expert ecologist acting as a relevance grader for a study of the Philippine Biodiversity "
       "Strategy and Action Plan (PBSAP) 2024-2040.")

PROMPT = """What the study is about. The Philippine Biodiversity Strategy and Action Plan (PBSAP) 2024 to 2040 has 23
national biodiversity goals. We want to know how much scientific research exists for each goal. We collected 17,242 papers on
Philippine ecology and environment (2005 to 2025). For each goal, computer search tools picked 20 papers that may match,
giving 460 goal-and-paper pairs.

How to grade. You will see one goal and one paper (title and abstract). Ask: would this paper help someone working on
this goal? Give one grade:
3 = highly relevant: directly about the goal and could help meet or monitor it.
2 = relevant: on topic, but only partly or indirectly.
1 = marginal: shares some words, but would not really help.
0 = not relevant: not related to the goal.

Rules.
1. Grade how related the paper is to the goal, not how good the paper is.
2. Use only the title and abstract.
3. A study outside the Philippines can get a 2. Give 3 mainly to Philippine studies or studies that clearly apply to the Philippines.

Answer ONLY with JSON: {{"grade": <0-3>, "reason": "<one sentence>"}}

GOAL: {title}
{description}
{actions}{narr}

PAPER TITLE: {ptitle}
{year}
PAPER ABSTRACT: {abstract}"""


def build_prompt(t, r, nocontext=False, narr=""):
    """Same fields the human graders saw: goal title, description, action list; paper title, year, full abstract."""
    ab = str(r.abstract) if str(r.abstract) not in ("", "nan") else "No abstract available. Grade from the title."
    if nocontext:
        return PROMPT_NOCONTEXT.format(title=t["title"], description=t.get("description", ""), narr=narr,
                                       ptitle=r.title, abstract=ab[:3500])
    yr = f"Published {int(r.year)}" if str(getattr(r, "year", "")) not in ("", "nan") else ""
    return PROMPT.format(title=t["title"], description=t.get("description", ""),
                         actions=t.get("actions") or "No action list.", narr=narr,
                         ptitle=r.title, year=yr, abstract=ab)


def parse(out):
    m = re.search(r'"grade"\s*:\s*([0-3])', out or "")   # strict: no loose digit fallback
    return int(m.group(1)) if m else None


def main():
    judge = (sys.argv[1] if len(sys.argv) > 1 else "claude").lower()
    use_narr = "--narrative" in sys.argv
    nocontext = "--nocontext" in sys.argv   # ablation: no Philippine context, no country rule
    sys_p, prompt_t = (SYS_NOCONTEXT, PROMPT_NOCONTEXT) if nocontext else (SYS, PROMPT)
    tag = f"llm_{judge}" + ("_nocontext" if nocontext else "") + ("_narr" if use_narr else "")
    out_path = QRELS / f"labels_{tag}.csv"
    pool = pd.read_csv(POOL / "pool.csv")
    topics = {t["topic_id"]: t for t in map(json.loads, open(TOPICS / "topics.jsonl"))}
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["pair_id"])
    todo = pool[~pool["pair_id"].isin(set(done["pair_id"]))]
    if judge == "haiku":
        call = lambda p, **kw: claude(p, model=HAIKU_MODEL, **kw)
    elif judge == "claude":
        call = claude
    elif judge == "gpt":
        call = gpt
    else:
        call = ollama
    rows = []
    for i, r in enumerate(tqdm(todo.itertuples(), total=len(todo), desc=tag)):
        t = topics[r.topic_id]
        narr = f"\nRELEVANCE GUIDANCE: {t['narrative']}" if use_narr and t.get("narrative") else ""
        p = build_prompt(t, r, nocontext=nocontext, narr=narr)
        out = call(p, system=sys_p, kind=f"judge_{tag}", json_mode=True) if judge in ("ollama", "gpt") else \
            call(p, system=sys_p, kind=f"judge_{tag}")
        g = parse(out)
        rows.append({"pair_id": r.pair_id, "topic_id": r.topic_id, "doc_id": r.doc_id,
                     "grade": g, "raw": out.strip()[:500], "ts": time.time()})
        if len(rows) >= 20 or i == len(todo) - 1:
            pd.concat([done, pd.DataFrame(rows)]).to_csv(out_path, index=False)
            done = pd.read_csv(out_path); rows = []
    d = pd.read_csv(out_path)
    print(f"{tag}: {len(d)} judgments, {d['grade'].isna().sum()} unparseable -> {out_path}")
    print(d["grade"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
