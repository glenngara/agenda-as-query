"""Step 3: generate query formulations per topic.

qtype      | text(s) used as the query                                   | source
-----------|--------------------------------------------------------------|---------------
title      | target title only                                            | policy doc
desc       | title + official description                                 | policy doc
verbose    | title + description + action list (long, policy-style text)  | policy doc
expert     | the lead expert's short query                                       | domain expert
decomp     | 3-5 LLM sub-queries, fused with RRF at retrieval time        | GPT (OpenAI)
hyde       | LLM-written hypothetical research abstract                   | GPT (OpenAI)

GPT generates queries; Claude judges (different vendors: avoids generator=judge circularity).
Output: data/queries/queries.jsonl
"""
import json, re, sys
from tqdm import tqdm
from config import TOPICS, QUERIES, OPENAI_API_KEY, OPENAI_MODEL
from llm import gpt


def llm_ready():
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("PASTE_"):
        print("[!] OPENAI_API_KEY is not set in .env. Paste your key, then re-run `make queries`.")
        return False
    return True


SYS = "You are an ecologist and information retrieval expert helping search the scientific literature."

DECOMP = """A national biodiversity policy target is given below. Write 3 to 5 short search queries
(4-10 words each) that would retrieve PEER-REVIEWED ECOLOGY OR ENVIRONMENTAL SCIENCE RESEARCH relevant to
this target in the Philippines. Cover different facets (species/ecosystems, threats, methods, management).
Return JSON: {{"queries": ["...", "..."]}}

Target: {title}
Description: {description}"""

HYDE = """A national biodiversity policy target is given below. Write the abstract (120-180 words) of a
plausible peer-reviewed research paper from the Philippines that would directly inform this target.
Use the style of a scientific abstract (aim, methods, results, implications). Output only the abstract.

Target: {title}
Description: {description}"""


def main():
    topics = [json.loads(l) for l in open(TOPICS / "topics.jsonl")]
    out_path = QUERIES / "queries.jsonl"
    done = set()
    if out_path.exists():
        done = {(q["topic_id"], q["qtype"]) for q in map(json.loads, open(out_path))}
    with open(out_path, "a") as f:
        def put(t, qtype, texts):
            if (t["topic_id"], qtype) in done or not texts:
                return
            f.write(json.dumps({"topic_id": t["topic_id"], "qtype": qtype, "texts": texts}) + "\n")
            f.flush(); done.add((t["topic_id"], qtype))

        # 1) policy-text formulations for ALL topics first (no LLM needed)
        for t in topics:
            desc = t.get("description", "")
            put(t, "title", [t["title"]])
            put(t, "desc", [f"{t['title']}. {desc}"])
            put(t, "verbose", [f"{t['title']}. {desc} {t.get('actions', '')}"[:4000]])
            if t.get("query", "").strip():
                put(t, "expert", [t["query"].strip()])

        # 2) LLM formulations (GPT)
        if not llm_ready():
            print("Policy-text queries written; LLM queries (decomp, hyde) NOT generated.")
            sys.exit(1)
        for t in tqdm(topics, desc="LLM queries"):
            if (t["topic_id"], "decomp") not in done:
                raw = gpt(DECOMP.format(**t), system=SYS, kind="decomp", json_mode=True)
                try:
                    qs = [q for q in json.loads(raw).get("queries", []) if isinstance(q, str)][:5]
                except json.JSONDecodeError:
                    qs = [l.strip("-• ").strip() for l in raw.splitlines() if l.strip()][:5]
                put(t, "decomp", qs)
            if (t["topic_id"], "hyde") not in done:
                put(t, "hyde", [gpt(HYDE.format(**t), system=SYS, kind="hyde").strip()])
    n = sum(1 for _ in open(out_path))
    print(f"Queries -> {out_path}  ({n} query sets for {len(topics)} topics)")


if __name__ == "__main__":
    main()
