"""Step 6c: stability of the LLM judges. Re-grades the 60 shared (overlap) pairs N more times per judge
and reports self-agreement across runs and the spread of agreement with the expert.

Usage: python 06c_judge_stability.py claude haiku gpt   [--runs 2]
Output: data/qrels/stability/labels_llm_<judge>_run<k>.csv  and  results/judge_stability.csv
Run 1 is the main judgment file (data/qrels/labels_llm_<judge>.csv); this adds runs 2..N+1.
"""
import sys, json, time, importlib, itertools
import numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from config import POOL, TOPICS, QRELS, RESULTS, HAIKU_MODEL, PRIMARY_EXPERT
from llm import claude, gpt

J = importlib.import_module("06_llm_judge")
OUT = QRELS / "stability"; OUT.mkdir(exist_ok=True)


def ac1(a, b, cats=(0, 1, 2, 3)):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pi = np.array([((a == c).mean() + (b == c).mean()) / 2 for c in cats])
    pe = (pi * (1 - pi)).sum() / (len(cats) - 1)
    return (po - pe) / (1 - pe)


def stats(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    return dict(n=len(a), exact=(a == b).mean(),
                kappa_q=cohen_kappa_score(a, b, weights="quadratic", labels=[0, 1, 2, 3]), ac1=ac1(a, b))


def caller(judge):
    if judge == "haiku":
        return lambda p, **kw: claude(p, model=HAIKU_MODEL, **kw)
    return claude if judge == "claude" else gpt


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 2
    judges = [a for a in args if a in ("claude", "haiku", "gpt")] or ["claude", "haiku", "gpt"]
    pool = pd.read_csv(POOL / ("pool.csv" if (POOL / "pool.csv").exists() else "pool_ids.csv"))  # ids suffice when re-runs exist
    pool = pool[pool["overlap"] == 1]
    topics = {t["topic_id"]: t for t in map(json.loads, open(TOPICS / "topics.jsonl"))}
    expert = pd.read_csv(QRELS / f"labels_{PRIMARY_EXPERT}.csv")[["pair_id", "grade"]]
    rows = []
    for judge in judges:
        call = caller(judge)
        main_f = QRELS / f"labels_llm_{judge}.csv"
        if not main_f.exists():
            print(f"skip {judge}: run `make judge` for it first"); continue
        files = [main_f]
        for k in range(2, runs + 2):
            f = OUT / f"labels_llm_{judge}_run{k}.csv"; files.append(f)
            done = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["pair_id"])
            todo = pool[~pool["pair_id"].isin(set(done["pair_id"]))]
            new = []
            for r in tqdm(list(todo.itertuples()), desc=f"{judge} run{k}"):
                t = topics[r.topic_id]
                p = J.build_prompt(t, r)
                kw = dict(system=J.SYS, kind=f"stability_{judge}_run{k}")
                out = call(p, json_mode=True, **kw) if judge == "gpt" else call(p, **kw)
                new.append({"pair_id": r.pair_id, "topic_id": r.topic_id, "doc_id": r.doc_id,
                            "grade": J.parse(out), "raw": out.strip()[:500], "ts": time.time()})
            if new:
                pd.concat([done, pd.DataFrame(new)]).to_csv(f, index=False)
        L = [pd.read_csv(f)[["pair_id", "grade"]].dropna() for f in files]
        L = [d[d.pair_id.isin(pool.pair_id)] for d in L]
        for (i, a), (j, b) in itertools.combinations(enumerate(L, 1), 2):
            m = a.merge(b, on="pair_id")
            rows.append(dict(judge=judge, compare=f"run{i} vs run{j}", **stats(m.grade_x, m.grade_y)))
        for i, a in enumerate(L, 1):
            m = expert.merge(a, on="pair_id")
            rows.append(dict(judge=judge, compare=f"expert vs run{i}", **stats(m.grade_x, m.grade_y)))
        # majority vote over runs
        mv = L[0].rename(columns={"grade": "g1"})
        for i, d in enumerate(L[1:], 2):
            mv = mv.merge(d.rename(columns={"grade": f"g{i}"}), on="pair_id")
        mv["grade"] = mv.filter(like="g").median(axis=1).round().astype(int)
        m = expert.merge(mv[["pair_id", "grade"]], on="pair_id")
        rows.append(dict(judge=judge, compare="expert vs majority", **stats(m.grade_x, m.grade_y)))
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "judge_stability.csv", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
