"""Step 5: build the judging pool (TREC-style pooling).

Union of the top POOL_DEPTH docs from every run, per topic. Order is shuffled and
system provenance is hidden from annotators. A stratified subset is flagged for the
second annotator (inter-annotator agreement).
Usage: python 05_build_pool.py [--depth 10] [--overlap 60] [--max-per-topic 25]
Output: data/pool/pool.csv, data/pool/pool_provenance.csv
"""
import argparse, json
import numpy as np
import pandas as pd
from config import RUNS, POOL, CORPUS, TOPICS, POOL_DEPTH

ap = argparse.ArgumentParser()
ap.add_argument("--depth", type=int, default=POOL_DEPTH)
ap.add_argument("--overlap", type=int, default=60)
ap.add_argument("--max-per-topic", type=int, default=20,
                help="cap pool size per topic to keep judging feasible (keeps docs found by most systems)")
ap.add_argument("--seed", type=int, default=13)
a = ap.parse_args()

topics = {t["topic_id"] for t in map(json.loads, open(TOPICS / "topics.jsonl"))}
prov = []
for f in sorted(RUNS.glob("*.tsv")):
    run = f.stem
    df = pd.read_csv(f, sep="\t")
    df = df[(df["rank"] <= a.depth) & df["topic_id"].isin(topics)]
    df["run"] = run
    prov.append(df[["topic_id", "doc_id", "run", "rank"]])
prov = pd.concat(prov)
if prov["topic_id"].nunique() < len(topics):
    print(f"[!] Runs cover only {prov['topic_id'].nunique()} of {len(topics)} topics. Re-run `make queries` and `make retrieve` first.")
    raise SystemExit(1)
prov.to_csv(POOL / "pool_provenance.csv", index=False)

# votes = how many runs retrieved the doc in their top-depth; best rank as tie-break
agg = prov.groupby(["topic_id", "doc_id"]).agg(votes=("run", "nunique"), best_rank=("rank", "min")).reset_index()
agg = agg.sort_values(["topic_id", "votes", "best_rank"], ascending=[True, False, True])
agg = agg.groupby("topic_id").head(a.max_per_topic)

rng = np.random.default_rng(a.seed)
agg = agg.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
agg["pair_id"] = [f"P{i:05d}" for i in range(len(agg))]

# stratified overlap subset for 2nd annotator
per_topic = max(1, a.overlap // max(1, agg["topic_id"].nunique()))
ov = agg.groupby("topic_id").head(per_topic)["pair_id"]
if len(ov) < a.overlap:
    rest = agg[~agg["pair_id"].isin(ov)].head(a.overlap - len(ov))["pair_id"]
    ov = pd.concat([ov, rest])
agg["overlap"] = agg["pair_id"].isin(ov).astype(int)

corpus = pd.read_parquet(CORPUS / "corpus.parquet", columns=["doc_id", "title", "abstract", "year"])
pool = agg.merge(corpus, on="doc_id", how="left")[["pair_id", "topic_id", "doc_id", "title", "abstract", "year", "overlap"]]
pool.to_csv(POOL / "pool.csv", index=False)
print(f"Pool: {len(pool)} pairs over {pool['topic_id'].nunique()} topics "
      f"(avg {len(pool)/pool['topic_id'].nunique():.1f}/topic); overlap subset: {pool['overlap'].sum()}")
print("Next: make annotate (the lead expert (A1), full pool) and, in parallel, make judge-all")
