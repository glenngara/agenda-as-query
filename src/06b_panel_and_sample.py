"""Step 6b (consensus) / optional sample: AI-panel consensus labels (+ stratified sample for a reduced-annotation variant).
In the main design run with --no-sample: only the panel consensus is built; the lead expert (A1) labels the full pool.

1. Merges the LLM judges (claude, haiku, gpt; whichever exist) into consensus labels
   = median grade -> data/qrels/labels_panel.csv. Reports inter-judge agreement.
2. Draws the expert sample (EXPERT_SAMPLE pairs) stratified by topic and by consensus grade,
   over-sampling pairs where judges disagree by >=2 grades, so the expert check is informative.
   -> data/pool/expert_sample.csv   (the labeling app shows only these pairs to the lead expert (A1))
Usage: python 06b_panel_and_sample.py [--n 100] [--seed 7]
"""
import argparse, itertools
import numpy as np
import pandas as pd
from config import QRELS, POOL, EXPERT_SAMPLE
from metrics import agreement

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=EXPERT_SAMPLE)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--no-sample", action="store_true")
a = ap.parse_args()

judges = {}
for name in ["llm_claude", "llm_haiku", "llm_gpt", "llm_ollama"]:
    f = QRELS / f"labels_{name}.csv"
    if f.exists():
        d = pd.read_csv(f).dropna(subset=["grade"])
        judges[name] = d.set_index("pair_id")["grade"].astype(int)
if len(judges) < 2:
    raise SystemExit("Need at least 2 LLM judges (run make judge-all).")

G = pd.DataFrame(judges).dropna()
pool = pd.read_csv(POOL / "pool.csv").set_index("pair_id")
panel = pd.DataFrame({
    "pair_id": G.index,
    "topic_id": pool.loc[G.index, "topic_id"].values,
    "doc_id": pool.loc[G.index, "doc_id"].values,
    "grade": np.floor(G.median(axis=1)).astype(int).values,   # median; ties (2 judges) round down
    "spread": (G.max(axis=1) - G.min(axis=1)).values,
})
panel.to_csv(QRELS / "labels_panel.csv", index=False)
print(f"Panel labels: {len(panel)} pairs from judges {list(judges)}")
print(panel["grade"].value_counts().sort_index().to_string())

rows = [dict(a=x, b=y, **agreement(G[x], G[y])) for x, y in itertools.combinations(G.columns, 2)]
print("\nInter-judge agreement:")
print(pd.DataFrame(rows).round(3).to_string(index=False))

if a.no_sample:
    raise SystemExit(0)
# ---- stratified expert sample ----
rng = np.random.default_rng(a.seed)
panel["disagree"] = (panel["spread"] >= 2).astype(int)
n_dis = min(int(round(a.n * 0.2)), int(panel["disagree"].sum()))
dis = panel[panel["disagree"] == 1].sample(n_dis, random_state=a.seed) if n_dis else panel.iloc[:0]
rest = panel.drop(dis.index)
need = a.n - len(dis)
# equal share per consensus grade (as available), spread across topics within grade
per_grade = {g: need // 4 for g in range(4)}
for g in range(need % 4):
    per_grade[3 - g] += 1
picks = []
for g in range(4):
    sub = rest[rest["grade"] == g].sample(frac=1.0, random_state=a.seed)
    sub = sub.assign(_r=sub.groupby("topic_id").cumcount()).sort_values("_r")  # round-robin over topics
    picks.append(sub.head(per_grade[g]))
sample = pd.concat([dis] + picks)
if len(sample) < a.n:  # top up if some grade was scarce
    sample = pd.concat([sample, rest.drop(sample.index, errors="ignore").sample(a.n - len(sample), random_state=a.seed)])
sample = sample.sample(frac=1.0, random_state=a.seed)
out = pool.loc[sample["pair_id"]].reset_index()[["pair_id", "topic_id", "doc_id", "title", "abstract", "year"]]
out.to_csv(POOL / "expert_sample.csv", index=False)
print(f"\nExpert sample: {len(out)} pairs over {out['topic_id'].nunique()} topics "
      f"({n_dis} judge-disagreement pairs) -> data/pool/expert_sample.csv")
print("Next: make annotate  (the lead expert (A1))")
