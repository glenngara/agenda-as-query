"""Step 7b: extended evaluation used in the paper (run after 07_evaluate.py).

(a) Agreement with the expert: exact, quadratic kappa, Gwet AC1, kappa/PABAK for "highly relevant" (grade 3),
    and mean grades on Philippine vs non-Philippine papers  -> results/agreement_extended.csv
(b) Retrieval effectiveness on condensed lists (unjudged documents removed; Sakai 2007), nDCG@10 and
    P@10 with grade 3 = relevant, bootstrap CIs, ranking agreement under each label source
    -> results/retrieval_expert_condensed.csv, results/system_ranking_tau_condensed.csv
(c) Pool evidence per goal: grade-3 papers and grade-3 Philippine papers per goal -> results/pool_evidence_by_goal.csv
Philippine-study flag (non_ph) = keyword match on title+abstract (Philippine places, agencies, languages), checked by hand on the pool.
"""
import glob, os, json
import numpy as np, pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score as K
from config import QRELS, RESULTS, RUNS, TOPICS

rng = np.random.default_rng(0)
E = pd.read_csv(QRELS / "labels_A1.csv")


def ac1(a, b, c=(0, 1, 2, 3)):
    a, b = np.array(a), np.array(b); po = (a == b).mean()
    pi = np.array([((a == x).mean() + (b == x).mean()) / 2 for x in c]); pe = (pi * (1 - pi)).sum() / (len(c) - 1)
    return (po - pe) / (1 - pe)


# (a) agreement
src = {"A1_repeat": "labels_A1_repeat.csv", "A2": "labels_A2.csv", "A3": "labels_A3.csv",
       "A4": "labels_A4.csv", "llm_gpt": "labels_llm_gpt.csv", "llm_claude": "labels_llm_claude.csv",
       "llm_haiku": "labels_llm_haiku.csv", "panel": "labels_panel.csv", "llm_claude_nocontext": "labels_llm_claude_nocontext.csv"}
overlap = set(pd.read_csv(QRELS / "labels_A2.csv").pair_id)
rows = []
for n, f in src.items():
    if not (QRELS / f).exists():
        continue
    d = pd.read_csv(QRELS / f)[["pair_id", "grade"]].dropna()
    for scope in (["all", "shared60"] if len(d) > 60 else ["shared60"]):
        m = E.merge(d, on="pair_id", suffixes=("_e", "_j"))
        if scope == "shared60":
            m = m[m.pair_id.isin(overlap)]
        a, b = m.grade_e.astype(int), m.grade_j.astype(int)
        ph, nph = m[m.non_ph == 0], m[m.non_ph == 1]
        rows.append(dict(judge=n, scope=scope, n=len(m), exact=np.mean(a == b),
                         kappa_q=K(a, b, weights="quadratic", labels=[0, 1, 2, 3]), ac1=ac1(a, b),
                         kappa_rel3=K(a >= 3, b >= 3), pabak_rel3=2 * np.mean((a >= 3) == (b >= 3)) - 1, mean_diff=np.mean(b - a),
                         judge_mean_ph=ph.grade_j.mean(), judge_mean_nonph=nph.grade_j.mean(),
                         expert_mean_ph=ph.grade_e.mean(), expert_mean_nonph=nph.grade_e.mean(),
                         judge_grade3_rate_nonph=(nph.grade_j == 3).mean(), expert_grade3_rate_nonph=(nph.grade_e == 3).mean()))
pd.DataFrame(rows).to_csv(RESULTS / "agreement_extended.csv", index=False)


# (b) condensed retrieval evaluation
def qd(f):
    d = pd.read_csv(QRELS / f).dropna(subset=["grade"]); out = {}
    for r in d.itertuples():
        out.setdefault(r.topic_id, {})[r.doc_id] = int(r.grade)
    return out


def ndcg(ranked, rels, k=10):
    g = [rels.get(d, 0) for d in ranked[:k]]; dcg = sum(x / np.log2(i + 2) for i, x in enumerate(g))
    ideal = sorted(rels.values(), reverse=True)[:k]; idcg = sum(x / np.log2(i + 2) for i, x in enumerate(ideal))
    return dcg / idcg if idcg > 0 else np.nan


runs = {os.path.basename(f)[:-4]: pd.read_csv(f, sep="\t") for f in sorted(glob.glob(str(RUNS / "*.tsv")))}


def evaluate(qrels):
    rows = []
    for name, run in runs.items():
        for tid, g in run.sort_values(["topic_id", "rank"]).groupby("topic_id"):
            if tid not in qrels:
                continue
            rels = qrels[tid]; full = g.doc_id.tolist(); ranked = [d for d in full if d in rels]
            rows.append(dict(run=name, topic_id=tid, ndcg10=ndcg(ranked, rels),
                             p10_rel3=np.mean([rels.get(d, 0) >= 3 for d in ranked[:10]]) if ranked else np.nan,
                             judged10=np.mean([d in rels for d in full[:10]])))
    return pd.DataFrame(rows)


pt = evaluate(qd("labels_A1.csv"))
(RESULTS / "per_topic").mkdir(exist_ok=True)
pt.to_csv(RESULTS / "per_topic" / "retrieval_expert_condensed_per_topic.csv", index=False)
tab = []
for name, g in pt.groupby("run"):
    x = g.ndcg10.dropna().values; b = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    tab.append(dict(run=name, ndcg10_condensed=x.mean(), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5),
                    p10_rel3_condensed=g.p10_rel3.mean(), judged10=g.judged10.mean()))
T = pd.DataFrame(tab).sort_values("ndcg10_condensed", ascending=False)
T.to_csv(RESULTS / "retrieval_expert_condensed.csv", index=False)
base = T.set_index("run").ndcg10_condensed
out = []
for n in ["llm_gpt", "llm_claude", "llm_haiku", "panel", "llm_claude_nocontext"]:
    if not (QRELS / f"labels_{n}.csv").exists():
        continue
    o = evaluate(qd(f"labels_{n}.csv")).groupby("run").ndcg10.mean(); tau, p = kendalltau(base, o.reindex(base.index))
    out.append(dict(labels=n, kendall_tau=tau, p=p, top_system=o.idxmax(), same_top=o.idxmax() == base.idxmax()))
pd.DataFrame(out).to_csv(RESULTS / "system_ranking_tau_condensed.csv", index=False)

# (c) pool evidence per goal
titles = {t["topic_id"]: t["title"] for t in map(json.loads, open(TOPICS / "topics.jsonl"))}
g = E.groupby("topic_id").agg(pairs=("grade", "size"), grade3=("grade", lambda s: (s == 3).sum()),
                              nonph=("non_ph", "sum"))
g["grade3_ph"] = E[(E.grade == 3) & (E.non_ph == 0)].groupby("topic_id").size().reindex(g.index, fill_value=0)
g["goal"] = g.index.map(titles)
g.sort_values("grade3_ph").to_csv(RESULTS / "pool_evidence_by_goal.csv")
print("wrote agreement_extended, retrieval_expert_condensed, system_ranking_tau_condensed, pool_evidence_by_goal")

# (d) significance agreement with the expert (paired t-test, alpha = 0.05, 105 run pairs)
import itertools
from scipy.stats import ttest_rel


def sig_conclusions(labels):
    piv = evaluate(qd(labels)).pivot(index="topic_id", columns="run", values="ndcg10")
    out = {}
    for x, y in itertools.combinations(sorted(piv.columns), 2):
        d = piv[[x, y]].dropna(); t, p = ttest_rel(d[x], d[y])
        out[(x, y)] = (p < 0.05, np.sign(t))
    return out


ref = sig_conclusions("labels_A1.csv"); rows = []
for n in ["llm_gpt", "llm_claude", "llm_haiku", "panel", "llm_claude_nocontext"]:
    if not (QRELS / f"labels_{n}.csv").exists():
        continue
    S = sig_conclusions(f"labels_{n}.csv")
    agree = sum(1 for k in ref if ref[k][0] == S[k][0] and (not ref[k][0] or ref[k][1] == S[k][1]))
    swaps = sum(1 for k in ref if ref[k][0] and S[k][0] and ref[k][1] != S[k][1])
    rows.append(dict(labels=n, sig_pairs_expert=sum(v[0] for v in ref.values()), sig_pairs_judge=sum(v[0] for v in S.values()),
                     conclusion_agreement=agree / len(ref), swaps=swaps))
pd.DataFrame(rows).to_csv(RESULTS / "significance_agreement.csv", index=False)

# (e) share of Philippine studies in each run's top ten (tag: data/corpus/ph_tag.csv, terms: docs/philippine_terms.txt)
ph = dict(pd.read_csv(RUNS.parent / "corpus" / "ph_tag.csv").values)
rows = []
for name, run in runs.items():
    top = run[run["rank"] <= 10]
    rows.append(dict(run=name, ph_share_top10=top.doc_id.map(ph).mean()))
pd.DataFrame(rows).to_csv(RESULTS / "ph_share_top10_by_run.csv", index=False)
print("wrote significance_agreement, ph_share_top10_by_run")
