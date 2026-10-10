"""Step 8: priority coverage estimates.

For each retriever (using one query formulation, default = best expert-qrels nDCG@10 for that retriever):
  1. calibrate a score threshold on expert labels (maximize F1 for grade>=2; AI panel if no expert labels);
  2. coverage(topic) = number of corpus docs in the top-1000 scoring >= threshold;
  3. 95% interval by bootstrapping the labeled pairs (threshold re-fitted each time);
  4. split covered docs by period and by authorship group (local / collaborative / foreign);
  5. stability: Kendall tau of topic coverage rankings between retrievers.
BM25 scores are max-normalized per topic; dense scores are cosine similarities.
Usage: python 08_coverage.py [--qtype desc] [--boot 500]
"""
import argparse, itertools, json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from config import RUNS, QRELS, RESULTS, CORPUS, TOPICS, TOP_K

ap = argparse.ArgumentParser()
ap.add_argument("--qtype", default=None)
ap.add_argument("--boot", type=int, default=500)
a = ap.parse_args()

_ef = QRELS / "labels_A1.csv"
expert = pd.read_csv(_ef).dropna(subset=["grade"]) if _ef.exists() else None
labels = expert if expert is not None else pd.read_csv(QRELS / "labels_panel.csv").dropna(subset=["grade"])
corpus = pd.read_parquet(CORPUS / "corpus.parquet", columns=["doc_id", "year", "author_group"])
topics = {t["topic_id"]: t["title"] for t in map(json.loads, open(TOPICS / "topics.jsonl"))}
eff = pd.read_csv(RESULTS / "retrieval_expert.csv")


def norm(run, retr):
    run = run.copy()
    if retr == "bm25":
        run["score"] = run["score"] / run.groupby("topic_id")["score"].transform("max").clip(lower=1e-9)
    return run


def best_threshold(scores, rel):
    order = np.argsort(-scores)
    s, y = scores[order], rel[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y); P = y.sum()
    if P == 0:
        return np.inf
    f1 = 2 * tp / (tp + fp + P)
    return s[int(np.argmax(f1))]


def coverage_counts(run, thr):
    c = run[run["score"] >= thr].groupby("topic_id").size()
    return c.reindex(sorted(run["topic_id"].unique()), fill_value=0)


summary, per_retr_cov = [], {}
period_rows, group_rows = [], []
for retr in sorted(eff["retriever"].unique()):
    qtype = a.qtype or eff[eff["retriever"] == retr].sort_values("ndcg10", ascending=False).iloc[0]["qtype"]
    f = RUNS / f"{retr}__{qtype}.tsv"
    if not f.exists():
        continue
    run = norm(pd.read_csv(f, sep="\t"), retr)
    lab = labels.merge(run[["topic_id", "doc_id", "score"]], on=["topic_id", "doc_id"], how="inner")
    if len(lab) < 20:
        print(f"{retr}: too few labeled pairs in run ({len(lab)}); skipping"); continue
    y = (lab["grade"] >= 2).astype(int).values
    sc = lab["score"].values
    thr = best_threshold(sc, y)
    cov = coverage_counts(run, thr)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(a.boot):
        i = rng.integers(0, len(lab), len(lab))
        boots.append(coverage_counts(run, best_threshold(sc[i], y[i])).values)
    boots = np.array(boots)
    df = pd.DataFrame({"topic_id": cov.index, "title": [topics.get(t, "") for t in cov.index],
                       "coverage": cov.values, "lo": np.percentile(boots, 2.5, 0), "hi": np.percentile(boots, 97.5, 0),
                       "at_cap": cov.values >= TOP_K})
    df.to_csv(RESULTS / f"coverage_{retr}.csv", index=False)
    per_retr_cov[retr] = df.set_index("topic_id")["coverage"]
    prec = (y[sc >= thr]).mean() if (sc >= thr).any() else np.nan
    rec = (y[sc >= thr]).sum() / max(1, y.sum())
    exp_prec = np.nan
    if expert is not None:
        ex = expert.merge(run[["topic_id", "doc_id", "score"]], on=["topic_id", "doc_id"])
        above = ex[ex["score"] >= thr]
        exp_prec = (above["grade"] >= 2).mean() if len(above) else np.nan
    summary.append(dict(retriever=retr, qtype=qtype, threshold=thr, calib_pairs=len(lab),
                        calib_precision=prec, calib_recall=rec, expert_checked_precision=exp_prec,
                        topics_at_cap=int(df["at_cap"].sum())))

    covered = run[run["score"] >= thr].merge(corpus, on="doc_id", how="left")
    covered["period"] = pd.cut(covered["year"], [2004, 2009, 2014, 2019, 2025],
                               labels=["2005-09", "2010-14", "2015-19", "2020-25"])
    period_rows.append(covered.groupby(["topic_id", "period"], observed=False).size().rename("n").reset_index().assign(retriever=retr))
    group_rows.append(covered.groupby(["topic_id", "author_group"]).size().rename("n").reset_index().assign(retriever=retr))

pd.DataFrame(summary).to_csv(RESULTS / "coverage_summary.csv", index=False)
if period_rows:
    pd.concat(period_rows).to_csv(RESULTS / "coverage_by_period.csv", index=False)
    pd.concat(group_rows).to_csv(RESULTS / "coverage_by_group.csv", index=False)

stab = []
for r1, r2 in itertools.combinations(per_retr_cov, 2):
    j = pd.concat([per_retr_cov[r1], per_retr_cov[r2]], axis=1, join="inner")
    tau, p = kendalltau(j.iloc[:, 0], j.iloc[:, 1])
    stab.append(dict(retriever_a=r1, retriever_b=r2, kendall_tau=tau, p=p, topics=len(j)))
pd.DataFrame(stab).to_csv(RESULTS / "coverage_stability.csv", index=False)

print(pd.DataFrame(summary).round(3).to_string(index=False))
print("\nCoverage-ranking stability across retrievers:")
print(pd.DataFrame(stab).round(3).to_string(index=False))
