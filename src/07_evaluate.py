"""Step 7: evaluation. Primary qrels = expert (the lead expert (A1), full pool); falls back to AI panel if absent.

(a) Retrieval effectiveness of every run under PRIMARY qrels (nDCG@10, P@10, judged@10, 95% CI,
    paired permutation test vs. baseline bm25__desc). Also under each single judge.
(b) Validation: agreement of each LLM judge and of the panel with the EXPERT on the sample
    (exact, quadratic & binary kappa, with bootstrap 95% CI); inter-judge agreement.
(c) Robustness: Kendall tau between system rankings under each single judge vs the panel.
Outputs in results/.
"""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from config import RUNS, QRELS, RESULTS, PRIMARY_EXPERT
from metrics import qrels_dict, per_topic, boot_ci, paired_perm_p, agreement

BASELINE = "bm25__desc"


def load_labels():
    labs = {}
    for f in sorted(QRELS.glob("labels_*.csv")):
        d = pd.read_csv(f).dropna(subset=["grade"])
        d["grade"] = d["grade"].astype(int)
        labs[f.stem.replace("labels_", "")] = d
    if PRIMARY_EXPERT not in labs and "panel" not in labs:
        raise SystemExit("No labels yet. Run `make annotate` (expert) and `make judge-all`.")
    return labs


def run_table(qrels, tag):
    runs = {f.stem: pd.read_csv(f, sep="\t") for f in sorted(RUNS.glob("*.tsv"))}
    pts = {name: per_topic(r, qrels) for name, r in runs.items()}
    base = pts.get(BASELINE)
    rows = []
    for name, pt in pts.items():
        lo, hi = boot_ci(pt["ndcg10"])
        p = None
        if base is not None and name != BASELINE:
            p = paired_perm_p(pt.set_index("topic_id")["ndcg10"].reindex(base["topic_id"]).values, base["ndcg10"].values)
        retr, qtype = name.split("__")
        rows.append(dict(run=name, retriever=retr, qtype=qtype, topics=len(pt),
                         ndcg10=pt["ndcg10"].mean(), ndcg10_lo=lo, ndcg10_hi=hi,
                         p10=pt["p10"].mean(), judged10=pt["judged10"].mean(), p_vs_baseline=p))
        pt.assign(run=name).to_csv(RESULTS / "per_topic" / f"{tag}__{name}.csv", index=False)
    return pd.DataFrame(rows).sort_values("ndcg10", ascending=False)


def kappa_ci(x, y, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    ks = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        ks.append(agreement(x[i], y[i])["kappa_quadratic"])
    return np.nanpercentile(ks, 2.5), np.nanpercentile(ks, 97.5)


def main():
    (RESULTS / "per_topic").mkdir(exist_ok=True)
    labs = load_labels()

    # (a) effectiveness under panel qrels (main table)
    primary = PRIMARY_EXPERT if PRIMARY_EXPERT in labs else "panel"
    main_tbl = run_table(qrels_dict(labs[primary]), primary)
    main_tbl.to_csv(RESULTS / "retrieval_primary.csv", index=False)
    main_tbl.to_csv(RESULTS / "retrieval_expert.csv", index=False)  # read by 08/09
    print(f"\n=== Retrieval effectiveness ({primary} qrels) ===")
    print(main_tbl[["run", "ndcg10", "ndcg10_lo", "ndcg10_hi", "p10", "judged10", "p_vs_baseline"]].round(3).to_string(index=False))

    # (b) validation against the expert + inter-judge agreement
    rows = []
    for a_, b_ in itertools.combinations(labs.keys(), 2):
        m = labs[a_][["pair_id", "grade"]].merge(labs[b_][["pair_id", "grade"]], on="pair_id", suffixes=("_a", "_b"))
        if len(m) < 10:
            continue
        r = dict(source_a=a_, source_b=b_, **agreement(m["grade_a"], m["grade_b"]))
        r["kappa_q_lo"], r["kappa_q_hi"] = kappa_ci(m["grade_a"], m["grade_b"])
        rows.append(r)
    ag = pd.DataFrame(rows)
    ag.to_csv(RESULTS / "agreement.csv", index=False)
    print("\n=== Agreement (expert vs 2nd annotator vs AI judges) ===")
    print(ag.round(3).to_string(index=False))

    if PRIMARY_EXPERT in labs:
        for k in labs:
            if k.startswith("llm_") or k == "panel":
                m = labs[PRIMARY_EXPERT][["pair_id", "grade"]].merge(labs[k][["pair_id", "grade"]], on="pair_id", suffixes=("_expert", "_llm"))
                pd.crosstab(m["grade_expert"], m["grade_llm"]).to_csv(RESULTS / f"confusion_expert_vs_{k}.csv")
        # binary precision/recall of panel relevance (grade>=2) against the expert
        if "panel" not in labs:
            labs["panel"] = labs[next(k for k in labs if k.startswith("llm_"))] if any(k.startswith("llm_") for k in labs) else labs[PRIMARY_EXPERT]
        m = labs[PRIMARY_EXPERT][["pair_id", "grade"]].merge(labs["panel"][["pair_id", "grade"]], on="pair_id", suffixes=("_e", "_p"))
        tp = ((m.grade_p >= 2) & (m.grade_e >= 2)).sum()
        prec = tp / max(1, (m.grade_p >= 2).sum()); rec = tp / max(1, (m.grade_e >= 2).sum())
        pd.DataFrame([dict(n=len(m), panel_precision=prec, panel_recall=rec)]).to_csv(RESULTS / "panel_vs_expert_binary.csv", index=False)
        print(f"\nPanel vs expert (relevant = grade>=2): precision {prec:.2f}, recall {rec:.2f} on {len(m)} pairs")
    else:
        print("\n(No expert labels yet: validation rows will appear after the lead expert (A1) labels the sample.)")

    # (c) system rankings under each other label source vs the primary qrels
    rows = []
    for k in labs:
        if k == primary:
            continue
        other = run_table(qrels_dict(labs[k]), k)
        j = main_tbl[["run", "ndcg10"]].merge(other[["run", "ndcg10"]], on="run", suffixes=("_primary", "_judge"))
        tau, p = kendalltau(j["ndcg10_primary"], j["ndcg10_judge"])
        rows.append(dict(judge=k, runs=len(j), kendall_tau=tau, p=p,
                         same_top_system=j.sort_values("ndcg10_primary").iloc[-1]["run"] == j.sort_values("ndcg10_judge").iloc[-1]["run"]))
    tau_df = pd.DataFrame(rows)
    tau_df.to_csv(RESULTS / "system_ranking_tau.csv", index=False)
    print(f"\n=== System-ranking correlation vs {primary} qrels ===")
    print(tau_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
