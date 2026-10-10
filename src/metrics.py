"""IR metrics, agreement statistics and bootstrap helpers (no compiled deps)."""
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def qrels_dict(labels: pd.DataFrame):
    d = {}
    for r in labels.dropna(subset=["grade"]).itertuples():
        d.setdefault(r.topic_id, {})[r.doc_id] = int(r.grade)
    return d


def ndcg_at_k(ranked, rels, k=10):
    gains = [rels.get(d, 0) for d in ranked[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else np.nan


def p_at_k(ranked, rels, k=10, thr=2):
    return np.mean([rels.get(d, 0) >= thr for d in ranked[:k]])


def judged_at_k(ranked, rels, k=10):
    return np.mean([d in rels for d in ranked[:k]])


def per_topic(run: pd.DataFrame, qrels, k=10):
    rows = []
    for tid, g in run.sort_values(["topic_id", "rank"]).groupby("topic_id"):
        if tid not in qrels:
            continue
        ranked = g["doc_id"].tolist()
        rels = qrels[tid]
        rows.append(dict(topic_id=tid, ndcg10=ndcg_at_k(ranked, rels, k),
                         p10=p_at_k(ranked, rels, k), judged10=judged_at_k(ranked, rels, k)))
    return pd.DataFrame(rows)


def boot_ci(x, n=2000, seed=0):
    x = np.asarray([v for v in x if not np.isnan(v)])
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def paired_perm_p(a, b, n=10000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1, 1], (n, len(d)))
    return float((np.abs((signs * d).mean(1)) >= obs).mean())


def agreement(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    return dict(
        n=len(a),
        exact=float((a == b).mean()),
        kappa_quadratic=float(cohen_kappa_score(a, b, weights="quadratic", labels=[0, 1, 2, 3])),
        kappa_binary=float(cohen_kappa_score(a >= 2, b >= 2)) if len(set(a >= 2) | set(b >= 2)) > 1 else np.nan,
        mean_diff=float((b - a).mean()),
    )
