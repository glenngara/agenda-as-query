"""Step 4: run every retriever x query formulation; save TREC run files (top TOP_K).

Retrievers: bm25 (rank_bm25), bge (BAAI/bge-base-en-v1.5), specter2 (allenai/specter2_base).
Multi-text queries (decomp) are run per sub-query and fused with Reciprocal Rank Fusion.
Output: data/runs/<retriever>__<qtype>.tsv  (topic_id, doc_id, rank, score)
"""
import json, re, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from config import CORPUS, QUERIES, RUNS, INDEX, DENSE_MODELS, BGE_QUERY_PREFIX, TOP_K, RRF_K

STOP = set("""a an the of and or in on for to with by from at as is are was were be been this that these those
its it their our we which using use based study studies results paper""".split())


def tok(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1]


def device():
    import torch
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


class Dense:
    def __init__(self, name, model_id, docs, doc_ids):
        from sentence_transformers import SentenceTransformer
        self.name, self.prefix = name, (BGE_QUERY_PREFIX if name == "bge" else "")
        self.model = SentenceTransformer(model_id, device=device())
        self.model.max_seq_length = 512
        cache = INDEX / f"{name}_emb.npy"
        ids_cache = INDEX / f"{name}_ids.json"
        if cache.exists() and json.load(open(ids_cache)) == doc_ids:
            self.E = np.load(cache)
        else:
            print(f"Encoding {len(docs)} docs with {model_id} (one-time, cached)...")
            self.E = self.model.encode(docs, batch_size=32, show_progress_bar=True,
                                       normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            np.save(cache, self.E)
            json.dump(doc_ids, open(ids_cache, "w"))

    def search(self, q, k):
        v = self.model.encode([self.prefix + q], normalize_embeddings=True)[0]
        s = self.E @ v
        idx = np.argpartition(-s, min(k, len(s) - 1))[:k]
        idx = idx[np.argsort(-s[idx])]
        return idx, s[idx]


class BM25:
    name = "bm25"

    def __init__(self, docs):
        self.bm = BM25Okapi([tok(d) for d in tqdm(docs, desc="bm25 index")])

    def search(self, q, k):
        s = self.bm.get_scores(tok(q))
        idx = np.argsort(-s)[:k]
        return idx, s[idx]


def rrf(results, k):
    agg = {}
    for idx, _ in results:
        for r, i in enumerate(idx):
            agg[i] = agg.get(i, 0) + 1.0 / (RRF_K + r + 1)
    best = sorted(agg.items(), key=lambda x: -x[1])[:k]
    return np.array([b[0] for b in best]), np.array([b[1] for b in best])


def main():
    corpus = pd.read_parquet(CORPUS / "corpus.parquet")
    docs, doc_ids = corpus["text"].tolist(), corpus["doc_id"].tolist()
    queries = [json.loads(l) for l in open(QUERIES / "queries.jsonl")]
    from config import TOPICS
    n_topics = sum(1 for _ in open(TOPICS / "topics.jsonl"))
    by_type = {}
    for q in queries:
        by_type.setdefault(q["qtype"], set()).add(q["topic_id"])
    print("Query sets per type:", {k: len(v) for k, v in by_type.items()}, f"(topics: {n_topics})")
    incomplete = [k for k, v in by_type.items() if len(v) < n_topics and k != "expert"]
    if incomplete or not {"decomp", "hyde"} <= set(by_type):
        print("[!] Queries are incomplete. Run `make queries` successfully first (Ollama must be running).")
        if "--force" not in sys.argv:
            sys.exit(1)
    only = [a for a in sys.argv[1:] if not a.startswith("--")]  # optional: restrict to retriever names
    retrievers = []
    if not only or "bm25" in only:
        retrievers.append(BM25(docs))
    for name, mid in DENSE_MODELS.items():
        if not only or name in only:
            retrievers.append(Dense(name, mid, docs, doc_ids))
    for R in retrievers:
        by_qtype = {}
        for q in tqdm(queries, desc=R.name):
            res = [R.search(t, TOP_K) for t in q["texts"]]
            idx, sc = res[0] if len(res) == 1 else rrf(res, TOP_K)
            for rank, (i, s) in enumerate(zip(idx, sc), 1):
                by_qtype.setdefault(q["qtype"], []).append((q["topic_id"], doc_ids[i], rank, float(s)))
        for qtype, rows in by_qtype.items():
            pd.DataFrame(rows, columns=["topic_id", "doc_id", "rank", "score"]).to_csv(
                RUNS / f"{R.name}__{qtype}.tsv", sep="\t", index=False)
        print(f"{R.name}: wrote {len(by_qtype)} runs")


if __name__ == "__main__":
    main()
