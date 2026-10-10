# Agenda-as-Query

Code, runs and relevance judgments for the NLPIR 2026 paper:

> **Agenda-as-Query: Jurisdiction-Aware Relevance in a Test Collection for Mapping Research to National Biodiversity Targets.**
> Gretchie C. Gara, Charisse T. Evangelio, Rusty Keith E. Placino and Kristine Hannah V. Guevarra.
> 10th International Conference on Natural Language Processing and
> Information Retrieval (NLPIR 2026), Nara, Japan. Springer LNNS (to appear).

The collection uses the 23 national targets of the Philippine Biodiversity Strategy and Action Plan (PBSAP)
2024–2040 as search topics over 17,242 Philippine ecology and environment papers (2005–2025). It contains
15 retrieval runs, a 460-pair judging pool graded 0–3 by a domain expert (plus three second graders and an
intra-rater repeat), and grades from three LLM assessors (Claude Sonnet 5.5, Claude Haiku 5.5, GPT-4.1)
with stability re-runs and a no-context ablation.

## Reproduce the paper's tables (no API keys, about 1 minute)

```bash
bash setup.sh            # Python venv + packages (Python 3.9+)
make reproduce-tables    # recomputes every table from the released runs and grades
```

Outputs are written to `results/`. The files used in the paper are:

| Paper element | File |
|---|---|
| Agreement with the expert (Table 2) | `results/agreement_extended.csv` |
| LLM stability across runs (Table 3) | `results/judge_stability.csv` |
| Ranking and significance agreement (Table 3) | `results/system_ranking_tau_condensed.csv`, `results/significance_agreement.csv` |
| Retrieval effectiveness, condensed lists (Fig. 4) | `results/retrieval_expert_condensed.csv` |
| Philippine share of top-10 per run (Fig. 2b) | `results/ph_share_top10_by_run.csv` |
| Evidence per target (Fig. 5) | `results/pool_evidence_by_goal.csv` |
| Corpus completeness audit | `results/corpus_recall.csv`, `results/corpus_source_audit.csv` |

`retrieval_primary.csv`, `retrieval_expert.csv`, `system_ranking_tau.csv` and `coverage_{bge,bm25,specter2}.csv`
come from the original pipeline and treat unjudged documents as non-relevant or calibrate on grade >= 2. With a
shallow pool these scores mostly reflect how many of a run's documents were judged (Spearman rho = 0.74 with
judged@10), so the paper reports the condensed-list and grade-3 versions instead. `labels_panel.csv` (majority
of the three LLM assessors) is released for completeness and is not used in the paper.

## Re-run the full pipeline

```bash
cp .env.example .env     # add ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENALEX_EMAIL
make rehydrate           # downloads titles/abstracts for the released IDs (we do not redistribute them)
make retrieve            # BM25, BGE-base-en-v1.5, SPECTER2-base runs (top-1000)
make judge-all           # LLM assessors with the prompt matched to the human grading guide
make judge-nocontext     # ablation: no Philippine context, no country rule
make stability           # 60 shared pairs graded twice more per LLM
make panel evaluate evaluate-extended
```

Query formulations (`data/queries/queries.jsonl`) were generated once with GPT-4.1 and are released, so
`make queries` is only needed to study generation itself. Retrieval on re-downloaded text can differ slightly
if upstream abstracts changed after our harvest.

## Repository layout

```
src/                  pipeline (00_rehydrate ... 09_figures), LLM wrappers, metrics
data/topics/          23 PBSAP 2024-2040 targets (title, description, actions)
data/queries/         5 query formulations per target (115 query sets)
data/corpus/          corpus_ids.csv: IDs, DOIs and metadata of the 17,242 papers (no text);
                      ph_tag.csv: Philippine-study tag per paper (terms in docs/philippine_terms.txt)
data/runs/            15 TREC-style runs (topic_id, doc_id, rank, score), top-1000
data/pool/            pool_ids.csv: 460 judged pairs; overlap = 1 marks the 60 shared pairs
data/qrels/           grades: A1 (lead expert), A1_repeat, A2-A4 (second graders), LLM assessors, panel
data/qrels/original/  human grades before graders' own revisions (20 of 700 grades changed)
data/qrels/stability/ LLM re-runs on the 60 shared pairs
results/              all tables and figures
```

Each grade file has `pair_id, topic_id, doc_id, grade` (0–3). LLM files also keep the model's one-sentence
reason (`raw`). Human files have `revised` (grader changed the grade before submitting) and `non_ph`
(paper is not a Philippine study; keyword match on title and abstract, checked by hand on the pool).

## LLM settings

| Assessor | Model ID | Sampling |
|---|---|---|
| Claude Sonnet | `claude-sonnet-5-5` | provider default (the model rejects non-default `temperature`); adaptive thinking on by default |
| Claude Haiku | `claude-haiku-5-5` | provider default; adaptive thinking on by default |
| GPT | `gpt-4.1` | `temperature=0`, `seed=42` (best effort on the provider side) |

Prompts are in `src/06_llm_judge.py`: `PROMPT` is the main prompt matched to the human grading guide
(`docs/grading_guide.md`); `PROMPT_NOCONTEXT` is the ablation.

## Ethics and privacy

Graders took part voluntarily and are anonymized (A1–A4). No personal data about authors of the indexed papers
is released beyond public bibliographic metadata. Paper abstracts are not redistributed; `make rehydrate`
fetches them from OpenAlex, Semantic Scholar and Crossref under those services' terms.

## License

Code: MIT (see `LICENSE`). Topics, queries, runs and relevance judgments: CC BY 4.0 (see `DATA_LICENSE`).
The PBSAP target texts are reproduced from the public Philippine government document for research use.

## Citation

If you use this collection or code, please cite:

```bibtex
@inproceedings{gara2026agenda,
  title     = {Agenda-as-Query: Jurisdiction-Aware Relevance in a Test Collection for Mapping
               Research to National Biodiversity Targets},
  author    = {Gara, Gretchie C. and Evangelio, Charisse T. and Placino, Rusty Keith E. and
               Guevarra, Kristine Hannah V.},
  booktitle = {Proceedings of the 10th International Conference on Natural Language Processing
               and Information Retrieval (NLPIR 2026)},
  series    = {Lecture Notes in Networks and Systems},
  publisher = {Springer},
  year      = {2026},
  note      = {To appear}
}
```
