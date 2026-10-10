# Agenda-as-Query pipeline. Run from the project folder:  make <target>
PY = .venv/bin/python
ST = .venv/bin/streamlit

.PHONY: rehydrate reproduce-tables evaluate-extended judge-nocontext stability recall-check supplement judge-haiku judge-all panel sample setup harvest topics topics-final queries retrieve pool annotate annotate-2 judge judge-gpt evaluate coverage figures all check

setup:            ## one-time: venv + packages
	bash setup.sh

check:            ## sanity-check keys and models
	cd src && ../$(PY) -c "import config as c; print('Claude key set:', bool(c.ANTHROPIC_API_KEY) and not c.ANTHROPIC_API_KEY.startswith('PASTE_'), '| model', c.CLAUDE_MODEL, '/', c.HAIKU_MODEL); print('OpenAI key set:', bool(c.OPENAI_API_KEY) and not c.OPENAI_API_KEY.startswith('PASTE_'), '| model', c.OPENAI_MODEL)"
	cd src && ../$(PY) -c "import config as c, anthropic; cl=anthropic.Anthropic(api_key=c.ANTHROPIC_API_KEY); [print('Claude OK:', m, '->', cl.messages.create(model=m, max_tokens=5, messages=[{'role':'user','content':'Say OK'}]).content[0].text) for m in (c.CLAUDE_MODEL, c.HAIKU_MODEL)]"
	cd src && ../$(PY) -c "import config as c; from openai import OpenAI; o=OpenAI(api_key=c.OPENAI_API_KEY); ids=sorted(m.id for m in o.models.list()); print('OpenAI OK:', c.OPENAI_MODEL in ids, '| GPT models available:', [i for i in ids if i.startswith(('gpt-4','gpt-5','o'))][:15])"

harvest:          ## 1. OpenAlex corpus -> data/corpus/corpus.parquet
	cd src && ../$(PY) 01_harvest_openalex.py

supplement:       ## 1b. add Semantic Scholar + Crossref (PH journals), audit OpenAlex gaps
	cd src && ../$(PY) 01b_supplement_sources.py

recall-check:     ## 1c. corpus recall vs expert-review reference lists (data/recall/)
	cd src && ../$(PY) 01c_recall_check.py

topics:           ## 2a. draft topics -> data/topics/topics_draft.xlsx (the lead expert (A1) edits)
	cd src && ../$(PY) 02_build_topics.py

topics-final:     ## 2b. edited xlsx -> data/topics/topics.jsonl
	cd src && ../$(PY) 02_build_topics.py --final

queries:          ## 3. query formulations (GPT-4.1)
	cd src && ../$(PY) 03_query_formulations.py

retrieve:         ## 4. BM25 + bge + SPECTER2 runs
	cd src && ../$(PY) 04_retrieve.py

pool:             ## 5. judging pool
	cd src && ../$(PY) 05_build_pool.py

annotate:         ## 5b. labeling app (the lead expert (A1): full pool; colleague: overlap subset)
	$(ST) run src/annotate_app.py

judge:            ## 6a. Claude judge
	cd src && ../$(PY) 06_llm_judge.py claude

judge-gpt:        ## 6b. GPT judge (OpenAI)
	cd src && ../$(PY) 06_llm_judge.py gpt

judge-haiku:      ## 6c. Claude Haiku judge
	cd src && ../$(PY) 06_llm_judge.py haiku

judge-all: judge judge-haiku judge-gpt   ## all three AI judges (prompt matched to the grading guide)

judge-nocontext:  ## ablation: Claude Sonnet without Philippine context or country rule
	cd src && ../$(PY) 06_llm_judge.py claude --nocontext

stability:        ## 6c. re-grade the 60 shared pairs twice more per AI judge (run-to-run stability)
	cd src && ../$(PY) 06c_judge_stability.py claude haiku gpt --runs 2

panel:            ## 6d. AI-panel consensus labels (main design)
	cd src && ../$(PY) 06b_panel_and_sample.py --no-sample

sample:           ## (alt. design only) 100-pair stratified expert sample
	cd src && ../$(PY) 06b_panel_and_sample.py

evaluate:         ## 7. effectiveness, agreement, system-ranking tau
	cd src && ../$(PY) 07_evaluate.py

coverage:         ## 8. coverage estimates with intervals
	cd src && ../$(PY) 08_coverage.py

figures:          ## 9. paper figures -> results/figures
	cd src && ../$(PY) 09_figures.py

all: evaluate coverage figures

evaluate-extended: ## 7b. condensed-list retrieval eval, AC1/PABAK, country split, pool evidence per goal
	cd src && ../$(PY) 07b_extended_eval.py

rehydrate:        ## 0. reproducers: download titles/abstracts for the released IDs (corpus.parquet, pool.csv)
	cd src && ../$(PY) 00_rehydrate.py

reproduce-tables: ## re-compute every table and figure value in the paper from the released files (no API keys needed)
	cd src && ../$(PY) 07_evaluate.py && ../$(PY) 07b_extended_eval.py && ../$(PY) 06c_judge_stability.py claude haiku gpt --runs 2
