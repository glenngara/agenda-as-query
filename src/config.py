"""Shared paths and settings for the Agenda-as-Query pipeline."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
RAW = DATA / "raw"
CORPUS = DATA / "corpus"
TOPICS = DATA / "topics"
QUERIES = DATA / "queries"
RUNS = DATA / "runs"
INDEX = DATA / "index"
POOL = DATA / "pool"
QRELS = DATA / "qrels"
LLM_LOGS = DATA / "llm_logs"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
POLICY = ROOT / "policy-docs"

for p in [RAW, CORPUS, TOPICS, QUERIES, RUNS, INDEX, POOL, QRELS, LLM_LOGS, RESULTS, FIGS]:
    p.mkdir(parents=True, exist_ok=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")
S2_API_KEY = os.getenv("S2_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
HAIKU_MODEL = os.getenv("HAIKU_MODEL", "claude-haiku-5-5")
PRIMARY_EXPERT = "A1"   # expert labels = primary qrels (full pool)
EXPERT_SAMPLE = 100           # only used by the optional `make sample` variant
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")   # optional, not used by default
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ---- Corpus definition (OpenAlex) ----
YEARS = "2005-2025"
# OpenAlex subfields (ASJC-based) covering ecology / environment / biodiversity
SUBFIELDS = {
    1105: "Ecology, Evolution, Behavior and Systematics",
    1104: "Aquatic Science",
    1103: "Animal Science and Zoology",
    1107: "Forestry",
    1109: "Insect Science",
    1110: "Plant Science",
    2303: "Ecology",
    2306: "Global and Planetary Change",
    2309: "Nature and Landscape Conservation",
    2310: "Pollution",
    2312: "Water Science and Technology",
}

# ---- Supplementary sources (01b) ----
S2_FIELDS = ["Environmental Science", "Biology", "Agricultural and Food Sciences"]
# Philippine journals checked via Crossref (ISSN). Verify/extend this list; counts per journal are logged.
PH_JOURNALS = {
    "0031-7683": "Philippine Journal of Science",
    "0031-7454": "Philippine Agricultural Scientist",
    "1908-6865": "Philippine Journal of Systematic Biology",
    "0119-1144": "Journal of Environmental Science and Management",
}
ECO_TERMS = """biodiversity species ecosystem ecolog forest marine coral reef mangrove seagrass fish fisher
conservation wildlife habitat invasive pollut water quality wetland river lake bird bat amphibian reptile
mammal insect plant flora fauna endemic protected area climate watershed soil restoration aquatic
plankton algae benthic""".split()

# ---- Retrievers ----
DENSE_MODELS = {
    "bge": "BAAI/bge-base-en-v1.5",        # general-purpose embedder
    "specter2": "allenai/specter2_base",   # scientific embedder (citation-trained)
}
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K = 1000          # depth saved per run (used for coverage estimates)
POOL_DEPTH = 10       # depth pooled for human judging
RRF_K = 60

def require_key():
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.startswith("PASTE_"):
        raise SystemExit("ANTHROPIC_API_KEY is not set. Open .env and paste your key.")


def require_openai():
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("PASTE_"):
        raise SystemExit("OPENAI_API_KEY is not set. Open .env and paste your key.")
