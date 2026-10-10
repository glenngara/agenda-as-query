"""Blind relevance-labeling app.  Run:  make annotate   (or: streamlit run src/annotate_app.py)

- Shows the stratified expert sample (data/pool/expert_sample.csv) if it exists, else the full pool.
- Pairs are shuffled; which system retrieved them is hidden; authors/affiliations are hidden.
- Every click is saved immediately to data/qrels/labels_<name>.csv (safe to close and resume).
"""
import json, time
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data/pool/pool.csv"
SAMPLE = ROOT / "data/pool/expert_sample.csv"
TOPICS = ROOT / "data/topics/topics.jsonl"
QRELS = ROOT / "data/qrels"
PRIMARY = "A1"

GRADES = {
    3: ("3 · Highly relevant", "Directly studies the target's subject; findings could inform how the target is met or monitored."),
    2: ("2 · Relevant", "Clearly addresses the target's subject, but indirectly or only in part."),
    1: ("1 · Marginal", "Mentions related concepts; would not meaningfully inform the target."),
    0: ("0 · Not relevant", "Unrelated to the target."),
}

st.set_page_config(page_title="Grading: Agenda-as-Query", layout="wide")
pool = pd.read_csv(SAMPLE if SAMPLE.exists() else POOL)
if "overlap" not in pool.columns:
    pool["overlap"] = 1
topics = {t["topic_id"]: t for t in map(json.loads, open(TOPICS))}

default_name = st.query_params.get("name", PRIMARY)
name = st.sidebar.text_input("Annotator name", value=default_name).strip().lower() or PRIMARY
items = pool if name == PRIMARY else pool[pool["overlap"] == 1]
items = items.sort_values(["topic_id", "pair_id"]).reset_index(drop=True)  # group by topic: less context switching
out = QRELS / f"labels_{name}.csv"
labels = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["pair_id", "topic_id", "doc_id", "grade", "note", "ts"])
done = set(labels["pair_id"])

remaining = items[~items["pair_id"].isin(done)]
st.sidebar.progress(len(done & set(items["pair_id"])) / max(1, len(items)),
                    text=f"{len(done & set(items['pair_id']))} / {len(items)} labeled")
st.sidebar.markdown("**Grades**\n\n" + "\n\n".join(f"**{v[0]}**: {v[1]}" for v in GRADES.values()))
show_done = st.sidebar.checkbox("Review / change past labels")

if show_done:
    pid = st.sidebar.selectbox("Pair", sorted(done))
    row = items[items["pair_id"] == pid]
    if row.empty:
        st.stop()
    row = row.iloc[0]
elif remaining.empty:
    st.success("All pairs labeled. Thank you!")
    st.stop()
else:
    row = remaining.iloc[0]

t = topics.get(row["topic_id"], {})
c1, c2 = st.columns([2, 3])
with c1:
    st.caption(f"{row['topic_id']} · {t.get('source','')}")
    st.subheader(t.get("title", ""))
    st.write(t.get("description", ""))
    if t.get("narrative"):
        st.info(f"**Relevance guidance:** {t['narrative']}")
    with st.expander("Policy actions (context)"):
        st.write(t.get("actions", ""))
with c2:
    st.caption(f"{row['pair_id']} · {int(row['year']) if pd.notna(row['year']) else ''}")
    st.markdown(f"### {row['title']}")
    st.write(row["abstract"])

note = st.text_input("Optional note", key=f"note_{row['pair_id']}")
cols = st.columns(4)
for col, g in zip(cols, [3, 2, 1, 0]):
    if col.button(GRADES[g][0], key=f"g{g}_{row['pair_id']}", use_container_width=True):
        labels = labels[labels["pair_id"] != row["pair_id"]]
        labels = pd.concat([labels, pd.DataFrame([{
            "pair_id": row["pair_id"], "topic_id": row["topic_id"], "doc_id": row["doc_id"],
            "grade": g, "note": note, "ts": time.time()}])])
        labels.to_csv(out, index=False)
        st.rerun()
