"""Step 9: paper figures (static PNG + PDF, 300 dpi) from results/*.csv."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from config import RESULTS, FIGS, TOPICS

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # fixed categorical order
SEQ = LinearSegmentedColormap.from_list("blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
QORDER = ["title", "desc", "verbose", "expert", "decomp", "hyde"]
QLABEL = {"title": "Title", "desc": "Title+desc.", "verbose": "Verbose", "expert": "Expert",
          "decomp": "LLM decomp.+RRF", "hyde": "HyDE"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
topics = {t["topic_id"]: t["title"] for t in map(json.loads, open(TOPICS / "topics.jsonl"))}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_retrieval():
    df = pd.read_csv(RESULTS / "retrieval_primary.csv")
    retrs = sorted(df["retriever"].unique())
    qts = [q for q in QORDER if q in set(df["qtype"])]
    fig, ax = plt.subplots(figsize=(7, 3))
    w = 0.8 / len(retrs)
    x = np.arange(len(qts))
    for i, r in enumerate(retrs):
        d = df[df["retriever"] == r].set_index("qtype").reindex(qts)
        ax.bar(x + i * w, d["ndcg10"], w * 0.92, color=SERIES[i], label=r,
               yerr=[d["ndcg10"] - d["ndcg10_lo"], d["ndcg10_hi"] - d["ndcg10"]],
               error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
    ax.set_xticks(x + w * (len(retrs) - 1) / 2, [QLABEL[q] for q in qts])
    ax.set_ylabel("nDCG@10 (expert qrels)")
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=len(retrs), loc="upper left")
    save(fig, "fig_retrieval_ndcg")


def fig_coverage():
    s = pd.read_csv(RESULTS / "coverage_summary.csv")
    eff = pd.read_csv(RESULTS / "retrieval_expert.csv")
    best = eff.sort_values("ndcg10", ascending=False).iloc[0]["retriever"]
    df = pd.read_csv(RESULTS / f"coverage_{best}.csv").sort_values("coverage")
    fig, ax = plt.subplots(figsize=(6.5, 0.22 * len(df) + 0.8))
    y = np.arange(len(df))
    ax.barh(y, df["coverage"], color=SERIES[0], height=0.7)
    ax.errorbar(df["coverage"], y, xerr=[df["coverage"] - df["lo"], df["hi"] - df["coverage"]],
                fmt="none", ecolor=MUTED, lw=0.8, capsize=2)
    ax.set_yticks(y, [f"{t} {topics.get(t,'')[:38]}" for t in df["topic_id"]], fontsize=7)
    ax.set_xlabel(f"Estimated relevant papers ({best}, 95% bootstrap interval)")
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    save(fig, "fig_coverage_by_target")


def fig_period():
    df = pd.read_csv(RESULTS / "coverage_by_period.csv")
    eff = pd.read_csv(RESULTS / "retrieval_expert.csv")
    best = eff.sort_values("ndcg10", ascending=False).iloc[0]["retriever"]
    m = df[df["retriever"] == best].pivot(index="topic_id", columns="period", values="n").fillna(0)
    m = m.div(m.sum(1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(4.5, 0.22 * len(m) + 0.8))
    im = ax.imshow(m.values, cmap=SEQ, aspect="auto", vmin=0, vmax=max(0.6, m.values.max()))
    ax.set_xticks(range(m.shape[1]), m.columns); ax.set_yticks(range(len(m)), m.index, fontsize=7)
    for (i, j), v in np.ndenumerate(m.values):
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=6, color="white" if v > 0.4 else INK)
    fig.colorbar(im, ax=ax, label="Share of covered papers", fraction=0.04)
    save(fig, "fig_coverage_by_period")


def fig_groups():
    df = pd.read_csv(RESULTS / "coverage_by_group.csv")
    eff = pd.read_csv(RESULTS / "retrieval_expert.csv")
    best = eff.sort_values("ndcg10", ascending=False).iloc[0]["retriever"]
    order = ["local_only", "local_led_collab", "foreign_led_collab", "foreign_only"]
    labels = ["PH only", "PH-led collab.", "Foreign-led collab.", "Foreign only"]
    hatches = ["", "//", "\\\\", ".."]
    m = df[df["retriever"] == best].pivot(index="topic_id", columns="author_group", values="n").reindex(columns=order).fillna(0)
    m = m.div(m.sum(1).replace(0, 1), axis=0).sort_values("foreign_only")
    fig, ax = plt.subplots(figsize=(6.5, 0.22 * len(m) + 1))
    left = np.zeros(len(m))
    for k, (c, lab) in enumerate(zip(order, labels)):
        ax.barh(range(len(m)), m[c], left=left, color=SERIES[k], hatch=hatches[k],
                edgecolor="white", linewidth=1, label=lab, height=0.75)
        left += m[c].values
    ax.set_yticks(range(len(m)), m.index, fontsize=7); ax.set_xlim(0, 1)
    ax.set_xlabel("Share of covered papers by authorship"); ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_coverage_by_authorship")


def fig_agreement():
    import glob
    files = sorted((RESULTS).glob("confusion_expert_vs_*.csv"))
    if not files:
        return
    fig, axes = plt.subplots(1, len(files), figsize=(3 * len(files), 2.8))
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, files):
        c = pd.read_csv(f, index_col=0).reindex(index=range(4), columns=[str(i) for i in range(4)]).fillna(0)
        ax.imshow(c.values, cmap=SEQ)
        for (i, j), v in np.ndenumerate(c.values):
            ax.text(j, i, int(v), ha="center", va="center", fontsize=8, color="white" if v > c.values.max() * 0.5 else INK)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xlabel(f.stem.replace("confusion_expert_vs_", "")); ax.set_ylabel("Expert grade")
    save(fig, "fig_agreement_confusion")


if __name__ == "__main__":
    for fn in [fig_retrieval, fig_agreement, fig_coverage, fig_period, fig_groups]:
        try:
            fn()
        except FileNotFoundError as e:
            print(f"skip {fn.__name__}: {e.filename} not found yet")
