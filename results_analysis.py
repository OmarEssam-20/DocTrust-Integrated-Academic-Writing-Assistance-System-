"""
results_analysis.py
Generates visualisations and a saved report from pipeline_demo results.
Run: python results_analysis.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessor   import preprocess
from summarizer     import summarize
from paraphraser    import paraphrase
from plagiarism     import check_plagiarism
from evaluator      import rouge_scores, semantic_similarity
from dataset_loader import load_cnn_sample, get_paraphrase_samples

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")


# ── 1. Summarization analysis ─────────────────────────────────────────────────
def analyse_summarization(n: int = 10):
    print("[Analysis] Running summarization on CNN/DailyMail sample ...")
    df = load_cnn_sample(n)

    records = []
    for _, row in df.iterrows():
        article   = preprocess(row["article"])
        reference = preprocess(row["highlights"])
        summary   = summarize(article)
        r         = rouge_scores(summary, reference)
        sim       = semantic_similarity(summary, reference)
        records.append({
            "rouge1": r["rouge1"],
            "rouge2": r["rouge2"],
            "rougeL": r["rougeL"],
            "semantic_sim": sim,
            "article_words":  len(article.split()),
            "summary_words":  len(summary.split()),
            "reference_words": len(reference.split()),
        })

    results_df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Summarization Evaluation", fontsize=13, fontweight="bold")

    metrics = ["rouge1", "rouge2", "rougeL", "semantic_sim"]
    means   = [results_df[m].mean() for m in metrics]
    axes[0].bar(metrics, means, color=["#4f8ef7", "#7c5cfc", "#4fcf8e", "#f7b955"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Average Scores")
    axes[0].set_ylabel("Score")
    for i, v in enumerate(means):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    axes[1].scatter(
        results_df["article_words"],
        results_df["rouge1"],
        alpha=0.7, color="#4f8ef7"
    )
    axes[1].set_title("Article Length vs ROUGE-1")
    axes[1].set_xlabel("Article Word Count")
    axes[1].set_ylabel("ROUGE-1")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "summarization_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    results_df.to_csv(os.path.join(OUTPUT_DIR, "summarization_results.csv"), index=False)
    return results_df.mean().to_dict()


# ── 2. Paraphrasing analysis ──────────────────────────────────────────────────
def analyse_paraphrasing():
    print("[Analysis] Running paraphrasing evaluation ...")
    samples = get_paraphrase_samples()
    records = []
    for s in samples:
        original = preprocess(s["original"])
        para, _  = paraphrase(original)
        r        = rouge_scores(para, original)
        sim      = semantic_similarity(para, original)
        records.append({
            "semantic_sim": sim,
            "rouge1": r["rouge1"],
            "rouge2": r["rouge2"],
            "rougeL": r["rougeL"],
            "original_words":   len(original.split()),
            "paraphrase_words": len(para.split()),
        })

    results_df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Paraphrasing Evaluation", fontsize=13, fontweight="bold")

    labels = [f"S{i+1}" for i in range(len(records))]
    x = np.arange(len(labels))
    axes[0].bar(x - 0.2, results_df["semantic_sim"], 0.4, label="Semantic Sim", color="#4f8ef7")
    axes[0].bar(x + 0.2, results_df["rouge1"],       0.4, label="ROUGE-1",      color="#4fcf8e")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Per-Sample Scores")
    axes[0].legend()

    axes[1].scatter(
        results_df["original_words"],
        results_df["paraphrase_words"],
        color="#7c5cfc", alpha=0.8
    )
    max_w = max(results_df[["original_words", "paraphrase_words"]].max())
    axes[1].plot([0, max_w], [0, max_w], "k--", alpha=0.3, label="Equal length")
    axes[1].set_title("Word Count: Original vs Paraphrase")
    axes[1].set_xlabel("Original Words")
    axes[1].set_ylabel("Paraphrase Words")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "paraphrasing_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    results_df.to_csv(os.path.join(OUTPUT_DIR, "paraphrasing_results.csv"), index=False)
    return results_df.mean().to_dict()


# ── 3. Plagiarism analysis ────────────────────────────────────────────────────
def analyse_plagiarism():
    print("[Analysis] Running plagiarism detection evaluation ...")
    test_cases = [
        ("High-risk",     "Plagiarism detection systems compare submitted texts with a reference corpus to identify similarities."),
        ("High-risk",     "Cosine similarity between sentence embeddings is an effective metric for detecting paraphrased plagiarism."),
        ("Moderate-risk", "Tools that check for academic work scan submissions to find content resembling existing sources."),
        ("Moderate-risk", "Systems designed to identify copied academic writing use vector comparisons to flag similar passages."),
        ("Low-risk",      "The migratory patterns of Arctic tern populations are influenced by geomagnetic field variations."),
        ("Low-risk",      "Fermentation kinetics in anaerobic bioreactors are modelled using Monod-type substrate uptake equations."),
    ]

    records = []
    for expected_label, text in test_cases:
        result = check_plagiarism(preprocess(text), top_k=1)
        records.append({
            "expected":      expected_label,
            "predicted":     result["overall_risk"],
            "max_similarity": result["max_similarity"],
        })

    results_df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Plagiarism Detection Evaluation", fontsize=13, fontweight="bold")

    risk_colors = {"🔴 High Risk": "#f76f6f", "🟠 Moderate Risk": "#f7b955", "🟢 Low Risk": "#4fcf8e"}
    colors = [risk_colors.get(r, "#aaa") for r in results_df["predicted"]]
    axes[0].bar(range(len(results_df)), results_df["max_similarity"], color=colors)
    axes[0].axhline(0.80, color="#f76f6f", linestyle="--", alpha=0.7, label="High threshold (0.80)")
    axes[0].axhline(0.60, color="#f7b955", linestyle="--", alpha=0.7, label="Moderate threshold (0.60)")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Similarity Scores per Test Case")
    axes[0].set_ylabel("Normalised Similarity")
    axes[0].set_xticks(range(len(results_df)))
    axes[0].set_xticklabels([f"T{i+1}" for i in range(len(results_df))], fontsize=9)
    axes[0].legend(fontsize=8)

    risk_counts = results_df["predicted"].value_counts()
    axes[1].pie(
        risk_counts.values,
        labels=risk_counts.index,
        colors=[risk_colors.get(r, "#aaa") for r in risk_counts.index],
        autopct="%1.0f%%",
        startangle=90,
    )
    axes[1].set_title("Risk Level Distribution")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plagiarism_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    results_df.to_csv(os.path.join(OUTPUT_DIR, "plagiarism_results.csv"), index=False)
    return results_df["max_similarity"].mean()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("📊  DocTrust — Results & Analysis\n")

    sum_stats  = analyse_summarization(n=10)
    para_stats = analyse_paraphrasing()
    plag_avg   = analyse_plagiarism()

    report = {
        "summarization":  sum_stats,
        "paraphrasing":   para_stats,
        "plagiarism_avg_similarity": round(plag_avg, 4),
    }

    report_path = os.path.join(OUTPUT_DIR, "full_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅  Full report saved to {report_path}")
    print(json.dumps(report, indent=2))