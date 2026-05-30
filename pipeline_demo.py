"""
pipeline_demo.py
End-to-end demonstration of the DocTrust pipeline.
Run: python pipeline_demo.py
"""

from preprocessor   import preprocess
from summarizer     import summarize
from paraphraser    import paraphrase
from plagiarism     import check_plagiarism
from evaluator      import evaluate_summarization, evaluate_paraphrasing
from dataset_loader import load_cnn_sample, get_paraphrase_samples

import json


DIVIDER = "=" * 60


def _header(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def demo_summarization(n_samples: int = 5):
    _header("MODULE 1 — TEXT SUMMARIZATION (BART)")
    df = load_cnn_sample(n_samples)

    summaries, references = [], []
    for _, row in df.iterrows():
        article   = preprocess(row["article"])
        reference = preprocess(row["highlights"])
        summary   = summarize(article)
        summaries.append(summary)
        references.append(reference)
        print(f"\n[Article snippet] {article[:120]}...")
        print(f"[Reference]       {reference[:120]}...")
        print(f"[Generated]       {summary[:120]}...")

    metrics = evaluate_summarization(summaries, references)
    print(f"\n── Summarization Metrics (n={n_samples}) ──")
    for k, v in metrics.items():
        print(f"  {k:<28}: {v}")
    return metrics


def demo_paraphrasing():
    _header("MODULE 2 — TEXT PARAPHRASING (T5)")
    samples    = get_paraphrase_samples()
    paraphrases, originals = [], []

    for s in samples:
        original  = preprocess(s["original"])
        para, sim = paraphrase(original)
        paraphrases.append(para)
        originals.append(original)
        print(f"\n[Original]    {original}")
        print(f"[Paraphrase]  {para}")
        print(f"[Cosine Sim]  {sim}")

    metrics = evaluate_paraphrasing(paraphrases, originals)
    print(f"\n── Paraphrasing Metrics (n={len(samples)}) ──")
    for k, v in metrics.items():
        print(f"  {k:<28}: {v}")
    return metrics


def demo_plagiarism():
    _header("MODULE 3 — PLAGIARISM DETECTION (SBERT + FAISS)")

    test_cases = [
        {
            "label": "High-risk (near-verbatim)",
            "text":  "Plagiarism detection systems compare submitted texts with a reference corpus to identify similarities.",
        },
        {
            "label": "Moderate-risk (paraphrased)",
            "text":  "Tools that check for academic dishonesty scan submitted work to find content that matches existing sources.",
        },
        {
            "label": "Low-risk (original)",
            "text":  "The role of ocean salinity in regulating thermohaline circulation is a key topic in physical oceanography.",
        },
    ]

    results = []
    for case in test_cases:
        text   = preprocess(case["text"])
        result = check_plagiarism(text, top_k=3)
        results.append({**case, **result})
        print(f"\n[{case['label']}]")
        print(f"  Text:          {text[:80]}...")
        print(f"  Overall Risk:  {result['overall_risk']}")
        print(f"  Max Sim:       {result['max_similarity']}")
        if result["matches"]:
            top = result["matches"][0]
            print(f"  Top Match:     {top['matched_text'][:80]}...")

    return results


def run_full_demo():
    print("\n📄  DocTrust — Full Pipeline Demo")
    print("    Summarization · Paraphrasing · Plagiarism Detection\n")

    sum_metrics  = demo_summarization(n_samples=5)
    para_metrics = demo_paraphrasing()
    plag_results = demo_plagiarism()

    _header("SUMMARY OF RESULTS")
    print(json.dumps({
        "summarization":  sum_metrics,
        "paraphrasing":   para_metrics,
        "plagiarism_cases": len(plag_results),
    }, indent=2))

    print(f"\n✅  Demo complete. See outputs/ for saved results.")


if __name__ == "__main__":
    run_full_demo()