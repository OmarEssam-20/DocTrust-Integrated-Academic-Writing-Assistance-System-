"""
evaluator.py
ROUGE and cosine similarity evaluation for all three DocTrust modules.
"""

import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from model_loader import get_embedder


# ── ROUGE ─────────────────────────────────────────────────────────────────────
_ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]
_scorer = rouge_scorer.RougeScorer(_ROUGE_TYPES, use_stemmer=True)


def rouge_scores(hypothesis: str, reference: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 between hypothesis and reference.
    Returns dict: {'rouge1': float, 'rouge2': float, 'rougeL': float}
    """
    scores = _scorer.score(reference, hypothesis)
    return {
        k: round(scores[k].fmeasure, 4)
        for k in _ROUGE_TYPES
    }


def batch_rouge(hypotheses: list, references: list) -> dict:
    """
    Average ROUGE scores over a list of (hypothesis, reference) pairs.
    """
    assert len(hypotheses) == len(references), "Lists must be same length."
    all_scores = [rouge_scores(h, r) for h, r in zip(hypotheses, references)]
    averaged = {}
    for key in _ROUGE_TYPES:
        averaged[key] = round(float(np.mean([s[key] for s in all_scores])), 4)
    return averaged


# ── Cosine Similarity ─────────────────────────────────────────────────────────
def semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts using SBERT embeddings.
    Returns a float in [0, 1].
    """
    embedder = get_embedder()
    embs = embedder.encode([text_a, text_b], normalize_embeddings=True)
    sim = float(cosine_similarity([embs[0]], [embs[1]])[0][0])
    return round(np.clip(sim, 0.0, 1.0), 4)


def batch_semantic_similarity(texts_a: list, texts_b: list) -> list:
    """
    Compute semantic similarity for a batch of text pairs.
    Returns list of floats.
    """
    assert len(texts_a) == len(texts_b)
    embedder = get_embedder()
    embs_a = embedder.encode(texts_a, normalize_embeddings=True)
    embs_b = embedder.encode(texts_b, normalize_embeddings=True)
    sims = [
        round(float(np.clip(cosine_similarity([a], [b])[0][0], 0.0, 1.0)), 4)
        for a, b in zip(embs_a, embs_b)
    ]
    return sims


# ── Combined evaluation report ────────────────────────────────────────────────
def evaluate_summarization(summaries: list, references: list) -> dict:
    """Full evaluation for summarization: ROUGE + semantic similarity."""
    rouge = batch_rouge(summaries, references)
    sims  = batch_semantic_similarity(summaries, references)
    rouge["semantic_similarity"] = round(float(np.mean(sims)), 4)
    return rouge


def evaluate_paraphrasing(paraphrases: list, originals: list) -> dict:
    """
    Evaluate paraphrasing quality.
    Good paraphrase: high semantic similarity, moderate lexical divergence.
    """
    sims  = batch_semantic_similarity(paraphrases, originals)
    rouge = batch_rouge(paraphrases, originals)
    return {
        "avg_semantic_similarity": round(float(np.mean(sims)), 4),
        "avg_rouge1":              rouge["rouge1"],
        "avg_rouge2":              rouge["rouge2"],
        "avg_rougeL":              rouge["rougeL"],
    }


if __name__ == "__main__":
    h = "The cat sat on the mat."
    r = "A cat was sitting on the mat."
    print("ROUGE:", rouge_scores(h, r))
    print("Semantic sim:", semantic_similarity(h, r))