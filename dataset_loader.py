"""
dataset_loader.py
Loads and explores a sample NLP dataset for demonstrating the DocTrust pipeline.
Uses CNN/DailyMail (summarization) and a small custom set for paraphrase/plagiarism.
"""

from datasets import load_dataset
import pandas as pd


CNN_SAMPLE_SIZE = 50


def load_cnn_sample(n: int = CNN_SAMPLE_SIZE) -> pd.DataFrame:
    """
    Load n samples from CNN/DailyMail test split.
    Returns a DataFrame with columns: ['id', 'article', 'highlights'].
    """
    print(f"[DatasetLoader] Loading CNN/DailyMail ({n} samples) ...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split=f"test[:{n}]", trust_remote_code=True)
    df = pd.DataFrame({
        "id":         range(len(ds)),
        "article":    ds["article"],
        "highlights": ds["highlights"],
    })
    print(f"[DatasetLoader] Loaded {len(df)} samples.")
    return df


def explore_dataset(df: pd.DataFrame) -> dict:
    """
    Print basic statistics and return a stats dict.
    """
    article_lengths  = df["article"].str.split().str.len()
    summary_lengths  = df["highlights"].str.split().str.len()

    stats = {
        "num_samples":          len(df),
        "avg_article_words":    round(article_lengths.mean(), 1),
        "avg_summary_words":    round(summary_lengths.mean(), 1),
        "avg_compression_ratio": round(
            (article_lengths / summary_lengths.replace(0, 1)).mean(), 2
        ),
    }

    print("\n── Dataset Statistics ──────────────────────────────")
    for k, v in stats.items():
        print(f"  {k:<28}: {v}")
    print("────────────────────────────────────────────────────\n")
    return stats


PARAPHRASE_SAMPLES = [
    {
        "original": "Deep learning models require large amounts of labeled training data to achieve high performance.",
        "reference": "Neural networks with many layers need extensive annotated datasets to perform well.",
    },
    {
        "original": "Plagiarism detection systems identify similarities between submitted texts and reference corpora.",
        "reference": "Tools for detecting academic dishonesty compare submitted work against databases of existing content.",
    },
    {
        "original": "Natural language processing enables computers to understand and generate human language.",
        "reference": "NLP gives machines the ability to read, interpret, and produce text in human languages.",
    },
    {
        "original": "Transfer learning allows models pre-trained on large corpora to be fine-tuned on smaller datasets.",
        "reference": "Models initially trained on vast data can be adapted to specialized tasks with limited examples.",
    },
    {
        "original": "Text summarization condenses long documents while retaining the most important information.",
        "reference": "Summarization systems shorten lengthy texts by preserving their key ideas and main points.",
    },
]


def get_paraphrase_samples() -> list:
    return PARAPHRASE_SAMPLES


if __name__ == "__main__":
    df = load_cnn_sample(10)
    explore_dataset(df)
    print("Paraphrase samples:", len(get_paraphrase_samples()))