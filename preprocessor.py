"""
preprocessor.py
Text preprocessing pipeline used across all three DocTrust modules.
"""

import re
import unicodedata


# ── constants ─────────────────────────────────────────────────────────────────
MIN_WORDS_SUMMARIZE  = 20
MIN_WORDS_PARAPHRASE = 5
MAX_CHARS_DEFAULT    = 3000
MAX_CHARS_PARAPHRASE = 512


# ── core normalisation ────────────────────────────────────────────────────────
def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def remove_urls(text: str) -> str:
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def remove_excessive_punctuation(text: str) -> str:
    text = re.sub(r'([.!?,;])\1+', r'\1', text)
    return text


def fix_punctuation_spacing(text: str) -> str:
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])([^\s\d])', r'\1 \2', text)
    return text


# ── pipeline ──────────────────────────────────────────────────────────────────
def preprocess(
    text: str,
    max_chars: int = MAX_CHARS_DEFAULT,
    remove_urls_flag: bool = True,
) -> str:
    """
    Full preprocessing pipeline:
    1. Unicode normalisation
    2. URL removal (optional)
    3. Whitespace normalisation
    4. Excessive punctuation removal
    5. Punctuation spacing fix
    6. Hard truncation to max_chars
    """
    text = normalize_unicode(text)
    if remove_urls_flag:
        text = remove_urls(text)
    text = normalize_whitespace(text)
    text = remove_excessive_punctuation(text)
    text = fix_punctuation_spacing(text)
    text = normalize_whitespace(text)
    return text[:max_chars]


# ── validation ────────────────────────────────────────────────────────────────
def validate(text: str, task: str = "summarize") -> tuple:
    """
    Returns (is_valid: bool, error_message: str).
    task: 'summarize' | 'paraphrase' | 'plagiarism'
    """
    cleaned = preprocess(text)
    word_count = len(cleaned.split())

    if task == "summarize":
        if word_count < MIN_WORDS_SUMMARIZE:
            return False, f"Need at least {MIN_WORDS_SUMMARIZE} words (got {word_count})."
    elif task in ("paraphrase", "plagiarism"):
        if word_count < MIN_WORDS_PARAPHRASE:
            return False, f"Need at least {MIN_WORDS_PARAPHRASE} words (got {word_count})."

    return True, ""


# ── batch preprocessing ───────────────────────────────────────────────────────
def preprocess_batch(texts: list, max_chars: int = MAX_CHARS_DEFAULT) -> list:
    return [preprocess(t, max_chars=max_chars) for t in texts]


if __name__ == "__main__":
    sample = "  This is   a test.   Visit https://example.com for more!!!   "
    print(repr(preprocess(sample)))
    print(validate(sample, task="summarize"))