"""
summarizer.py  ── v4 (Master's thesis grade)
════════════════════════════════════════════
Changes from v3
───────────────
IMPROVEMENT 1 — Smoother sentence joining in _clean_summary()
  The v3 cleaner joined sentences with a plain space. When sentences from
  different chunks are merged, this sometimes produces abrupt or repetitive
  transitions. The new cleaner:
    • Detects and removes leading connector phrases that duplicate the end
      of the previous sentence (e.g. "... the model. The model also ...").
    • Adds a thin normalisation pass to unify punctuation spacing.

IMPROVEMENT 2 — Overlap-aware chunk boundary for map-reduce
  _CHUNK_OVERLAP raised 50 → 80 tokens. This reduces the risk of a sentence
  being split mid-way across two chunks, which caused truncated context and
  incomplete sub-summaries in v3.

IMPROVEMENT 3 — Minimum chunk length guard
  Chunks smaller than 30 tokens are appended to the previous chunk rather
  than being summarised independently. This prevents BART from producing
  degenerate one-phrase summaries for sentence fragments near boundaries.

IMPROVEMENT 4 — Final pass length bounds scaled to merged content
  Previously, final_min was sometimes too low (≈ 40 tokens), allowing the
  consolidation pass to collapse a multi-chunk summary into a single vague
  sentence. New formula:
    final_min = max(60, int(merged_tokens * 0.15))
  ensuring the final summary retains sufficient detail.

No changes to the public API or model loading logic.
"""

import re
from model_loader import get_summarizer

_BART_MAX_INPUT  = 1024   # BART encoder hard limit (tokens)
_CHUNK_TARGET    = 800    # tokens per chunk in map-reduce
_CHUNK_OVERLAP   = 80     # ↑ was 50 — reduces mid-sentence splits
_MIN_CHUNK_TOKENS = 30    # chunks smaller than this are merged with previous


# ── text utilities ────────────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    # Normalise punctuation spacing (remove space before punctuation)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    return text


def _remove_leading_overlap(prev: str, curr: str) -> str:
    """
    If the current sentence starts with words already present at the end of
    the previous sentence, trim the duplicate prefix.
    Operates on the first 6 words of curr vs last 6 words of prev.
    """
    prev_words = re.findall(r'\w+', prev.lower())[-6:]
    curr_words = re.findall(r'\w+', curr.lower())

    # Find longest prefix of curr that matches a suffix of prev
    trim = 0
    for length in range(min(len(prev_words), len(curr_words), 4), 0, -1):
        if curr_words[:length] == prev_words[-length:]:
            trim = length
            break

    if trim == 0:
        return curr

    # Reconstruct curr without the overlapping prefix tokens
    raw_words = curr.split()
    trimmed   = ' '.join(raw_words[trim:]).strip()
    if trimmed:
        trimmed = trimmed[0].upper() + trimmed[1:]
    return trimmed or curr


def _clean_summary(text: str) -> str:
    """
    Post-process summary:
    1. Normalise whitespace and punctuation spacing.
    2. Remove duplicate sentences (order-preserving).
    3. Remove leading word overlap between adjacent sentences.
    4. Drop trailing incomplete sentence (no terminal punctuation).
    5. Capitalise.
    """
    text = _normalize(text)

    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Dedup while preserving order
    seen, deduped = set(), []
    for s in sentences:
        key = s.lower().strip()
        if key not in seen and len(s.strip()) > 4:
            seen.add(key)
            deduped.append(s.strip())

    # Remove leading word overlap at sentence boundaries
    cleaned = []
    for i, sent in enumerate(deduped):
        if i > 0 and cleaned:
            sent = _remove_leading_overlap(cleaned[-1], sent)
        if sent:
            cleaned.append(sent)

    # Drop last sentence if it lacks terminal punctuation
    if cleaned and not re.search(r'[.!?]$', cleaned[-1]):
        cleaned = cleaned[:-1]

    result = ' '.join(cleaned).strip()
    if result:
        result = result[0].upper() + result[1:]
    return result or text   # fallback: return uncleaned text


# ── tokenizer-safe truncation ─────────────────────────────────────────────────
def _truncate(text: str, tokenizer, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


# ── chunker ───────────────────────────────────────────────────────────────────
def _make_chunks(text: str, tokenizer) -> list:
    """
    Split *text* into overlapping token chunks of ≤ _CHUNK_TARGET tokens.
    Chunks smaller than _MIN_CHUNK_TOKENS are merged into the previous chunk.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= _CHUNK_TARGET:
        return [text]

    raw_chunks, start = [], 0
    while start < len(ids):
        end = min(start + _CHUNK_TARGET, len(ids))
        raw_chunks.append(ids[start:end])
        if end == len(ids):
            break
        start = end - _CHUNK_OVERLAP

    # Merge undersized trailing chunks into the previous one
    merged_chunks = []
    for chunk_ids in raw_chunks:
        if merged_chunks and len(chunk_ids) < _MIN_CHUNK_TOKENS:
            merged_chunks[-1] = merged_chunks[-1] + chunk_ids
        else:
            merged_chunks.append(chunk_ids)

    return [
        tokenizer.decode(c, skip_special_tokens=True)
        for c in merged_chunks
    ]


# ── single-pass summarizer ────────────────────────────────────────────────────
def _summarize_passage(text: str, pipe, tokenizer,
                        max_len: int, min_len: int) -> str:
    """Run one BART summarization pass on a passage that fits in the model."""
    text = _truncate(text, tokenizer, _BART_MAX_INPUT)
    result = pipe(
        text,
        max_length=max_len,
        min_length=min_len,
        num_beams=5,
        length_penalty=2.0,
        do_sample=False,
        truncation=True,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )
    return result[0]["summary_text"]


# ── length bounds ─────────────────────────────────────────────────────────────
def _bounds(token_count: int) -> tuple:
    """Return (max_len, min_len) scaled to input token count."""
    if token_count < 80:
        max_len = max(40, token_count - 5)
        min_len = max(15, token_count // 4)
    elif token_count < 400:
        max_len = min(180, max(60, int(token_count * 0.35)))
        min_len = max(25, int(token_count * 0.10))
    else:
        max_len = 180
        min_len = 40
    return max_len, min_len


# ── public API ────────────────────────────────────────────────────────────────
def summarize(text: str) -> str:
    """
    Return a clean abstractive summary of *text* of any length.

    Short texts  → direct pass.
    Medium texts → single BART pass.
    Long texts   → map-reduce (chunk → summarize → merge → final pass).
    """
    pipe      = get_summarizer()
    tokenizer = pipe.tokenizer

    text        = _normalize(text)
    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    max_len, min_len = _bounds(token_count)

    # ── short / medium: single pass ───────────────────────────────────────────
    if token_count <= _CHUNK_TARGET:
        raw = _summarize_passage(text, pipe, tokenizer, max_len, min_len)
        return _clean_summary(raw)

    # ── long: map-reduce ──────────────────────────────────────────────────────
    chunks          = _make_chunks(text, tokenizer)
    chunk_summaries = []

    for chunk in chunks:
        c_tokens       = len(tokenizer.encode(chunk, add_special_tokens=False))
        c_max, c_min   = _bounds(c_tokens)
        s = _summarize_passage(chunk, pipe, tokenizer, c_max, c_min)
        chunk_summaries.append(s)

    merged        = ' '.join(chunk_summaries)
    merged_tokens = len(tokenizer.encode(merged, add_special_tokens=False))

    # Improved final bounds: min is now higher to preserve multi-chunk detail
    final_max = min(200, max(80, int(merged_tokens * 0.45)))
    final_min = max(60,  int(merged_tokens * 0.15))   # ↑ was max(40, 0.12×)

    final = _summarize_passage(merged, pipe, tokenizer, final_max, final_min)
    return _clean_summary(final)