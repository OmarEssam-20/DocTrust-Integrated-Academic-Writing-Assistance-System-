"""
model_loader.py
Centralized, cached model loading. All models are loaded once and reused.
"""

import functools
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)
from sentence_transformers import SentenceTransformer


# ── device ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ── internal cache ───────────────────────────────────────────────────────────
_cache: dict = {}


def _load_once(key: str, loader_fn):
    """Generic helper – call loader_fn exactly once, then return cached value."""
    if key not in _cache:
        _cache[key] = loader_fn()
    return _cache[key]


# ── summarization ─────────────────────────────────────────────────────────────
# facebook/bart-large-cnn is the gold-standard extractive→abstractive CNN model.
# We use the high-level `summarization` pipeline which returns `summary_text`.
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"


def get_summarizer():
    def _load():
        print(f"[ModelLoader] Loading summarizer ({SUMMARIZATION_MODEL}) …")
        tok = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            SUMMARIZATION_MODEL, torch_dtype=TORCH_DTYPE
        ).to(DEVICE)
        pipe = pipeline(
            "summarization",
            model=mdl,
            tokenizer=tok,
            device=0 if DEVICE == "cuda" else -1,
        )
        print("[ModelLoader] Summarizer ready.")
        return pipe

    return _load_once("summarizer", _load)


# ── paraphrasing ─────────────────────────────────────────────────────────────
# Vamsi/T5_Paraphrase_Paws is a T5-base fine-tuned on PAWS for paraphrasing.
PARAPHRASE_MODEL = "Vamsi/T5_Paraphrase_Paws"


def get_paraphrase_tokenizer():
    def _load():
        print(f"[ModelLoader] Loading paraphrase tokenizer ({PARAPHRASE_MODEL}) …")
        tok = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL)
        print("[ModelLoader] Paraphrase tokenizer ready.")
        return tok

    return _load_once("para_tokenizer", _load)


def get_paraphrase_model():
    def _load():
        print(f"[ModelLoader] Loading paraphrase model ({PARAPHRASE_MODEL}) …")
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            PARAPHRASE_MODEL, torch_dtype=TORCH_DTYPE
        ).to(DEVICE)
        print("[ModelLoader] Paraphrase model ready.")
        return mdl

    return _load_once("para_model", _load)


# ── sentence embeddings ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embedder():
    def _load():
        print(f"[ModelLoader] Loading embedder ({EMBEDDING_MODEL}) …")
        embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        print("[ModelLoader] Embedder ready.")
        return embedder

    return _load_once("embedder", _load)