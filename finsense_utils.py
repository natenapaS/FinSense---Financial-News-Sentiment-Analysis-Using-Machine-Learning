"""Shared helpers for FinSense (used by FinSense.ipynb and app.py).

`clean_text` is passed to TfidfVectorizer(preprocessor=...), so it must live in an
importable module for the saved joblib pipeline to load in app.py.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

LABELS = ["negative", "neutral", "positive"]

# Words that carry sentiment/direction in financial news and must NOT be removed
KEEP_WORDS = {
    "not", "no", "nor", "never", "cannot", "against",
    "up", "down", "above", "below", "over", "under",
    "less", "more", "most", "few", "least",
}
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) - KEEP_WORDS)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_SPACE_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(A-Z0-9])")
_ABBREV_RE = re.compile(r"\b(?:inc|ltd|corp|co|plc|mr|ms|mrs|dr|vs|approx|no|e\.g|i\.e)\.$", re.IGNORECASE)


def clean_text(text):
    """Lowercase, drop URLs, map numbers to a 'num' token, strip punctuation."""
    text = str(text).lower()
    text = _URL_RE.sub(" ", text)
    text = text.replace("n't", " not")
    text = _NUM_RE.sub(" num ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def split_sentences(text):
    """Split a news passage into sentences (the model was trained on single sentences).

    Breaks on newlines and on . ! ? followed by a capital/digit/quote, but not after
    common abbreviations such as "Corp." or "Inc.".
    """
    sentences = []
    for line in str(text).splitlines():
        for part in _SENT_SPLIT_RE.split(line.strip()):
            if not part:
                continue
            if sentences and _ABBREV_RE.search(sentences[-1]) and not sentences[-1].endswith("\n"):
                sentences[-1] += " " + part
            else:
                sentences.append(part)
        if sentences:
            sentences[-1] += "\n"  # line ends are hard breaks
    return [s.strip() for s in sentences if s.strip()]


def predict_with_confidence(model, texts):
    """Return (labels, probability matrix) for any fitted sklearn classifier pipeline.

    Models without predict_proba (e.g. LinearSVC) get a softmax over decision_function.
    """
    texts = list(texts)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(texts)
    else:
        scores = np.asarray(model.decision_function(texts), dtype=float)
        scores -= scores.max(axis=1, keepdims=True)
        proba = np.exp(scores)
        proba /= proba.sum(axis=1, keepdims=True)
    classes = np.asarray(model.classes_)
    return classes[proba.argmax(axis=1)], proba
