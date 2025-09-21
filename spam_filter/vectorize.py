# src/vectorize.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from data_utils import ROOT, load_labeled_texts

DICT_FILE = ROOT / "spam_filter" / "spam_dict" / "list.txt"
CACHE_TEXTS = ROOT / "spam_filter" / "work" / "texts_clean.csv"

def load_spam_vocab():
    if not DICT_FILE.exists():
        return None
    vocab = []
    with open(DICT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                vocab.append(w)
    # dedupe while preserving order
    seen = set()
    uniq = []
    for w in vocab:
        if w not in seen:
            uniq.append(w)
            seen.add(w)
    return uniq

def build_tfidf(texts, restrict_to_spam_dict=True):
    vocab = load_spam_vocab() if restrict_to_spam_dict else None
    vec = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        vocabulary=vocab  # None => full vocab, else restrict to spam dictionary
    )
    X = vec.fit_transform(texts)
    return vec, X

def main():
    # 1) Load & cache texts
    df = load_labeled_texts()
    CACHE_TEXTS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_TEXTS, index=False)
    print(f"[OK] Cached clean texts → {CACHE_TEXTS} ({len(df)} docs)")

    # 2) Build TF-IDF (restricted to spam dict if available)
    vec, X = build_tfidf(df["clean_text"].tolist(), restrict_to_spam_dict=True)
    print(f"[TFIDF] shape = {X.shape}  (docs × terms)")
    if vec.vocabulary_ is not None:
        print(f"[TFIDF] Using spam dictionary terms only: {len(vec.vocabulary_)} terms")
    else:
        print("[TFIDF] Using full vocabulary from the corpus.")

    # 3) Quick peek at top 10 features by idf (lowest idf = most common)
    import numpy as np
    terms = vec.get_feature_names_out()
    idf = vec.idf_
    top_idx = np.argsort(idf)[:10]
    preview = pd.DataFrame({"term": terms[top_idx], "idf": idf[top_idx]})
    print("\n[Preview] 10 most common terms in this vocab:")
    print(preview.to_string(index=False))

if __name__ == "__main__":
    main()
