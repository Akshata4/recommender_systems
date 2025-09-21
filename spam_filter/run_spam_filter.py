# run_spam_filter_simple.py
# Minimal, single-file version: prints outputs instead of saving.

from pathlib import Path
import re
import pdfplumber
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict, LeaveOneOut, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

# ----------- Helpers -----------

WORD_RX = re.compile(r"[A-Za-z']+")

def read_pdf_text(pdf_path: Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages)

def basic_clean(text: str) -> str:
    toks = WORD_RX.findall((text or "").lower())
    return " ".join(toks)

def load_spam_vocab(path: Path):
    if not path.exists():
        return None
    vocab = []
    for line in open(path, "r", encoding="utf-8"):
        w = line.strip().lower()
        if w:
            vocab.append(w)
    return sorted(set(vocab))

# ----------- Main -----------

def main():
    # DATA = Path("spam_filter/data")               # folder with PDFs
    # LABELS = Path("spam_filter/work/labels.csv")       # must have filename,label
    # DICT = Path("spam_filter/spam_dict/list.txt") # spam dictionary

    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT / "spam_filter" / "data"
    LABELS = ROOT / "spam_filter" / "work" / "labels.csv"
    DICT = ROOT / "spam_filter" / "spam_dict" / "list.txt"
    
    # Load labels
    if not LABELS.exists():
        raise SystemExit("labels.csv not found — create one with filename,label", LABELS)
    df = pd.read_csv(LABELS)
    if not {"filename","label"}.issubset(df.columns):
        raise SystemExit("labels.csv must have columns: filename,label")
    
    # Extract text
    texts = []
    for rel in df["filename"]:
        p = DATA / rel
        if not p.exists():
            raise SystemExit(f"Missing file: {p}")
        raw = read_pdf_text(p)
        clean = basic_clean(raw)
        texts.append(clean)
    df["clean_text"] = texts
    
    # Load dictionary vocab
    vocab = load_spam_vocab(DICT)
    if vocab:
        print(f"[INFO] Loaded spam dictionary with {len(vocab)} terms")
    else:
        print("[INFO] No dictionary found — using full vocab")
    
    # TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1,2),
        vocabulary=vocab
    )
    X = vectorizer.fit_transform(df["clean_text"])
    print(f"[TFIDF] Shape: {X.shape}")
    
    # Baseline spam score = sum of dictionary TF-IDF weights
    scores = np.asarray(X.sum(axis=1)).ravel()
    df["spam_score"] = scores
    print("\nBaseline TF-IDF spam scores:")
    print(df[["filename","spam_score"]])
    
    labels = df["label"].str.lower()
    if labels.nunique() >= 2:
        # ---- Supervised ----
        print("\n[MODE] Supervised (spam vs not_spam)")
        y = labels.map({"not_spam":0, "spam":1}).values
        
        # CV choice
        if len(df) <= 10:
            cv = LeaveOneOut()
        else:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                token_pattern=r"(?u)\b\w+\b",
                ngram_range=(1,2),
                vocabulary=vocab
            )),
            ("clf", MultinomialNB())
        ])
        
        y_pred = cross_val_predict(pipe, df["clean_text"], y, cv=cv)
        print("\nClassification report:")
        print(classification_report(y, y_pred, target_names=["not_spam","spam"]))
        print("Confusion matrix:")
        print(confusion_matrix(y, y_pred))
        
        df["pred"] = np.where(y_pred==1,"spam","not_spam")
        print("\nPer-document predictions:")
        print(df[["filename","label","pred"]])
        
    else:
        # ---- One-class ----
        print("\n[MODE] One-class (spam-only)")
        iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
        iso.fit(X)
        pred = iso.predict(X)   # +1 inlier, -1 outlier
        label_hat = np.where(pred==1,"spam","not_spam")
        scores = iso.decision_function(X)
        
        print("\nIsolationForest decision scores (higher=more spam-like):")
        for fn, sc, lb in zip(df["filename"], scores, label_hat):
            print(f"{fn:20s} score={sc:.3f} → {lb}")

if __name__ == "__main__":
    main()
