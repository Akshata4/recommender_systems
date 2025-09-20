# tfidf_rule_spam.py
# Simple TF-IDF-based spam classifier using a provided spam dictionary.
# No ML model training required.

from pathlib import Path
import pdfplumber
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ---- Config ----
DATA_DIR = Path("data")           # Document1.pdf ... Document6.pdf
SPAM_DICT_FILE = Path("spam_dict/list.txt")  # one term per line (from Spam Dictionary.pdf)
OUTPUT_CSV = Path("work/tfidf_rule_results.csv")

# Choose how to set the threshold:
#   "fixed": use a fixed numeric cutoff; good if you know you'll have ham docs too
#   "percentile": mark the top X% highest-scoring docs as spam (works even if all docs are spam)
THRESHOLD_MODE = "fixed"          # "fixed" or "percentile"
FIXED_THRESHOLD = 0.3             # tweak this if needed (depends on data length & dict size)
PERCENTILE = 50                   # e.g., top 50% scores = spam (only used if THRESHOLD_MODE="percentile")

def read_pdf_text(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

# ---- Load data ----
pdf_files = sorted(p for p in DATA_DIR.glob("Document*.pdf"))
if not pdf_files:
    raise SystemExit(f"No PDFs found under {DATA_DIR}")

texts = [read_pdf_text(p) for p in pdf_files]

# ---- Load spam dictionary ----
if not SPAM_DICT_FILE.exists():
    raise SystemExit(f"Spam dictionary text file not found: {SPAM_DICT_FILE}\n"
                     "Create it from Spam Dictionary.pdf (one term per line).")
with open(SPAM_DICT_FILE, "r", encoding="utf-8") as f:
    spam_terms = [w.strip().lower() for w in f if w.strip()]
spam_terms = list(dict.fromkeys(spam_terms))  # unique, keep order

if not spam_terms:
    raise SystemExit("Spam dictionary is empty.")

# ---- TF-IDF restricted to spam dictionary ----
# We restrict the vocabulary to the spam terms, so TF-IDF only scores those words.
vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b",
    ngram_range=(1, 2),          # allow unigrams & bigrams; acceptable for a dictionary
    vocabulary=spam_terms        # <-- key point: only score spam-dictionary terms
)

X = vectorizer.fit_transform(texts)  # shape: (n_docs, n_spam_terms)

# ---- Score each document ----
# A very simple score: sum of TF-IDF weights for all spam terms in the doc.
scores = np.asarray(X.sum(axis=1)).ravel()

# ---- Decide threshold & classify ----
if THRESHOLD_MODE == "fixed":
    thresh = FIXED_THRESHOLD
elif THRESHOLD_MODE == "percentile":
    thresh = np.percentile(scores, PERCENTILE)
else:
    raise ValueError("THRESHOLD_MODE must be 'fixed' or 'percentile'.")

labels = np.where(scores >= thresh, "SPAM", "NOT SPAM")

# ---- Save & print results ----
df = pd.DataFrame({
    "filename": [p.name for p in pdf_files],
    "tfidf_spam_score": scores,
    "threshold": [thresh]*len(scores),
    "label": labels
}).sort_values("tfidf_spam_score", ascending=False)

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print(df.to_string(index=False))
print(f"\nSaved results → {OUTPUT_CSV}")
