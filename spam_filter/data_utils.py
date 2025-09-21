# src/data_utils.py
from __future__ import annotations
from pathlib import Path
import re
import pdfplumber
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "spam_filter" / "data"
LABELS = ROOT / "spam_filter" / "work" / "labels.csv"

_WORD = re.compile(r"[A-Za-z']+")  # simple token pattern

def read_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF (concatenate all pages)."""
    with pdfplumber.open(pdf_path) as pdf:
        pages = []
        for pg in pdf.pages:
            txt = pg.extract_text() or ""
            pages.append(txt)
    return "\n".join(pages)

def basic_clean(text: str) -> str:
    """Lowercase, keep word-ish tokens, collapse spaces."""
    tokens = _WORD.findall(text.lower())
    return " ".join(tokens)

def load_labeled_texts() -> pd.DataFrame:
    """
    Returns a DataFrame with columns: filename, label, raw_text, clean_text.
    Looks for files under data/<relative filename>.
    """
    df = pd.read_csv(LABELS)
    rows = []
    for rel in df["filename"]:
        p = DATA / rel
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        raw = read_pdf_text(p)
        clean = basic_clean(raw)
        rows.append({"filename": rel, "label": df.loc[df["filename"] == rel, "label"].iloc[0],
                     "raw_text": raw, "clean_text": clean})
    return pd.DataFrame(rows)
