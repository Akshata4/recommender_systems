from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "spam_filter" / "data"
DICT = ROOT /"spam_filter" / "spam_dict" / "list.txt"
LABELS = ROOT / "spam_filter" / "work" / "labels.csv"

def main():
    print(f"[INFO] Project root: {ROOT}")
    # 1) Check labels.csv
    if not LABELS.exists():
        raise SystemExit(f"[ERROR] labels.csv not found at: {LABELS}")
    df = pd.read_csv(LABELS)
    required_cols = {"filename", "label"}
    if not required_cols.issubset(df.columns):
        raise SystemExit(f"[ERROR] labels.csv must have columns: {required_cols}")
    print(f"[OK] Loaded labels.csv with {len(df)} rows")

    # 2) Check each file in labels exists
    missing = []
    for rel_path in df["filename"]:
        p = DATA / rel_path  # e.g., data/spam/Document1.pdf
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("[ERROR] Missing files:")
        for m in missing:
            print("   -", m)
        raise SystemExit("[FAIL] Fix missing files listed above.")
    else:
        print("[OK] All labeled files exist.")

    # 3) Check spam dictionary
    if not DICT.exists():
        print(f"[WARN] Spam dictionary not found at {DICT}. You can proceed, but TF-IDF will not be constrained.")
    else:
        n_terms = sum(1 for _ in open(DICT, "r", encoding="utf-8") if _.strip())
        print(f"[OK] Spam dictionary found with ~{n_terms} terms")

    # 4) Print small summary
    print("\n[SUMMARY]")
    print(df.groupby('label').size())

if __name__ == "__main__":
    main()
