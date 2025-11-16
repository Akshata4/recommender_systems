import pandas as pd

# --- Paths ---
csv_path = "data/DCOILWTICO.csv"
out_path = "data/DCOILWTICO_weekly_hopping_mean_max.csv"

# --- Load & clean ---
df = pd.read_csv(csv_path)
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
# FRED uses '.' for missing values; coerce to NaN
df["DCOILWTICO"] = pd.to_numeric(df["DCOILWTICO"].replace({".": None}), errors="coerce")
df = df.dropna(subset=["DATE"]).sort_values("DATE").set_index("DATE")

# --- Hopping (tumbling) weekly window ---
# hop = window = 1 week; Sunday-ending weeks match common FRED weekly convention
weekly = df["DCOILWTICO"].resample("W-SUN").agg(["mean", "max"]).dropna(how="all")

# --- Save ---
weekly.to_csv(out_path, index_label="week_end_date")

print("Saved weekly hopping stats to:", out_path)
print(weekly.head())
