# apriori_next_item_prediction_final.py
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import MultiLabelBinarizer

# =========================
# TRAINING PHASE
# =========================
train_df = pd.read_csv("data/TRAIN-ARULES.csv")

# Normalize column names
train_df.columns = [c.strip().lower().replace(" ", "_") for c in train_df.columns]

# Group items by order id → transactions
transactions = train_df.groupby("order_id")["product_name"].apply(list).reset_index()

# One-hot encode items
mlb = MultiLabelBinarizer()
train_encoded = pd.DataFrame(
    mlb.fit_transform(transactions["product_name"]),
    columns=mlb.classes_,
    index=transactions["order_id"]
)

# Apply Apriori
min_support = 0.0045
frequent_itemsets = apriori(train_encoded, min_support=min_support, use_colnames=True)
print(f"Frequent itemsets found: {len(frequent_itemsets)}")

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values(by="lift", ascending=False)
print(f"Generated {len(rules)} association rules.")

# Save for reference
frequent_itemsets.to_csv("data/frequent_itemsets.csv", index=False)
rules.to_csv("data/association_rules.csv", index=False)

# =========================
# TEST / PREDICTION PHASE
# =========================
test_df = pd.read_csv("data/testarules.csv")

# Each row = a basket of items
test_transactions = []
for i in range(len(test_df)):
    basket = [str(item).strip() for item in test_df.iloc[i].dropna().tolist()]
    test_transactions.append(basket)

print(f"\nLoaded {len(test_transactions)} test baskets")

predictions = []

# Loop through test baskets
for idx, basket in enumerate(test_transactions):
    basket_set = set(basket)
    possible_next_items = []

    # Match rules where antecedents are subset of basket
    for _, rule in rules.iterrows():
        antecedent = set(rule["antecedents"])
        consequent = set(rule["consequents"])

        if antecedent.issubset(basket_set) and not consequent.issubset(basket_set):
            for item in consequent:
                possible_next_items.append((item, rule["confidence"], rule["lift"]))

    if possible_next_items:
        # sort by confidence and lift
        sorted_items = sorted(possible_next_items, key=lambda x: (x[1], x[2]), reverse=True)
        top_items = sorted_items[:3]  # top 3 predictions
        pred_items = [i[0] for i in top_items]
        predictions.append({
            "test_basket_id": idx + 1,
            "basket_items": list(basket_set),
            "predicted_items": pred_items,
            "confidence_lift": [(i[1], i[2]) for i in top_items]
        })
    else:
        predictions.append({
            "test_basket_id": idx + 1,
            "basket_items": list(basket_set),
            "predicted_items": [],
            "confidence_lift": []
        })

# Convert predictions to DataFrame
pred_df = pd.DataFrame(predictions)
pred_df.to_csv("data/next_item_predictions.csv", index=False)

print("\n✅ Predictions saved to 'next_item_predictions.csv'")
print(pred_df.head())
