import pandas as pd
from apyori import apriori

# Example dataset
dataset = [
    ['milk', 'bread', 'eggs'],
    ['milk', 'bread'],
    ['milk', 'eggs'],
    ['bread', 'eggs'],
    ['milk', 'bread', 'eggs', 'butter']
]

# Apply Apriori algorithm
rules = apriori(dataset, 
                min_support=0.1, 
                min_confidence=0.1, 
                min_lift=1.2, 
                min_length=2)

# Convert generator to list
results = list(rules)
print(results)
# Display results
for item in results:
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])
    print("Support:", item.support)
    print("Confidence:", item.ordered_statistics[0].confidence)
    print("Lift:", item.ordered_statistics[0].lift)
    print("===============")
