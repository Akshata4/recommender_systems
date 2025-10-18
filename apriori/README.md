# Market Basket Item - Apriori Algorithm Dataset from real-world retailer

This dataset contains a collection of market basket transactions from a real-world retailer. Each transaction represents a single customer's purchase and includes a list of items bought together. The dataset is ideal for applying the Apriori algorithm to discover frequent itemsets and association rules, which can help in understanding customer purchasing behavior and improving marketing strategies.

## Dataset Description
Each row in the dataset corresponds to a single item in a trasaction, with items separated by commas. 
We have grouped all items bought in a single transaction together using a unique transaction ID.

## Usage
To use this dataset with the Apriori algorithm, you can follow these steps:
1. Load the dataset into Pandas Dataframe.
2. Preprocess the data to create a list of transactions.
3. Apply the Apriori algorithm to find frequent itemsets.
4. Generate association rules from the frequent itemsets.
5. Analyze the results to gain insights into customer purchasing patterns.

## Code Description
The provided code demonstrates how to implement the Apriori algorithm using the `mlxtend` library in Python. It includes steps for loading the dataset, preprocessing the data, applying the Apriori algorithm, and generating association rules.  
 
Refer main.py for the complete implementation.

## Requirements
- Python 3.x
- Pandas
- mlxtend
- NumPy

## References
- [mlxtend Documentation](http://rasbt.github.io/mlxtend/)
- [Apriori Algorithm on Wikipedia](https://en.wikipedia.org/wiki/Apriori_algorithm) 
