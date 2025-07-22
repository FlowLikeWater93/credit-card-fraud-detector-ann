import pandas as pd

'''
SOURCE : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data

The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.
The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation.
Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data.
Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'.
Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset.
The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning.
Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC).
Confusion matrix accuracy is not meaningful for unbalanced classification.
'''

# Load dataset
df = pd.read_csv('creditcard.csv')

# Understand the structure of the dataset
print(df.info())
print()
## 284,807 rows
## 31 columns

# Analyze numeric variables
print(df.describe())
print()

# Analyze the distribution of the target (Y)
#Binary feature
## 0 = negative = no fraud
## 1 = positive = fraud
## Highly imbalanced dataset
print(df['Class'].value_counts())

# Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset
# we can omit 'Time' as it does not carry much value and won't help us when pridicting the target class

## NEXT STEPS ##
# Build a binary classifier neural network
# Binary cross entropy loss
# Adam optimizer
# Measure accuracy using F1 score, precison and recall
