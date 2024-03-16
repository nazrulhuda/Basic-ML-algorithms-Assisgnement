# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("revenue vs iras collection.csv")

# Check the available unique values for "financial_year" to see the range
print("Unique values for 'financial_year':", df['financial_year'].unique())

# Assuming 'financial_year' is a continuous variable, let's use it as a feature
features = ['financial_year', 'total_gov_operating_rev']
target = 'iras_collection'

# Extract relevant features and target variable
X = df[features]
y = df[target]

# Initialize Logistic Regression model
logreg_model = LogisticRegression()

# Train the model on the entire dataset
logreg_model.fit(X, y)

# Predict iras_collection for the given values
predicted_collection = logreg_model.predict([[2023, 80000000]])

# Print the predicted iras_collection
print("Predicted iras_collection:", predicted_collection[0])
