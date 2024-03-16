# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("iris.csv")

# Define features (X) and target variable (y)
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
target = 'Species'

X = df[features]

# Convert target classes to numeric format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[target])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K-Nearest Neighbors (KNN) Model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)



# Invert the label encoding for evaluation
y_test_original = label_encoder.inverse_transform(y_test)
knn_predictions_original = label_encoder.inverse_transform(knn_predictions)


# Evaluate the performance of each model
def evaluate_model(predictions, model_name):
    accuracy = accuracy_score(y_test_original, predictions)
    report = classification_report(y_test_original, predictions)
    print(f"{model_name} Model:")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", report)
    print()

# Evaluate KNN Model
evaluate_model(knn_predictions_original, "K-Nearest Neighbors (KNN)")

