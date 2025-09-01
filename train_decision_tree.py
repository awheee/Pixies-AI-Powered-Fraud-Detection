import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
train_data = pd.read_csv("fraudTrain.csv")
test_data = pd.read_csv("fraudTest.csv")
train_data = train_data.sample(100000, random_state=42)

# Features
selected_columns = [
    "amt", "merchant", "category", "lat", "long",
    "city_pop", "unix_time", "merch_lat", "merch_long"
]
X_train = train_data[selected_columns]
y_train = train_data["is_fraud"]

X_test = test_data[selected_columns]
y_test = test_data["is_fraud"]

# One-hot encoding
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

# Model
dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")
dt.fit(X_train, y_train)

# Predictions
y_pred = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Save model
joblib.dump(dt, "decision_tree_model.pkl")
print("Decision Tree model saved as decision_tree_model.pkl")
