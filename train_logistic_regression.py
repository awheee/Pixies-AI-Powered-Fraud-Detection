import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
train_data = pd.read_csv("fraudTrain.csv")
test_data = pd.read_csv("fraudTest.csv")

# Sample to speed up training
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
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Save model
joblib.dump(log_reg, "logistic_regression_model.pkl")
print("Logistic Regression model saved as logistic_regression_model.pkl")
