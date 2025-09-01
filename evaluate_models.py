import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# ======================
# Load dataset
# ======================
print("Loading data...")
train_data = pd.read_csv("fraudTrain.csv")
test_data = pd.read_csv("fraudTest.csv")
train_data = train_data.sample(100000, random_state=42)

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

# ======================
# Load trained models
# ======================
models = {
    "Logistic Regression": joblib.load("logistic_regression_model.pkl"),
    "Decision Tree": joblib.load("decision_tree_model.pkl"),
    "Random Forest": joblib.load("random_forest_model.pkl"),
    "XGBoost": joblib.load("xgboost_model.pkl")
}

# ======================
# Evaluate all models
# ======================
summary = []

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Fraud (class 1)
    prec_fraud = report["1"]["precision"]
    recall_fraud = report["1"]["recall"]
    f1_fraud = report["1"]["f1-score"]

    # Macro scores
    macro_prec = report["macro avg"]["precision"]
    macro_recall = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]

    summary.append([
        name, acc, prec_fraud, recall_fraud, f1_fraud,
        macro_prec, macro_recall, macro_f1
    ])

# ======================
# Print Summary
# ======================
results_df = pd.DataFrame(summary, columns=[
    "Model", "Accuracy",
    "Fraud Precision", "Fraud Recall", "Fraud F1",
    "Macro Precision", "Macro Recall", "Macro F1"
])

print("\n====== Model Comparison Summary ======")
print(results_df.to_string(index=False))

# Save summary CSV
results_df.to_csv("model_comparison_results.csv", index=False)
print("\nSaved summary to model_comparison_results.csv")
