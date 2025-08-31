# train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_df(path):
    df = pd.read_csv(path, on_bad_lines='skip')
    df = df.dropna(subset=["is_fraud"])
    return df

def preprocess(df, selected_columns):
    X = df[selected_columns].copy()
    X = pd.get_dummies(X, drop_first=True)
    return X

def main():
    train_path = "fraudTrain.csv"
    test_path = "fraudTest.csv"

    # same columns you used
    selected_columns = [
        "amt", "merchant", "category", "lat", "long",
        "city_pop", "unix_time", "merch_lat", "merch_long"
    ]

    print("Loading data...")
    train = load_df(train_path)
    test = load_df(test_path)

    print("Preprocessing...")
    X_train = preprocess(train, selected_columns)
    X_test = preprocess(test, selected_columns)

    # align columns to avoid mismatch after get_dummies
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    y_train = train["is_fraud"].astype(int)
    y_test = test["is_fraud"].astype(int)

    print("Training RandomForest...")
    model = RandomForestClassifier(n_estimators=200, random_state=42,
                                   class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)

    print("Predicting on test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred, digits=4))

    # Save model (joblib with compression)
    model_fname = "rf_model.joblib"
    joblib.dump(model, model_fname, compress=3)
    print(f"Saved model -> {model_fname}")

    # Save predictions (so evaluate.py can use them)
    out = X_test.copy()
    out['actual'] = y_test.values
    out['predicted'] = y_pred
    if y_proba is not None:
        out['prob'] = y_proba
    out.to_csv("predictions.csv", index=False)
    print("Saved predictions.csv")

if __name__ == "__main__":
    main()
