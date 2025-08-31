# inference.py
import pandas as pd
import joblib
import os

def main():
    model_path = "rf_model.joblib"
    input_csv = "new_transactions.csv"   # change this to your input file
    if not os.path.exists(model_path):
        print(f"{model_path} not found. Run train.py first.")
        return
    if not os.path.exists(input_csv):
        print(f"{input_csv} not found. Put a CSV with same selected features in repo root.")
        return

    model = joblib.load(model_path)
    df = pd.read_csv(input_csv, on_bad_lines='skip')
    # selected columns (same as training script)
    selected_columns = [
        "amt", "merchant", "category", "lat", "long",
        "city_pop", "unix_time", "merch_lat", "merch_long"
    ]
    X = df[selected_columns].copy()
    X = pd.get_dummies(X, drop_first=True)
    # align with model input columns (best-effort)
    # If you have an X_train saved, use it. Here we attempt to align with model.feature_names_in_
    model_feats = getattr(model, "feature_names_in_", None)
    if model_feats is not None:
        for f in model_feats:
            if f not in X.columns:
                X[f] = 0
        X = X[model_feats]
    preds = model.predict(X)
    proba = model.predict_proba(X)[:,1] if hasattr(model, "predict_proba") else None
    df_out = df.copy()
    df_out['predicted'] = preds
    if proba is not None:
        df_out['prob'] = proba
    df_out.to_csv("inference_predictions.csv", index=False)
    print("Saved inference_predictions.csv")

if __name__ == "__main__":
    main()
