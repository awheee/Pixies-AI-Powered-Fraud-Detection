# evaluate.py
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import joblib

sns.set(style="whitegrid")

os.makedirs("plots", exist_ok=True)

def main():
    # prefer to use predictions.csv created by train.py
    if not os.path.exists("predictions.csv"):
        print("predictions.csv not found. Run train.py first or place predictions.csv in repo root.")
        return

    df = pd.read_csv("predictions.csv")
    y = df['actual']
    y_pred = df['predicted']
    y_prob = df['prob'] if 'prob' in df.columns else None

    print("Classification report:")
    print(classification_report(y, y_pred, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Fraud","Fraud"], yticklabels=["Not Fraud","Fraud"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("plots/confusion_matrix.png", bbox_inches='tight')
    plt.close()
    print("Saved plots/confusion_matrix.png")

    # ROC & PR if probabilities exist
    if y_prob is not None:
        roc_auc = roc_auc_score(y, y_prob)
        fpr, tpr, _ = roc_curve(y, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
        plt.plot([0,1], [0,1], "--", alpha=0.5)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig("plots/roc_curve.png", bbox_inches='tight')
        plt.close()
        print("Saved plots/roc_curve.png")

        precision, recall, _ = precision_recall_curve(y, y_prob)
        ap = average_precision_score(y, y_prob)
        plt.figure()
        plt.plot(recall, precision, label=f"AP = {ap:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.savefig("plots/pr_curve.png", bbox_inches='tight')
        plt.close()
        print("Saved plots/pr_curve.png")
    else:
        print("No probability column found; ROC/PR plots skipped.")

    # Feature importances (if model exists)
    if os.path.exists("rf_model.joblib"):
        model = joblib.load("rf_model.joblib")
        features = getattr(model, "feature_names_in_", None)
        if features is None:
            print("Model does not have feature_names_in_. If you trained with pandas DataFrame and sklearn >=1.0 this is usually present.")
        else:
            importances = model.feature_importances_
            fi = pd.Series(importances, index=features).sort_values(ascending=True)
            plt.figure(figsize=(6,6))
            fi.tail(15).plot(kind='barh')
            plt.title("Top 15 Feature Importances")
            plt.tight_layout()
            plt.savefig("plots/feature_importances.png", bbox_inches='tight')
            plt.close()
            print("Saved plots/feature_importances.png")
    else:
        print("rf_model.joblib not found; feature importances skipped.")

if __name__ == "__main__":
    main()
