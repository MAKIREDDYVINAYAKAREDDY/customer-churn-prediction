import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


DATA_DIR = "data/processed"
MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)


def evaluate_at_threshold(model, X, y, threshold):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1": f1_score(y, predictions, zero_division=0),
    }


def main():
    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION - FINAL MODEL SELECTION")
    print("=" * 70)

    X_test = pd.read_csv(
        f"{DATA_DIR}/X_test.csv"
    )

    y_test = pd.read_csv(
        f"{DATA_DIR}/y_test.csv"
    ).squeeze()

    # Load tuned XGBoost
    model_path = f"{MODEL_DIR}/xgboost_tuned.joblib"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "xgboost_tuned.joblib not found. Run V8 first."
        )

    model = joblib.load(model_path)

    probabilities = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(f"\nROC-AUC: {roc_auc:.4f}")

    # ---------------------------------------------------------
    # Test multiple probability thresholds
    # ---------------------------------------------------------

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]

    threshold_results = []

    for threshold in thresholds:
        result = evaluate_at_threshold(
            model,
            X_test,
            y_test,
            threshold
        )

        threshold_results.append(result)

    threshold_df = pd.DataFrame(
        threshold_results
    )

    print("\n" + "=" * 70)
    print("THRESHOLD COMPARISON")
    print("=" * 70)

    print(
        threshold_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    threshold_df.to_csv(
        f"{REPORT_DIR}/threshold_comparison.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Select threshold based on F1
    # ---------------------------------------------------------

    best_row = threshold_df.loc[
        threshold_df["F1"].idxmax()
    ]

    best_threshold = float(
        best_row["Threshold"]
    )

    print("\n" + "=" * 70)
    print("BEST THRESHOLD")
    print("=" * 70)

    print(
        f"\nThreshold : {best_threshold:.2f}"
    )
    print(
        f"Accuracy  : {best_row['Accuracy']:.4f}"
    )
    print(
        f"Precision : {best_row['Precision']:.4f}"
    )
    print(
        f"Recall    : {best_row['Recall']:.4f}"
    )
    print(
        f"F1 Score  : {best_row['F1']:.4f}"
    )

    # ---------------------------------------------------------
    # Final predictions
    # ---------------------------------------------------------

    final_predictions = (
        probabilities >= best_threshold
    ).astype(int)

    print("\n" + "=" * 70)
    print("FINAL MODEL PERFORMANCE")
    print("=" * 70)

    print(
        f"\nAccuracy : "
        f"{accuracy_score(y_test, final_predictions):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_test, final_predictions):.4f}"
    )

    print(
        f"Recall   : "
        f"{recall_score(y_test, final_predictions):.4f}"
    )

    print(
        f"F1 Score : "
        f"{f1_score(y_test, final_predictions):.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{roc_auc:.4f}"
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            final_predictions
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            final_predictions,
            target_names=[
                "No Churn",
                "Churn"
            ]
        )
    )

    # ---------------------------------------------------------
    # Plot threshold vs metrics
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        threshold_df["Threshold"],
        threshold_df["Precision"],
        marker="o",
        label="Precision"
    )

    plt.plot(
        threshold_df["Threshold"],
        threshold_df["Recall"],
        marker="o",
        label="Recall"
    )

    plt.plot(
        threshold_df["Threshold"],
        threshold_df["F1"],
        marker="o",
        label="F1"
    )

    plt.xlabel("Probability Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Optimization")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(
        f"{REPORT_DIR}/threshold_optimization.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------------------------------------------------
    # Save final model configuration
    # ---------------------------------------------------------

    final_config = {
        "model": "XGBoost",
        "model_file": "xgboost_tuned.joblib",
        "threshold": best_threshold,
        "roc_auc": roc_auc,
        "accuracy": float(best_row["Accuracy"]),
        "precision": float(best_row["Precision"]),
        "recall": float(best_row["Recall"]),
        "f1": float(best_row["F1"]),
    }

    joblib.dump(
        final_config,
        f"{MODEL_DIR}/final_model_config.joblib"
    )

    print(
        "\nFinal configuration saved to:"
        " models/final_model_config.joblib"
    )

    print("\nV9 completed successfully.")


if __name__ == "__main__":
    main()
