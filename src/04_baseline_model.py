import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
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


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION - BASELINE MODEL")
    print("=" * 70)

    # Load processed data
    X_train = pd.read_csv(
        f"{DATA_DIR}/X_train.csv"
    )

    X_test = pd.read_csv(
        f"{DATA_DIR}/X_test.csv"
    )

    y_train = pd.read_csv(
        f"{DATA_DIR}/y_train.csv"
    ).squeeze()

    y_test = pd.read_csv(
        f"{DATA_DIR}/y_test.csv"
    ).squeeze()

    print("\nTraining shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    # Logistic Regression baseline
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probability)

    print("\n" + "=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Churn", "Churn"]
        )
    )

    # Save model
    model_path = f"{MODEL_DIR}/logistic_regression.joblib"

    joblib.dump(
        model,
        model_path
    )

    print(f"\nModel saved to: {model_path}")
    print("\nBaseline model completed successfully.")


if __name__ == "__main__":
    main()
