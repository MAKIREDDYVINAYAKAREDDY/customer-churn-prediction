import os
import joblib
import pandas as pd

from xgboost import XGBClassifier
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
    print("CUSTOMER CHURN PREDICTION - XGBOOST")
    print("=" * 70)

    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")

    y_train = pd.read_csv(
        f"{DATA_DIR}/y_train.csv"
    ).squeeze()

    y_test = pd.read_csv(
        f"{DATA_DIR}/y_test.csv"
    ).squeeze()

    # Calculate class imbalance ratio
    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()
    scale_pos_weight = negative / positive

    print(f"\nNegative samples: {negative}")
    print(f"Positive samples: {positive}")
    print(f"Scale pos weight: {scale_pos_weight:.4f}")

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining XGBoost...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probability)

    print("\n" + "=" * 70)
    print("XGBOOST RESULTS")
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

    model_path = f"{MODEL_DIR}/xgboost.joblib"

    joblib.dump(model, model_path)

    print(f"\nModel saved to: {model_path}")
    print("\nXGBoost completed successfully.")


if __name__ == "__main__":
    main()
