import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_probability),
    }

    return metrics, y_pred


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION - RANDOM FOREST")
    print("=" * 70)

    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")

    y_train = pd.read_csv(
        f"{DATA_DIR}/y_train.csv"
    ).squeeze()

    y_test = pd.read_csv(
        f"{DATA_DIR}/y_test.csv"
    ).squeeze()

    print("\nTraining shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    metrics, y_pred = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\n" + "=" * 70)
    print("RANDOM FOREST RESULTS")
    print("=" * 70)

    for metric, value in metrics.items():
        print(f"{metric:10s}: {value:.4f}")

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

    model_path = f"{MODEL_DIR}/random_forest.joblib"

    joblib.dump(model, model_path)

    print(f"\nModel saved to: {model_path}")
    print("\nRandom Forest completed successfully.")


if __name__ == "__main__":
    main()
